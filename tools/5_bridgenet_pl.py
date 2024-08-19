import sys
import os
import json
import math
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as L
from safetensors import safe_open
from typing import Any, Iterator, Optional, List
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm


model_name = 'gpt2'
model_path = '/kaggle/working/bridgenet.pt'

DEFAULT_CONTEXT_LENGTH = 30

def get_device() -> torch.device:
    device = "cuda" if torch.cuda.is_available() else \
             "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else \
             "cpu"
    return torch.device(device)

class Flickr8kDataset(Dataset):
    def __init__(self, split: str = 'train', data_dir: str = None) -> None:
        super().__init__()
        allowed_splits = ('train', 'val', 'test')
        if split not in allowed_splits:
            raise ValueError(f"Allowed splits are: {','.join(allowed_splits)}")
        self.split = split

        self.data_dir = os.path.join(data_dir, split)
        self.image_embeddings: Optional[torch.Tensor] = None
        self.caption_embeddings: Optional[torch.Tensor] = None
        self.metadata: Optional[List] = None
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self._load_data()
        self._load_image_embeddings()
        self._load_caption_embeddings()

    def is_train(self) -> bool:
        return self.split == 'train'
    
    def is_validation(self) -> bool:
        return self.split == 'val'
    
    def is_test(self) -> bool:
        return self.split == 'test'

    def _load_data(self) -> None:
        metadata_filepath = os.path.join(self.data_dir, 'metadata.json')
        with open(metadata_filepath, 'r') as stream:
            self.metadata = json.load(stream)

    def _load_image_embeddings(self) -> None:
        image_embeddings_filepath = os.path.join(self.data_dir, 'image_embeddings.safetensors')
        with safe_open(image_embeddings_filepath, framework='pt') as f:
            self.image_embeddings = f.get_slice('embeddings')

    def _load_caption_embeddings(self) -> None:
        caption_embeddings_filepath = os.path.join(self.data_dir, 'caption_embeddings.safetensors')
        with safe_open(caption_embeddings_filepath, framework='pt') as f:
            self.caption_embeddings = f.get_slice('embeddings')


    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, index: int) -> Any:
        metadata_item = self.metadata[index]
        caption = metadata_item['caption']
        i = metadata_item['image_embeddings_index']
        j = metadata_item['caption_embeddings_index']
        image_embeddings = self.image_embeddings[i:i+1, :]
        caption_embeddings = self.caption_embeddings[j:j+1, :]
        caption_tokens = self.tokenizer.encode(caption, 
                                               padding='max_length',
                                               truncation=True,
                                               max_length=DEFAULT_CONTEXT_LENGTH, 
                                               return_tensors='pt').squeeze(0)
        return image_embeddings, caption_embeddings, caption_tokens


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        
        assert d_model % n_heads == 0, 'd_model must be divisible by n_heads'
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = F.softmax(attn_scores, dim=-1)
        attns = torch.matmul(attn_probs, V)
        return attns, attn_probs
    
    
    def split_heads(self, x) -> torch.Tensor:
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.n_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x) -> torch.Tensor:
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor=None) -> torch.Tensor:
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attns, _ = self.scaled_dot_product_attention(Q, K, V, mask)
        out = self.W_o(self.combine_heads(attns))
        return out
    
class PositionWiseFeedForward(nn.Module):
    
    def __init__(self, d_model, d_ff):
        super().__init__()
        
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        
    def forward(self, x) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        
        super().__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]
    

class EncoderLayer(nn.Module):
    
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        
        self.multi_head_attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask) -> torch.Tensor:
        
        attns = self.multi_head_attention(x, x, x, mask)
        
        x = self.norm1(x + self.dropout(attns))
        
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
    

class DecoderLayer(nn.Module):
    
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        
        self.multi_head_attention = MultiHeadAttention(d_model, n_heads)
        self.cross_attention = MultiHeadAttention(d_model, n_heads)
        
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask) -> torch.Tensor:
        attns = self.multi_head_attention(x, x, x, tgt_mask)
        
        x = self.norm1(x + self.dropout(attns))
        
        attns = self.cross_attention(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attns))
        
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x
    
class Transformer(nn.Module):
    
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_length, dropout):
        super().__init__()
        
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        
        seq_length = tgt.size(1)
        
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask
    
    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        
        src_embeddings = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embeddings = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))
        
        enc_output = src_embeddings
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embeddings
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
            
        output = self.fc(dec_output)
        output = F.softmax(output, dim=-1)
        return output


class SemanticBridgeNetwork(L.LightningModule):
    def __init__(self, context_length: int, clip_length: int, model_name: str = 'gpt2' ) -> None:
        super().__init__()

        self.context_length = context_length
        self.lm = GPT2LMHeadModel.from_pretrained(model_name)

        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.d_model: int = self.lm.transformer.embed_dim
        self.bridge = Bridge(context_length, clip_length, self.d_model)
        self.criterion = nn.CrossEntropyLoss()

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return self.bridge.parameters(recurse=recurse)

    def train(self, mode: bool = True) -> None:
        self.bridge.train(mode)
        self.lm.eval()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.bridge(x)
        out = self.lm(inputs_embeds=x, labels=y)
        return out

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x, _, y = batch
        logits = self(x, y).logits
        loss = self.criterion(logits.contiguous().view(-1, logits.shape[-1]), y.contiguous().view(-1))
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
            x, _, y = batch
            logits = self(x, y).logits
            tokens = torch.argmax(logits, dim=-1)

            print('y: ', self.tokenizer.decode(y[0].tolist(), skip_special_tokens=True))
            print('y_hat: ', self.tokenizer.decode(tokens[0].tolist(), skip_special_tokens=True))    
    
    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(self.parameters(), lr=2e-5)
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=self.trainer.max_steps)
        return optimizer
    


class Bridge(L.LightningModule):
    
    def __init__(self, context_length: int, embeddings_dim: int, d_model: int, n_layers: int = 12):
        super().__init__()
        
        self.context_length = context_length
        self.embeddings_dim = embeddings_dim
        self.d_model = d_model
        
        self.W = nn.Linear(embeddings_dim, context_length * d_model)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, 8, 2048, 0.1) for _ in range(n_layers)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.W(x).view(-1, self.context_length, self.d_model)
        for enc_layer in self.encoder_layers:
            x = enc_layer(x,mask=None)
        return x
    

if __name__ == '__main__':

    train_dataset = Flickr8kDataset(split='train', data_dir='/Users/I561210/Learning/mirs/.dev/datasets/flickr8k')
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_dataset = Flickr8kDataset(split='test', data_dir='/Users/I561210/Learning/mirs/.dev/datasets/flickr8k')
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True)
    model = SemanticBridgeNetwork(DEFAULT_CONTEXT_LENGTH, 512)
    trainer = L.Trainer(max_epochs=100)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)
