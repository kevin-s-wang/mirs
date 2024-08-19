import sys
import os
import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from safetensors import safe_open
from typing import Any, Optional, List
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)
    

class DownSample(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.double_conv(x)
        x = self.pool(y)
        return x, y
    
class UpSample(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        
        self.convt = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_channels, out_channels)
        
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x = self.convt(x1)
        x = crop_and_concat(x, x2)
        x = self.double_conv(x)
        return x
        
def crop_and_concat(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    c1 = (x2.shape[-1] - x1.shape[-1]) // 2
    c2 = (x2.shape[-2] - x1.shape[-2]) // 2
    x2 = F.pad(x2, (-c1, -c1, -c2, -c2))
    return torch.cat((x2, x1), dim=1)

class BottleNeck(nn.Module):
    
    def __init__(self, in_channels: int, out_channels) -> None:
        super().__init__()
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bottleneck(x)

class UNET(nn.Module):
    
    def __init__(self, n_classes: int = 2):
        super().__init__()
        
        self.down1 = DownSample(1, 64)
        self.down2 = DownSample(64, 128)
        self.down3 = DownSample(128, 256)
        self.down4 = DownSample(256, 512)
        
        self.bottleneck = BottleNeck(512, 1024)
        
        self.up1 = UpSample(1024, 512)
        self.up2 = UpSample(512, 256)
        self.up3 = UpSample(256, 128)
        self.up4 = UpSample(128, 64)
        
        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Down sampling
        x, x1 = self.down1(x)
        x, x2 = self.down2(x)        
        x, x3 = self.down3(x)
        x, x4 = self.down4(x)
        
        x = self.bottleneck(x)
        
        # Up sampling
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Final convolution
        x = self.final_conv(x)
        return x
    

# epochs = 10
# x = torch.randn(2, 1, 572, 572)
# y = torch.randn(2, 2, 388, 388)
# model = UNET().to(device)
# print(model)

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=2e-5)

# model.train()
# n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print('Trainable parameters: ', n_trainable_params, 'or', n_trainable_params/1e6, 'million')
# for epoch in range(epochs):
#     x.to(device)
#     y.to(device)
    
#     y_hat = model(x)

#     loss = criterion(y_hat, y)
#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()
    
#     print('Loss: ', loss.item())
#     break


model_name = 'gpt2'

model_path = '/kaggle/working/bridgenet.pt'


DEFAULT_CONTEXT_LENGTH = 20

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
    
class BridgeEncoder(nn.Module):
    
    def __init__(self, context_length: int, embeddings_dim: int, d_model: int, n_layers: int = 6):
        super().__init__()
        
        self.context_length = context_length
        self.embeddings_dim = embeddings_dim
        self.d_model = d_model
        
#         self.W = nn.Linear(embeddings_dim, context_length * d_model)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, 4, 1024, 0.1) for _ in range(n_layers)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.W(x).view(-1, self.context_length, self.d_model)
        for enc_layer in self.encoder_layers:
            x = enc_layer(x,mask=None)
        return x

class SimpleCNN(nn.Module):
    
    def __init__(self, in_channels: int, context_length: int,  d_model: int, dropout=0.1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(32*2*64, 1024)
        self.fc2 = nn.Linear(1024, context_length*d_model)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.dropout(F.relu(self.conv1(x))))
        x = self.pool(self.dropout(F.relu(self.conv2(x))))
        x = self.pool(self.dropout(F.relu(self.conv3(x))))
        
        x = x.flatten(start_dim=1)    
        x = self.dropout(F.relu(self.fc1(x)))
        
        x = F.relu(self.fc2(x))
        
        return x
        
class BridgeNet1(nn.Module):
    def __init__(self, clip_length: int, context_length: int = DEFAULT_CONTEXT_LENGTH):
        super().__init__()
        self.context_length = context_length
        self.clip_length = clip_length
        self.lm = GPT2LMHeadModel.from_pretrained(model_name, device_map=get_device())

        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.d_model: int = self.lm.transformer.embed_dim
        
        self.W = nn.Linear(clip_length, 3*clip_length*context_length)
        self.sample = SimpleCNN(3, context_length, self.d_model)
        self.bridge = BridgeEncoder(context_length, clip_length, self.d_model)

    def parameters(self):
        params = []
        for p in self.W.parameters():
            params.append(p)
            
        for p in self.sample.parameters():
            params.append(p)
            
        for p in self.bridge.parameters():
            params.append(p)
        return params

    def train(self, mode: bool = True):
        self.sample.train(mode)
        self.bridge.train(mode)
        self.lm.eval()
        return self


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.W(x).view(-1, 3, self.context_length, self.clip_length)
        x = self.sample(x)
        x = self.bridge(x.view(-1, self.context_length, self.d_model))
        out = self.lm(inputs_embeds=x)
        return out
    
class BridgeNet(nn.Module):
    def __init__(self, clip_length: int, context_length: int = DEFAULT_CONTEXT_LENGTH):
        super().__init__()
        self.context_length = context_length
        self.lm = GPT2LMHeadModel.from_pretrained(model_name, device_map=get_device())

        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.d_model: int = self.lm.transformer.embed_dim
        
        self.W1 = nn.Linear(clip_length, 572*572)
        self.W2 = nn.Linear(2*388*388, context_length*self.d_model)
        self.sample = UNET()
        
        self.bridge = BridgeEncoder(context_length, clip_length, self.d_model)

    def parameters(self):
        params = []
        for p in self.bridge.parameters():
            params.append(p)
            
        for p in self.sample.parameters():
            params.append(p)
            
        return params

    def train(self, mode: bool = True):
        self.sample.train(mode)
        self.bridge.train(mode)
        self.lm.eval()
        return self


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.W1(x)).view(-1, 1, 572, 572)
        
        x = self.sample(x)
        x = F.relu(self.W2(x)).view(-1, self.context_length, self.d_model )
        
        x = self.bridge(x)
        out = self.lm(inputs_embeds=x)
        return out

def evaluate(model, reload: bool = False):
    if reload:
        model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        for x, _, y in test_dataloader:
            x = x.to(device)
            logits = model(x).logits
            tokens = torch.argmax(logits, dim=-1)

            print('y: ', tokenizer.decode(y[0].tolist(), skip_special_tokens=True))
            print('y_hat: ', tokenizer.decode(tokens[0].tolist(), skip_special_tokens=True))    


if __name__ == '__main__':
    device = get_device()

    print('Running on device: ', device)
    batch_size = 218
    embeddings_dim = 512
    epochs = 100
    
    data_dir = '/kaggle/input/flickr8k-dataset/flickr8k'
    flickr8k_train_dataset = Flickr8kDataset(split='train', data_dir=data_dir)
    flickr8k_test_dataset  = Flickr8kDataset(split='test', data_dir=data_dir)
    print('Train dataset size: ', len(flickr8k_train_dataset))
    print('Test dataset size: ',  len(flickr8k_test_dataset))

    train_dataloader = DataLoader(flickr8k_train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader  = DataLoader(flickr8k_test_dataset, batch_size=32, shuffle=True)

    model = BridgeNet1(embeddings_dim).to(device)
    
    if os.path.exists(model_path):
        print('Loading existing model...')
        model.load_state_dict(torch.load(model_path))
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print('Trainable parameters: ', n_trainable_params, 'or', n_trainable_params/1e6, 'million')

    its = 1
    for epoch in range(epochs):
        print('Epoch: ', epoch+1)
        model.train()
        for x, _, y in train_dataloader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x).logits
            loss = criterion(logits.contiguous().view(-1, logits.shape[-1]), y.contiguous().view(-1))
                
            loss.backward()
            print('Iterations: ', its, 'Loss:', loss.item())
            optimizer.step()
            optimizer.zero_grad()
            its += 1
            
        if (epoch + 1) % 10 == 0:
            print('Saving model...')
            torch.save(model.state_dict(), f'bridgenet_{epoch}.pt')
        evaluate(model)