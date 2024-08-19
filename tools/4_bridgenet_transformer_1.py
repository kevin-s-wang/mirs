import sys
import os
import json
import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from safetensors import safe_open
from typing import Any, Iterator, Optional, List
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

def get_device() -> torch.device:
    device = "cuda" if torch.cuda.is_available() else \
             "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else \
             "cpu"
    return torch.device(device)

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

    i, j = x2.shape[-1] - x1.shape[-1], x2.shape[-2] - x1.shape[-2]

    mi, mj = i // 2, j // 2
    ni, nj = i - mi, j - mi
    
    x2 = F.pad(x2, (-mi, -ni, -mj, -nj))
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
# x = torch.randn(2, 1, 256, 256)
# y = torch.randn(2, 10, 4, 4)
# model = UNET(n_classes=1).to(device)
# print(model)

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=2e-5)

# model.train()
# n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print('Trainable parameters: ', n_trainable_params, 'or', n_trainable_params/1e6, 'million')
# for epoch in range(epochs):
#     x = x.to(device)
#     y = y.to(device)
    
#     y_hat = model(x)
#     print('y_hat', y_hat.shape)
    
#     loss = criterion(y_hat, y)
#     loss.backward()
#     optimizer.step()
#     optimizer.zero_grad()
    
#     print('Loss: ', loss.item())
#     break



# Transformer Encoder
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


class SemanticVisualPrompt(nn.Module):
    def __init__(self, prompt_length: int, clip_length: int,  d_model: int) -> None:
        super().__init__()

        self.promt_length = prompt_length
        self.clip_length = clip_length
        self.d_model = d_model

    

        self.encoder_layers = nn.ModuleList([])

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return x

class SemanticPrompt(nn.Module):
    def __init__(self, prompt_length: int, clip_length: int,  d_model: int) -> None:
        super().__init__()

        self.promt_length = prompt_length # 10
        self.clip_length = clip_length    # 512
        self.d_model = d_model            # 768 (gpt2)

        self.proj_in = nn.Linear(clip_length, 256, bias=False)

        self.sampling = UNET(n_classes=prompt_length)

        self.proj_out = nn.Linear(68*68*prompt_length, prompt_length*d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(start_dim=1)
        x = F.dropout(F.relu(self.proj_in(x)), p=0.1).unsqueeze(1)
        x = torch.bmm(x.transpose(1, 2), x).unsqueeze(1)
        x = self.sampling(x) # (B, 10, 68, 68)
        x = x.flatten(start_dim=1) # (B, 10, 68*68)
        x = self.proj_out(x) # (B, 10, 768)
        return x

class SemanticBrigeNetwork(nn.Module):
    def __init__(self, prompt_length: int, clip_length: int,  d_model: int, language_model: Any) -> None:
        super().__init__()

        self.promt_length = prompt_length
        self.clip_length = clip_length
        self.d_modoel = d_model

        self.prompt = SemanticPrompt(prompt_length, clip_length, d_model)

        self.language_model = language_model

    def train(self, mode: bool = True):
        self.prompt.train(mode)
        self.language_model.eval()
        return self

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return self.prompt.parameters(recurse=recurse)

    def forward(self, prompt: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        x = self.prompt(prompt).view(-1, self.promt_length, self.d_modoel)
        
        target_embeds = self.language_model.transformer.wte(target.clone()).squeeze(1)
        x = torch.cat((x, target_embeds), dim=1)
        mask = mask.squeeze(1)
        mask = torch.cat((torch.ones(x.shape[0], self.promt_length, device=get_device()), mask), dim=1)
        x = self.language_model(inputs_embeds=x, attention_mask=mask)
        return x
    
class Flickr8kDataset(Dataset):
    def __init__(self, context_length: int, tokenizer: Any, split: str = 'train', data_dir: str = None) -> None:
        super().__init__()
        allowed_splits = ('train', 'val', 'test')
        if split not in allowed_splits:
            raise ValueError(f"Allowed splits are: {','.join(allowed_splits)}")
        self.split = split
        self.context_length = context_length
        self.data_dir = os.path.join(data_dir, split)
        self.image_embeddings: Optional[torch.Tensor] = None
        self.caption_embeddings: Optional[torch.Tensor] = None
        self.metadata: Optional[List] = None
        self.tokenizer = tokenizer
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
        caption_tokens = self.tokenizer(caption, 
                                        padding='max_length',
                                        truncation=True,
                                        max_length=self.context_length, 
                                        return_tensors='pt')
        return image_embeddings, caption_embeddings, caption_tokens['input_ids'], caption_tokens['attention_mask']

def evaluate(model, reload: bool = False):
    model.eval()
    with torch.no_grad():
        for x, _, y, mask in test_dataloader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            logits = model(x, y, mask).logits
            logits = logits[:, prompt_length:, :]
            tokens = torch.argmax(logits, dim=-1)
            print('y: ', tokenizer.decode(y[0].squeeze(0).tolist(), skip_special_tokens=True))
            print('y_hat: ', tokenizer.decode(tokens[0].squeeze(0).tolist(), skip_special_tokens=True))    

torch.autograd.set_detect_anomaly(True)
if __name__ == '__main__':
    device = get_device()

    print('Running on device: ', device)
    model_name = 'gpt2'
    batch_size = 32
    clip_length = 512
    context_length = 20
    prompt_length = 10
    epochs = 10
    d_model = 768

    language_model = GPT2LMHeadModel.from_pretrained(model_name, device_map=device)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    data_dir = '/Users/I561210/Learning/mirs/.dev/datasets/flickr8k'
    flickr8k_train_dataset = Flickr8kDataset(context_length, tokenizer, split='train', data_dir=data_dir)
    flickr8k_test_dataset  = Flickr8kDataset(context_length, tokenizer, split='test', data_dir=data_dir)
    print('Train dataset size: ', len(flickr8k_train_dataset))
    print('Test dataset size: ',  len(flickr8k_test_dataset))

    train_dataloader = DataLoader(flickr8k_train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader  = DataLoader(flickr8k_test_dataset, batch_size=batch_size, shuffle=True)

    model = SemanticBrigeNetwork(prompt_length, clip_length, d_model, language_model).to(device)

    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Trainable parameters: ', n_trainable_params, 'or', n_trainable_params/1e6, 'million')
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = get_linear_schedule_with_warmup(
                    optimizer, 
                    num_warmup_steps=5000, 
                    num_training_steps=len(train_dataloader)*epochs
                )
    
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    train_loss = []
    test_loss = []
    for epoch in range(epochs):
        batch_count = 0
        epoch_loss = 0.0
        model.train()

        for x, _, y, mask in train_dataloader:
            model.zero_grad()
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)

            logits = model(x, y, mask).logits
            tokens = logits.argmax(dim=-1)[0]
            
            print('y: ', tokenizer.decode(y[0].squeeze(0).tolist(), skip_special_tokens=True))
            print('tokens: ', tokenizer.decode(tokens.squeeze(0).tolist(), skip_special_tokens=True))
            # logits = logits[:, prompt_length-1: -1]

            

            y[~mask.bool()] = -100
            idx = torch.randperm(prompt_length)
            prompt_y = y[:, :, idx]
            _y = torch.cat((prompt_y, y), dim=2)
            
            loss = criterion(logits.contiguous().view(-1, logits.shape[-1]), _y.contiguous().view(-1))
            # loss = criterion(logits.contiguous().view(-1, logits.shape[-1]), y.contiguous().view(-1))
                
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            batch_count += 1
            epoch_loss += loss.item()
            print('Epoch: ', epoch+1, 'Batch: ', batch_count, 'Loss:', loss.item())
            # evaluate(model)
            
        
        print('Epoch: ', epoch+1, 'Train loss:', epoch_loss/batch_count)
        train_loss.append(epoch_loss/batch_count)
        
        evaluate(model)
        print('Saving model...')
        torch.save(model.state_dict(), f'bridgenet_{epoch}.pt')
