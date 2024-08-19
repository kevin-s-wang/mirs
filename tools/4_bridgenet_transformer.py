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
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

model_name = 'gpt2'
model_path = '/kaggle/working/bridgenet.pt'

DEFAULT_CONTEXT_LENGTH = 10
DEFAULT_PREFIX_LENGTH = 10

def get_device() -> torch.device:
    device = "cuda" if torch.cuda.is_available() else \
             "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else \
             "cpu"
    return torch.device(device)

device = get_device()

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
        print(x.shape, x1.shape, x2.shape)
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
        
        self.down1 = DownSample(1, 4)
        self.down2 = DownSample(4, 8)

        
        self.bottleneck = BottleNeck(8, 16)
        
        self.up1 = UpSample(16, 8)
        self.up2 = UpSample(8, 4)
        
        self.final_conv = nn.Conv2d(4, n_classes, kernel_size=1)
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # Down sampling
        x, x1 = self.down1(x)
        x, x2 = self.down2(x)        

        x = self.bottleneck(x)
        
        # Up sampling
        x = self.up1(x, x2)
        x = self.up2(x, x1)
        
        # Final convolution
        x = self.final_conv(x)
        print(x.shape)
        return x

# epochs = 10
# x = torch.randn(2, 1, 512, 512)
# y = torch.randn(2, 1, 324, 324)
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




class Flickr8kDataset(Dataset):
    def __init__(self, tokenizer, split: str = 'train', data_dir: str = None) -> None:
        super().__init__()
        allowed_splits = ('train', 'val', 'test')
        if split not in allowed_splits:
            raise ValueError(f"Allowed splits are: {','.join(allowed_splits)}")
        self.split = split

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
                                               max_length=DEFAULT_CONTEXT_LENGTH, 
                                               return_tensors='pt')
        return image_embeddings, caption_embeddings, caption_tokens['input_ids'], caption_tokens['attention_mask']


    
class SemanticSampler(nn.Module):
    def __init__(self, prefix_length: int, clip_length: int, d_model: int):
        super().__init__()

        self.prefix_length = prefix_length
        self.clip_length = clip_length
        self.d_model = d_model

        self.W = nn.Linear(clip_length, prefix_length*clip_length, bias=False)
        self.unet = UNET(n_classes=prefix_length) # (batch_size, context_length, 324)

        self.mlp = nn.Sequential(
            nn.Linear(prefix_length*24*24, 2*clip_length*prefix_length),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2*clip_length*prefix_length, 4*prefix_length*clip_length),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(4*prefix_length*clip_length, prefix_length*d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.W(x).view(-1, 64, 64).unsqueeze(1)
        x = self.unet(x)

        x = x.flatten(start_dim=1)
        x = self.mlp(x)
        x = x.view(-1, self.prefix_length, self.d_model)
        return x

class SemanticBridgeNetwork(nn.Module):

    def __init__(self, 
                 context_length: int, 
                 clip_length: int, 
                 d_model: int,
                 language_model: Any):
        super().__init__()

        self.context_length = context_length
        self.clip_length = clip_length
        self.language_model = language_model


        self.prefix = SemanticSampler(DEFAULT_PREFIX_LENGTH, clip_length, d_model)
        

    def train(self, mode: bool = True):
        self.prefix.train(mode)
        self.language_model.eval()
        return self

    def parameters(self, recurse: bool = True) -> Any:
        return self.prefix.parameters(recurse=recurse)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.prefix(x.view(-1, self.clip_length))
        y = self.language_model.transformer.wte(y).squeeze(1)
        input_embeds = torch.cat((x, y), dim=1)

        out = self.language_model(inputs_embeds=input_embeds)
        return out

torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':
    device = get_device()

    print('Running on device: ', device)
    batch_size = 128
    clip_length = 512
    epochs = 10
    
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    data_dir = '/Users/I561210/Learning/mirs/.dev/datasets/flickr8k'
    flickr8k_train_dataset = Flickr8kDataset(tokenizer, split='train', data_dir=data_dir)
    flickr8k_test_dataset  = Flickr8kDataset(tokenizer, split='test', data_dir=data_dir)
    print('Train dataset size: ', len(flickr8k_train_dataset))
    print('Test dataset size: ',  len(flickr8k_test_dataset))

    train_dataloader = DataLoader(flickr8k_train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader  = DataLoader(flickr8k_test_dataset, batch_size=32, shuffle=True)

    language_model = GPT2LMHeadModel.from_pretrained(model_name, device_map=device)
    d_model: int = language_model.transformer.embed_dim

    model = SemanticBridgeNetwork(DEFAULT_CONTEXT_LENGTH, clip_length, d_model, language_model) \
                .to(device)
    
    print(model)
    train_loss = []
    eval_loss = []

    def evaluate(model, reload: bool = False):
        if reload:
            model.load_state_dict(torch.load(model_path))
        model.eval()
        n_iters = 0
        total_loss = 0.0
        with torch.no_grad():
            for x, _, y, mask in test_dataloader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x).logits
                _y = y.clone()
                y[~mask.bool()] = -100
                loss = criterion(logits.contiguous().view(-1, logits.shape[-1]), y.contiguous().view(-1))
                tokens = torch.argmax(logits, dim=-1)
                n_iters += 1
                total_loss += loss.item()
                print('y: ', tokenizer.decode(_y[0][0].tolist(), skip_special_tokens=True))
                print('y_hat: ', tokenizer.decode(tokens[0].tolist(), skip_special_tokens=True))
        return total_loss / n_iters

    if os.path.exists(model_path):
        print('Loading existing model...')
        model.load_state_dict(torch.load(model_path))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = get_linear_schedule_with_warmup(
                    optimizer, 
                    num_warmup_steps=5000, 
                    num_training_steps=len(train_dataloader)*epochs
                )
    
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Trainable parameters: ', n_trainable_params, 'or', n_trainable_params/1e6, 'million')

    iters = 1
    for epoch in range(epochs):
        print('Epoch: ', epoch+1)
        model.train()
        n_iters = 0
        total_loss = 0.0
        for x, _, y, mask in train_dataloader:
            model.zero_grad()
            x = x.to(device)
            y = y.to(device)

            logits = model(x, y).logits
            print(logits.shape)
            
            logits = logits[:, DEFAULT_PREFIX_LENGTH-1: -1]
            y[~mask.bool()] = -100

            print('logi', logits.shape, y.shape)
 
            loss = criterion(logits.contiguous().view(-1, logits.shape[-1]), y.contiguous().view(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()

            # optimizer.zero_grad()
            print('Iterations: ', iters, 'Loss:', loss.item())
            iters += 1
            n_iters += 1
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print('Saving model...')
            torch.save(model.state_dict(), f'bridgenet_{epoch}.pt')
        
        eval_loss.append(evaluate(model))
        train_loss.append(total_loss / n_iters)

    x = np.arange(1, epochs+1)
    plt.plot(x, np.array(train_loss), label='train')
    plt.plot(x, np.array(eval_loss), label='eval')
    plt.legend()
    plt.title('Loss statistics')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('loss.png')

    