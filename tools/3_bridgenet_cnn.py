import sys
sys.path.append('../mirs')

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import open_clip
from PIL import Image
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from mirs.ai.utils import get_device
from typing import Dict, List

lm_model_name = 'gpt2-xl'

class BridgeNetDataset(Dataset):
    def __init__(self):
        pass
    
    def __len__(self) -> int:
        return 0

    def __getitem__(self, index: int) -> torch.Tensor:
        pass

class BridgeNet(nn.Module):

    def __init__(self, max_n_tokens: int = 77):
        super().__init__()

        device = get_device()

        self.max_n_tokens = max_n_tokens

        self.lm = GPT2LMHeadModel.from_pretrained(lm_model_name, device_map=device)
        self.lm.eval() # Frozen LM weights

        self.lm_tokenizer = GPT2Tokenizer.from_pretrained(lm_model_name)
        # self.d_model = self.lm.transformer.wte.weight.shape[1]
        self.d_model: int = self.lm.transformer.embed_dim
        print('d_model: ', self.d_model)
        print('max_n_tokens: ', max_n_tokens)


        self.clip, _, self.preprocess = open_clip.create_model_and_transforms(
                                                    model_name='ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)
        self.embeddings_dim: int = self.clip.visual.output_dim

        print('embeddings_dim: ', self.embeddings_dim)

        self.up_sampling = nn.Sequential(
            nn.Conv1d(1, max_n_tokens, 1),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(self.embeddings_dim, self.d_model),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.clip.encode_image(x)
        x = x.unsqueeze(1)
        x = self.up_sampling(x)
        return x


if __name__ == '__main__':
    device = get_device()
    model = BridgeNet().to(device)
    im = Image.open('/Users/I561210/Learning/mirs/.dev/data/images/00a6d6ec47548584205bd080c679816141a1dd63c6ec859113b2e0345b60265c4dc7f0641e5d7c4bea2a0df9d1918a53af01b9f6aa40b35e5804143851d31c57.jpg')

    x = model.preprocess(im).unsqueeze(0).to(device)
    out = model(x)
    print(out.shape)


    