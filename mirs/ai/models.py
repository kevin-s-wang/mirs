import torch
import open_clip
from mirs.ai.utils import get_device
from PIL import Image
from mirs.conf.config import config
import  torch.nn.functional as F
from typing import List

class CLIPModel(object):
    def __init__(self, model_name: str = config.clip.get_model(), pretrained=config.clip.get_pretrained()) -> None:
        self.device = get_device()
        self._load_model(model_name, pretrained)
        self._load_tokenizer(model_name)

    def _load_model(self, model_name: str, pretrained: str) -> None:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                                                    model_name, pretrained=pretrained, device=self.device)
            
    def _load_tokenizer(self, model_name: str) -> None:
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def get_text_embeddings(self, texts: List[str]) -> torch.Tensor:
        with torch.no_grad():
            tokens = self.tokenizer(texts).to(self.device)
            text_features = self.model.encode_text(tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def get_image_embeddings(self, image: Image.Image) -> torch.Tensor:
        with torch.no_grad():
            _image = self.preprocess(image).unsqueeze(0).to(self.device)
            image_features = self.model.encode_image(_image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features
    
    def get_image_text_similarity(self, image: Image.Image, text: str) -> torch.Tensor:
        with torch.no_grad():
            image_features = self.get_image_embeddings(image)
            text_features = self.get_text_embeddings([text])
            similarity = F.cosine_similarity(image_features, text_features)
        return similarity
    
    def get_image_texts_similarity(self, image: Image.Image, texts: List[str]) -> torch.Tensor:
        with torch.no_grad():
            image_features = self.get_image_embeddings(image)
            texts_features = self.get_text_embeddings(texts)
        return F.cosine_similarity(image_features, texts_features)

    def get_images_similarity(self, image1: Image.Image, image2: Image.Image) -> torch.Tensor:
        with torch.no_grad():
            image1_featrues = self.get_image_embeddings(image1)
            image2_features = self.get_image_embeddings(image2)
            similarity = F.cosine_similarity(image1_featrues, image2_features)
        return similarity
    
    def get_texts_similarity(self, text1: str, text2: str) -> torch.Tensor:
        with torch.no_grad():
            text1_features = self.get_text_embeddings([text1])
            text2_features = self.get_text_embeddings([text2])
            similarity = F.cosine_similarity(text1_features, text2_features)
        return similarity

clip_model = CLIPModel()