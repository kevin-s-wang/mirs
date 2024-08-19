import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import json
import torch
import math
from safetensors import safe_open
from mirs.conf import config
from typing import List, Optional


if __name__ == '__main__':
    for split in ['train', 'test']:
        print(f'Validating {split} split...')
        split_dir = os.path.join(config.datasets_dir, 'flickr8k', split)
        image_embeddings_filepath = os.path.join(split_dir, 'image_embeddings.safetensors')
        caption_embeddings_filepath = os.path.join(split_dir, 'caption_embeddings.safetensors')
        metadata_filepath = os.path.join(split_dir, 'metadata.json')

        metadata: Optional[List] = None
        with open(metadata_filepath, 'r') as stream:
            metadata = json.load(stream)

        all_image_embeddings: Optional[torch.Tensor] = None
        with safe_open(image_embeddings_filepath, framework='pt') as f:
            all_image_embeddings = f.get_slice('embeddings')

        all_caption_embeddings: Optional[torch.Tensor] = None
        with safe_open(caption_embeddings_filepath, framework='pt') as f:
            all_caption_embeddings = f.get_slice('embeddings')

        for entry in metadata:
            i = entry['image_embeddings_index']
            j = entry['caption_embeddings_index']
            image_embeddings = all_image_embeddings[i:i+1, :]
            caption_embeddings = all_caption_embeddings[j:j+1, :]

            cosine_similarity = torch.nn.functional.cosine_similarity(image_embeddings, caption_embeddings).item()
            
            if math.fabs(cosine_similarity - entry['cosine_similarity']) > 0.0001:
                print(f'WARN: {entry["filename"]} -> {entry["caption"]} has a different similarity score: {cosine_similarity} -> {entry["cosine_similarity"]}')


