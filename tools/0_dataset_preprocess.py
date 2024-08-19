import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
import re
import json
from safetensors.torch import save_file
from mirs.conf import config
from mirs.utils import load_flickr8k_metadata
from mirs.ai.models import clip_model
from PIL import Image
from typing import List

from tqdm import tqdm


def main():

    merge_train_val = True
    datasets_dir = os.path.join(config.dev_dir, 'datasets')
    flickr8k_dataset_dir = os.path.join(datasets_dir, 'flickr8k')

    # Ensuring dirs exist

    os.makedirs(flickr8k_dataset_dir, exist_ok=True)
    splits = ['train', 'test'] if merge_train_val else ['train', 'val', 'test']        
    
    flickr8k_metadata = load_flickr8k_metadata(datasets_dir)

    flickr8k_images = flickr8k_metadata['images']

    for split in splits:
        print(f'Processing {split} split...')
        # create split directory if missing
        os.makedirs(os.path.join(flickr8k_dataset_dir, split), exist_ok=True)

        image_embeddings_list: List[torch.Tensor] = []
        caption_embeddings_list: List[torch.Tensor] = []
        metadata = []
        image_idx = 0
        caption_idx = 0

        for image in tqdm(flickr8k_images):
            if image['split'] != split:
                if not merge_train_val:
                    continue
                if split == 'test' or image['split'] == 'test':
                    continue
    
            image_filename = image['filename']
            image_filepath = os.path.join(datasets_dir, 'Flicker8k_Dataset', image_filename)

            im = Image.open(image_filepath)
            captions = [re.sub('\s+\.?$', '.', sentence['raw']) for sentence in image['sentences']]
            image_embeddings = clip_model.get_image_embeddings(im)
            image_embeddings_list.append(image_embeddings)
            for caption in captions:
                caption_embeddings = clip_model.get_text_embeddings([caption])
                caption_embeddings_list.append(caption_embeddings)
                cosine_similarity = F.cosine_similarity(image_embeddings, caption_embeddings).item()
                if cosine_similarity < 0.1:
                    print(f'WARN: {image_filename} -> {caption} has a low similarity score: {cosine_similarity}')

                metadata_entry = {
                    'filename': image_filename,
                    'caption': caption,
                    'image_embeddings_index': image_idx,
                    'caption_embeddings_index': caption_idx,
                    'cosine_similarity': cosine_similarity,
                }
                metadata.append(metadata_entry)
                caption_idx += 1
            
            image_idx += 1

        # saving files
        metadata_filepath = os.path.join(flickr8k_dataset_dir, split, 'metadata.json')
        image_embeddings_filepath = os.path.join(flickr8k_dataset_dir, split, 'image_embeddings.safetensors')
        caption_embeddings_filepath = os.path.join(flickr8k_dataset_dir, split, 'caption_embeddings.safetensors')
        
        print('INFO: saving metadata for split: ', split)
        with open(metadata_filepath, 'w') as stream:
            json.dump(metadata, stream)

        print('INFO: saving image embeddings for split: ', split)
        save_file({
            'embeddings': torch.vstack(image_embeddings_list),
        }, image_embeddings_filepath)

        print('INFO: saving caption embeddings for split: ', split)
        save_file({
            'embeddings': torch.vstack(caption_embeddings_list),
        }, caption_embeddings_filepath)

if __name__ == '__main__':
    main()