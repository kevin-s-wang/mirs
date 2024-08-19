import sys
sys.path.append('../mirs')
import os
import re
import json
import asyncio
from pathlib import Path
import glob
import pandas as pd
import shutil
from typing import Optional, List
from PIL import Image
from dataclasses import dataclass
from jsonargparse import CLI
from mirs.ai.llm.services import OllamaModel
from mirs.ai.models import clip_model as cm
from mirs.ai.utils import get_device
from pprint import pprint
from tqdm import tqdm
import torch
import numpy as np

import torch.nn.functional as F

def get_project_root():
    return os.path.dirname(os.path.dirname(__file__))


MAX_RETRY = 3

@dataclass
class PrepareArgs:
    force: Optional[bool] = False
    resume: Optional[bool] = False
    root_dir: Optional[str] = None
    cache_dir: Optional[str] = None
    dataset_dir: Optional[str] = None

@dataclass
class ImageTextPair:
    filename: str
    filepath: str
    caption: str
    dataset_name: str
    image_embeddings: np.ndarray
    caption_embeddings: np.ndarray
    cosine_similarity: float
    augmented: bool
    split: str

prompt_template = lambda x : f'''
Rewrite the sentence "{x}" in 10 variations with at most 77 tokens each.
You SHOULD always respond in a valid python string list and the string MUST be enclosed in double quotes.

```python
[sentences]
```

'''

def safe_extract_response(text: str) -> List[str]:
    p = r"```(?:\w+\s+)?(.*?)```"
    matches = re.findall(p, text, re.DOTALL)
    blocks = [block.strip() for block in matches]

    if blocks: 
        try:
            return json.loads(blocks[0])
        except Exception as e:
            print('ERROR:', e)
                
    return []
    
def unsafe_extract_response(text: str) -> List[str]:
    p = r"```(?:\w+\s+)?(.*?)```"
    matches = re.findall(p, text, re.DOTALL)
    blocks = [block.strip() for block in matches]
    return json.loads(blocks[0])
    

def _parse_cli_args():
    args = CLI(PrepareArgs, as_positional=False)

    if args.root_dir and not os.path.isdir(args.root_dir):
        raise FileNotFoundError(f'Invalid root directory: {args.root_dir}')

    if not args.root_dir:
        args.root_dir = get_project_root()
    
    if not args.cache_dir:
        args.cache_dir = os.path.join(args.root_dir, '.cache/prepare')
        os.makedirs(args.cache_dir, exist_ok=True)

    if not args.dataset_dir:
        args.dataset_dir = os.path.join(args.root_dir, 'datasets')
        if not os.path.isdir(args.dataset_dir):
            raise FileNotFoundError(f'Invalid dataset directory: {args.dataset_dir}')
    return args

def _ensure_data_dir(args: PrepareArgs):
    data_dir = os.path.join(args.cache_dir, 'data')
    
    if args.force:
        print('Force flag is set. Removing existing data...')
        shutil.rmtree(data_dir, ignore_errors=True)
    elif os.path.exists(data_dir) and os.listdir(data_dir) and not args.resume:
        raise FileExistsError(f'Data already exists in {data_dir}. Use --force to overwrite.')
        
    return data_dir

def get_cosine_similarity(image_embeddings, text_embeddings):
    return F.cosine_similarity(image_embeddings, text_embeddings).item()

async def _augment_text(data_dir: str):
    print('--> Augmenting training captions...')

    # Only augment the training data
    df = pd.read_parquet(data_dir, engine='pyarrow', filters=[('split', '=', 'train')])
    augmented = []
    lm = OllamaModel()
    total_augmented = 0

    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        caption = row['caption']
        retries = 0
        while retries <= MAX_RETRY:
            try:
                response = await lm.complete(prompt_template(caption))
                sentences = unsafe_extract_response(response)
                break
            except Exception as e:
                print('ERROR:', e)
                retries += 1
                print(f'[{retries}/{MAX_RETRY}] Retrying...')
       
        if retries > MAX_RETRY:
            print(f'Max retries({MAX_RETRY}) reached. Skipping caption:', caption)
            # @todo: log the skipped caption
            continue
      
        for sentence in sentences:
            text_embeds = cm.get_text_embeddings([sentence])
            image_embeds = torch.tensor(row['image_embeddings']).to(get_device())

            cosine_similarity = get_cosine_similarity(image_embeds, text_embeds)    
            
            similarity_diff = cosine_similarity - row['cosine_similarity']
            
            if similarity_diff >= .0:
                #print('Augmented caption:', sentence, 'Similarity difference:', similarity_diff)

                augmented.append(ImageTextPair(
                    filename=row['filename'],
                    filepath=row['filepath'],
                    caption=sentence,
                    dataset_name=row['dataset_name'],
                    image_embeddings=row['image_embeddings'],
                    caption_embeddings=text_embeds.cpu().detach().numpy()[0],
                    cosine_similarity=cosine_similarity,
                    augmented=True,
                    split=row['split']))
                
                total_augmented += 1
    print('Total augmented:', total_augmented)
    save_to_parquet(augmented, data_dir)


def save_to_parquet(data, data_dir):
    print('--> Saving to parquet', data_dir)
    pd.DataFrame(data).to_parquet(
        data_dir,
        engine='pyarrow',
        partition_cols=['dataset_name', 'split'],
        index=False,
    )

def _augment_image():
    print('--> Augmenting image...')


def _process_original_datasets(data_dir: str, dataset_dir: str):
    for filename in glob.glob('*.json', root_dir=dataset_dir):
        if filename == 'metadata.json':
            continue

        metadata_path = os.path.join(dataset_dir, filename)
        dataset_name = Path(metadata_path).stem
        with open(metadata_path, 'r') as stream:
            metadata = json.load(stream)
    
        image_text_pairs = []
        images = metadata['images']
        print('--> Processing metadata file', metadata_path)
        for image in tqdm(images):
            for sentence in image['sentences']:
                filename = image['filename']
                split = 'train' if image['split'] == 'restval' else image['split']
                if 'filepath' in image:
                    filepath = os.path.join(dataset_name, image['filepath'], filename)
                else:
                    filepath = os.path.join(dataset_name, filename)
                caption = re.sub('\s+\.?$', '.', sentence['raw'])
                im = Image.open(os.path.join(dataset_dir, filepath))
                image_embeddings = cm.get_image_embeddings(im)
                caption_embeddings = cm.get_text_embeddings([caption])
                image_text_pairs.append(ImageTextPair(
                    filename=filename,
                    filepath=filepath,
                    caption=caption,
                    dataset_name=dataset_name,
                    image_embeddings=image_embeddings.cpu().detach().numpy()[0],
                    caption_embeddings=caption_embeddings.cpu().detach().numpy()[0],
                    cosine_similarity=get_cosine_similarity(image_embeddings, caption_embeddings),
                    augmented=False,
                    split=split))
        
        save_to_parquet(image_text_pairs, data_dir)

async def main():
    args = _parse_cli_args()
    data_dir = _ensure_data_dir(args)

    _process_original_datasets(data_dir=data_dir, dataset_dir=args.dataset_dir)
    await _augment_text(data_dir)


if __name__ == '__main__':
    asyncio.run(main())

