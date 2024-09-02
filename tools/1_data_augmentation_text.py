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
from mirs.ai.models import clip_model as cm
from mirs.ai.utils import get_device
from pprint import pprint
from tqdm import tqdm
import torch
import numpy as np
from ollama import AsyncClient

import torch.nn.functional as F

def get_project_root():
    return os.path.dirname(os.path.dirname(__file__))


LANG_MODEL = 'mistral-nemo:latest'

MAX_RETRY = 3

@dataclass
class PrepareArgs:
    force: Optional[bool] = False
    resume: Optional[bool] = False
    root_dir: Optional[str] = None
    cache_dir: Optional[str] = None
    dataset_dir: Optional[str] = None
    data_dir: Optional[str] = None
    sample_dir: Optional[str] = None
    frac: Optional[float] = 0.03

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
    
    if not args.data_dir:
        args.data_dir = os.path.join(args.cache_dir, 'data')
    
    if not args.sample_dir:
        args.sample_dir = os.path.join(args.cache_dir, 'sample')
    
    if args.force:
        print('Force flag is set. Removing existing data and sample data...')
        shutil.rmtree(args.data_dir, ignore_errors=True)
        shutil.rmtree(args.sample_dir, ignore_errors=True)

    elif os.path.exists(args.data_dir) and os.listdir(args.data_dir) and not args.resume:
        raise FileExistsError(f'Data already exists in {args.data_dir}. Use --force to overwrite.')
    
    elif os.path.exists(args.sample_dir) and os.listdir(args.sample_dir) and not args.resume:
        raise FileExistsError(f'Sample data already exists in {args.sample_dir}. Use --force to overwrite.')
    
    return args


def get_cosine_similarity(image_embeddings, text_embeddings):
    return F.cosine_similarity(image_embeddings, text_embeddings).item()


def text_augmentation_sample(args: PrepareArgs) -> pd.DataFrame:
    print(f'--> Sampling training data for text augmentation with frac={args.frac}...')
    # Only augment the training data
    sample_df = pd.read_parquet(args.data_dir, engine='pyarrow', filters=[('split', '=', 'train')]) \
            .groupby('dataset_name', observed=True) \
            .sample(frac=args.frac, random_state=42)
    
    save_to_parquet(sample_df, args.sample_dir)
    return sample_df


async def _augment_text(args: PrepareArgs):
    print('--> Augmenting training captions...')

    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir, exist_ok=True)

    if not os.listdir(args.sample_dir):
        df = text_augmentation_sample(args)
    else:
        df = pd.read_parquet(args.sample_dir, engine='pyarrow')
    
    print('Sampling statistics:\n', df.groupby('dataset_name', observed=True)['dataset_name'].count())
    print('Total unique image samples:', df['filepath'].nunique())

    augmented_dir = os.path.join(args.cache_dir, 'augmented')
    
    if not os.path.exists(augmented_dir):
        os.makedirs(augmented_dir, exist_ok=True)

    augmented = []
    ollama_client = AsyncClient()
    total_augmented = 0

    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        caption = row['caption']
        filename = row['filename']
        dataset_name = row['dataset_name']
        split = row['split']

        augmented_df = pd.read_parquet(augmented_dir, engine='pyarrow', filters=[('dataset_name', '=', dataset_name), ('split', '=', split)])
        if filename in augmented_df['filename'].values:
            print('\nSkipping duplicate image:',  filename)
            continue

        retries = 0
        while retries <= MAX_RETRY:
            try:
                response = await ollama_client.generate(model=LANG_MODEL, prompt=prompt_template(caption))
                sentences = unsafe_extract_response(response['response'])
                break
            except Exception as e:
                print('ERROR:', e)
                retries += 1
                print(f'[{retries}/{MAX_RETRY}] Retrying...')
       
        if retries > MAX_RETRY:
            print(f'Max retries({MAX_RETRY}) reached. Skipping caption:', caption)
            # @todo: log the skipped caption
            continue
        if sentences:
            texts_embeds = cm.get_text_embeddings(sentences)
            image_embeds = torch.tensor(row['image_embeddings']).to(get_device())
            scores = F.cosine_similarity(image_embeds, texts_embeds)
            i = scores.argmax().item()
            cosine_similarity = scores.max().item()
            similarity_diff = cosine_similarity - row['cosine_similarity']
        
            if similarity_diff >= .0:
                augmented.append(ImageTextPair(
                        filename=filename,
                        filepath=row['filepath'],
                        caption=sentences[i],
                        dataset_name=dataset_name,
                        image_embeddings=row['image_embeddings'],
                        caption_embeddings=texts_embeds[i].cpu().detach().numpy()[0],
                        cosine_similarity=cosine_similarity,
                        augmented=True,
                        split=split))
                total_augmented += 1
            else:
                print('Skipping low similarity:', cosine_similarity)
                # @todo: log the skipped caption

            # Save per 1000 augmented samples
            if total_augmented % 1000 == 0 and augmented:
                save_to_parquet(augmented, augmented_dir)
                augmented.clear()
    # Save the remaining augmented samples
    if augmented:
        save_to_parquet(augmented, augmented_dir)
    


def save_to_parquet(data, data_dir):
    print('--> Saving to parquet', data_dir)
    pd.DataFrame(data).to_parquet(
        data_dir,
        engine='pyarrow',
        partition_cols=['dataset_name', 'split'],
        index=False,
    )

def _process_original_datasets(args: PrepareArgs):
    
    dataset_dir = args.dataset_dir
    data_dir = args.data_dir

    for filename in glob.glob('*.json', root_dir=dataset_dir):
        if filename == 'metadata.json':
            continue

        metadata_path = os.path.join(dataset_dir, filename)
        dataset_name = Path(metadata_path).stem
        with open(metadata_path, 'r') as stream:
            metadata = json.load(stream)
    
        image_text_pairs = []
        images = metadata['images']

        df = pd.read_parquet(data_dir, engine='pyarrow', 
                            filters=[('dataset_name', '=', dataset_name), ('augmented', '=', False)])
        
        n_images = len(images)
        n_processed = df['filepath'].nunique()
        if n_images == n_processed:
            print(f'{dataset_name} dataset is already processed. Skipping...')
            print(f'Processed {n_processed}/{n_images} images. Use --force to reprocess.')
            continue

        print('--> Processing metadata file', metadata_path)
        for image in tqdm(images):

            for sentence in image['sentences']:
                filename = image['filename']
                split = 'train' if image['split'] == 'restval' else image['split']
                if 'filepath' in image:
                    filepath = os.path.join(dataset_name, image['filepath'], filename)
                else:
                    filepath = os.path.join(dataset_name, filename)
               
                # >>> re.sub('\s+\.?$', '.', 'a person walking .')
                # 'a person walking.'
                # >>> re.sub('\s+\.?$', '.', 'a person walking.')
                # 'a person walking.'
                # >>> re.sub('\s+\.?$', '.', 'a person walking ')
                # 'a person walking.'
                
                caption = re.sub('\s+\.?$', '.', sentence['raw'])

                # Make sure each caption processed only once
                if caption in df['caption'].values:
                    print('Skipping duplicate caption:', caption)
                    continue

                im = Image.open(os.path.join(dataset_dir, filepath))

                # Generate embeddings for both image and caption
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

    _process_original_datasets(args)
    await _augment_text(args)


if __name__ == '__main__':
    asyncio.run(main())

