import logging
import os
import torch
import torch.nn.functional as F
import asyncio
from pathlib import Path
import json
from tqdm import tqdm
import httpx
from typing import Optional, List, Dict
import pandas as pd

def recall_at_k(scores, positive_pairs, k):
    pass


async def retrieve(text: Optional[str] = None, images: Optional[List[str]] = None, max_retry: int = 3):
    data: Optional[Dict] = None
    files: Optional[Dict] = None

    if text is None and images is None:
        raise ValueError("Either text or images must be provided")
    
    if text:
        data = {'prompt': text}
    
    if images:
        image_files: List[bytes] = []
        for image in images:
           with open(image, 'rb') as image_file:
                image_files.append(image_file.read())
        files = {'images': image_files}
    
    retried_times = 0
    async with httpx.AsyncClient() as http:
        while retried_times <= max_retry:
            try:
                response = await http.post("http://localhost:8889/api/v1/images/search", 
                                    follow_redirects=True,
                                    params={'limit': 100, 'offset': 0},
                                    files=files, 
                                    data=data,
                                    timeout=300.0)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                print('Retrieval error: ', e)
                retried_times += 1
    return None

async def t2i(texts: List[str]):
    pass

async def i2t(images: List[str]):
    pass


dataset_dir = '/Users/I561210/Learning/bridgenet/datasets'

async def evaluate():
    # flickr8k = pd.read_parquet('.cache/prepare/data', 
    #                 engine='pyarrow', 
    #                 filters=[('dataset_name', '==', 'flickr8k'), ('split', '==', 'test')]).sample(100)

    metadata_path = os.path.join(dataset_dir, 'flickr8k.json')
    # dataset_name = Path(metadata_path).stem
    with open(metadata_path, 'r') as stream:
        metadata = json.load(stream)

    images = [x for x in metadata['images'] if x['split'] == 'test']
    import random

    samples = random.sample(images, 100)

    for sample in samples:
        image_path = os.path.join(dataset_dir, sample['file_path'])
        retrieved = await retrieve(images=[image_path])
        print(retrieved)


    print(retrieved)
    # async with AsyncClient() as client:
    #     logging.info("Loading model...")
    #     model = torch.load("model.pth")
    #     model.eval()

    #     logging.info("Loading test data...")
    #     test_data = torch.load("test_data.pth")

    #     logging.info("Evaluating...")
    #     recalls = []
    #     for i, (user, positive_item) in enumerate(tqdm(test_data)):
    #         user = user.unsqueeze(0)
    #         positive_item = positive_item.unsqueeze(0)

    #         scores = model(user, positive_item)
    #         scores = scores.flatten()

    #         # Get the top 10 items
    #         top_10_scores, top_10_indices = torch.topk(scores, 10)
    #         top_10_indices = top_10_indices.cpu().numpy().tolist()

    #         # Get the positive item index
    #         positive_item_index = positive_item.cpu().numpy().item()

    #         # Recall@10
    #         recall = int(positive_item_index in top_10_indices)
    #         recalls.append(recall)

    #         if i % 100 == 0:
    #             logging.info(f"Recall@10: {sum(recalls) / len(recalls)}")

    #     logging.info(f"Recall@10: {sum(recalls) / len(recalls)}")


if __name__ == '__main__':
    asyncio.run(evaluate())