import os
import json
import httpx
import asyncio
from tqdm import tqdm
from pathlib import Path
import glob



async def main():
    dataset_dir = '/Users/I561210/Learning/bridgenet/datasets'
    for filename in glob.glob('*.json', root_dir=dataset_dir):
        if filename == 'metadata.json':
            continue

        metadata_path = os.path.join(dataset_dir, filename)
        dataset_name = Path(metadata_path).stem
        with open(metadata_path, 'r') as stream:
            metadata = json.load(stream)

    # Test
    # flickr8k_images = flickr8k_images[1:2]
    images = metadata['images']

    print('Uploading images for dataset: ', dataset_name)

    for image in tqdm(images):
        image_filepath = os.path.join(dataset_dir, dataset_name, image['filename'])
        with open(image_filepath, 'rb') as image_file:
            # print('Uploading image: ', image['filename'])
            captions = [sentence['raw'] for sentence in image['sentences']]
            async with httpx.AsyncClient() as client:
                response = await client.post('http://localhost:8889/api/v1/images', follow_redirects=True, 
                                        files={'image': image_file}, data={'captions': captions})
                response.raise_for_status()
                # print(response.json())
            
if __name__ == '__main__':
    asyncio.run(main())