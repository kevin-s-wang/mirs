import os
import json
import httpx
import asyncio
import zipfile
from tqdm import tqdm

FLICKR8K = 'dataset_flickr8k.json'

DATASET_DIR = '.dev/datasets'


async def main():
    root_dir = os.path.dirname(os.path.dirname(__file__))
    datasets_dir = os.path.join(root_dir, DATASET_DIR)
    flickr8k_filepath = os.path.join(datasets_dir, 'Flickr8k_Dataset.zip')
    print(flickr8k_filepath)
    flickr8k_basedir = os.path.join(datasets_dir, 'Flicker8k_Dataset')

    if not os.path.exists(flickr8k_basedir):
        print(f'Extracting {flickr8k_filepath} to {datasets_dir} ...')
        with zipfile.ZipFile(flickr8k_filepath, 'r') as zip_ref:
            zip_ref.extractall(path=datasets_dir)
        print('Done')

    flickr8k_metadata_filepath = os.path.join(datasets_dir, FLICKR8K)
    with open(flickr8k_metadata_filepath, 'r') as f:
        flickr8k_metadata = json.load(f)
    flickr8k_images = flickr8k_metadata['images']

    # Test
    # flickr8k_images = flickr8k_images[1:2]

    for image in tqdm(flickr8k_images):
        image_filepath = os.path.join(datasets_dir, 'Flicker8k_Dataset', image['filename'])
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