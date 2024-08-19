import sys
sys.path.append('../mirs')
import os
import re
import asyncio
import httpx
import json
from time import time
from abc import ABC, abstractmethod
from mirs.ai.llm.services.base import LanguageModel
from tqdm import tqdm
from mirs.conf.config import config
from enum import Enum
from mirs.ai.models import clip_model
from PIL import Image

OPENAI_COMPLETION_V1 = 'https://azure-openai-serv-i057149.cfapps.sap.hana.ondemand.com/api/v1/completions'
ACCESS_TOKEN_URL = 'https://lcap-qa.authentication.sap.hana.ondemand.com/oauth/token'
CLIENT_ID = 'sb-bf537e8d-dd7a-4046-9da7-5f63071d766a!b32725|azure-openai-service-i057149-xs!b16730'
CLIENT_SECRET = 'f099da3f-9093-4d6d-8c1d-f86c2488f333$9B4RdJ93nzJ6AYyplYkY9RmpqSZS5qUmLepuEPk17Q8='

GRANT_TYPE = 'client_credentials'


from typing import Dict, List

class AccessTokenProvider(ABC):
    @abstractmethod
    async def get(self) -> str:
        raise NotImplementedError


class AzureAccessTokenProvider(AccessTokenProvider):

    def __init__(self, root_dir: str) -> None:
        self.root_dir = root_dir

    async def get(self) -> str:

        token_cache_dir = os.path.join(self.root_dir, '.dev/cache')
        if not os.path.isdir(token_cache_dir):
            os.makedirs(token_cache_dir, exist_ok=True)

        token_cache = os.path.join(token_cache_dir, 'token.json')

        if os.path.exists(token_cache):
            with open(token_cache, 'r') as stream:
                token = json.load(stream)
                if time() - token['ts'] <= (token['expires_in'] - 300):
                    return token['access_token']

        async with httpx.AsyncClient() as client:
            response = await client.post(ACCESS_TOKEN_URL, follow_redirects=True,
                                            timeout=60.0,
                                            auth=(CLIENT_ID, CLIENT_SECRET), params={'grant_type': GRANT_TYPE})
            response.raise_for_status()
            token = response.json()
            token['ts'] = time()
            # Cache token
            with open(token_cache, 'w') as stream:
                json.dump(token, stream)
            
            return token['access_token']


class ChatHistory(ABC):
    def __init__(self) -> None:
        super().__init__()


    def add(self, message: str):
        raise NotImplementedError
    

class Role(Enum):
    SYSTEM = 'system'
    ASSISTANT = 'assistant'
    USER = 'user'


class ChatMessage(ABC):

    def __init__(self, role: Role, content: str) -> None:
        super().__init__()
        self.role = role
        self.content = content


class ProxiedGPT(LanguageModel):
    def __init__(self, 
        base_url: str, 
        access_token_provider: AccessTokenProvider, 
        deployment_id='gpt-35-turbo',  
        temperature=.8,
    ) -> None:
        self.base_url = base_url
        self.deployment_id = deployment_id
        self.temperature = temperature
        self.access_token_provider = access_token_provider

    def _get_request_body(self, **kwargs) -> Dict:
        return dict(
            deployment_id=self.deployment_id,
            temperature=self.temperature,
            **kwargs
        )

    async def complete(self, prompt: str) -> Dict:
        access_token: str = await self.access_token_provider.get()

        async with httpx.AsyncClient() as client:
            request_body = self._get_request_body(messages=[{'content': prompt, 'role': 'user'}])
            response = await client.post(OPENAI_COMPLETION_V1, follow_redirects=True, 
                        headers={'Authorization': f'Bearer {access_token}'},
                        timeout=None, # disable timeout
                        json=request_body)
            response.raise_for_status()
        return response.json()


def get_flickr8k_metadata(filepath: str) -> Dict:
    with open(filepath, 'r') as f:
        return json.load(f)

def load_augmented_flickr8k_metadata(filepath: str) -> List:

    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(e)
        return []


async def main():
    datasets_dir = os.path.join(config.root_dir, '.dev/datasets')
    flickr8k_metadata_filepath = os.path.join(datasets_dir, 'dataset_flickr8k.json')
    augmented_flickr8k_metadata_filepath = os.path.join(datasets_dir, 'dataset_flickr8k_augmented.json')

    proxy = ProxiedGPT(OPENAI_COMPLETION_V1, AzureAccessTokenProvider(config.root_dir))

    flickr8k_metadata = get_flickr8k_metadata(flickr8k_metadata_filepath)
    augmented_flickr8k_metadata = load_augmented_flickr8k_metadata(augmented_flickr8k_metadata_filepath)

    flickr8k_images = flickr8k_metadata['images']
    
    for image in tqdm(flickr8k_images):
        try:
            augmented = next(x for x in augmented_flickr8k_metadata if x['image'] == image['filename'])
            print(augmented['image'], 'has already been processed, skipping it...')
            continue
        except StopIteration:
            pass

        image_filepath = os.path.join(datasets_dir, 'Flicker8k_Dataset', image['filename'])
        captions = [re.sub('\s+\.?$', '.', sentence['raw']) for sentence in image['sentences']]
        print('Image: ', image_filepath)
        
        prompt = ['Rephrase the following sentences in a more creative way:']
        for caption in captions:
            prompt.append(f'  - {caption}')
        prompt.append('You should escape qouted strings within a string.')
        prompt.append('Rephrased sentences in json array:')
        request_completed = False
        retry_times = 0
        while not request_completed:
            try:
                print('Start completion task...')
                completion = await proxy.complete('\n'.join(prompt))
                content = completion['choices'][0]['message']['content']
                print(content)
                rephrased_sents = json.loads(content)
                if not isinstance(rephrased_sents, list):
                    raise ValueError('Returned result is not a list')

                augmented = {
                    'image': image['filename'],
                    'captions': []
                }

                for caption, sent in zip(captions, rephrased_sents):
                    print('-' * 120)
                    print('Caption: ', caption, '\nRephrased: ', sent)
                    _image = Image.open(image_filepath)
                    image_original_similarity = clip_model.get_image_text_similarity(_image, caption).item()
                    image_rephrased_similarity = clip_model.get_image_text_similarity(_image, sent).item()
                    original_rephrased_similarity = clip_model.get_texts_similarity(caption, sent).item()

                    print('Image + Original similarity: ', image_original_similarity)
                    print('Image + Rephrased similarity: ', image_rephrased_similarity)
                    print('Original + Rephrased similarity: ', original_rephrased_similarity)

                    augmented['captions'].append({
                        'original': caption,
                        'rephrased': sent,
                        'image_original_similarity': image_original_similarity,
                        'image_rephrased_similarity': image_rephrased_similarity,
                        'original_rephrased_similarity': original_rephrased_similarity,
                    })

                    augmented_flickr8k_metadata.append(augmented)
                    # Save to file per augmented item
                    with open(augmented_flickr8k_metadata_filepath, 'w') as stream:
                        json.dump(augmented_flickr8k_metadata, stream)
                    request_completed = True
            except Exception as e:
                print(e)
                if retry_times > 3:
                    print('Error: exceeded the maximum retry times, exit with 255')
                    sys.exit(255)
                else:
                    retry_times += 1
                    print('Retrying ...')
                    continue
        
if __name__ == '__main__':
   asyncio.run(main())