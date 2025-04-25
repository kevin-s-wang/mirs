import os
import io
import ollama
from fastapi import APIRouter, File, UploadFile, Form, status
from typing import List, Optional, Annotated, Dict, Any
import hashlib
import re
from confluent_kafka import Producer
import socket
import httpx
import numpy as np
import json
from mirs.conf import config
from sqlalchemy.orm import Session
from fastapi import Depends
from mirs.data.database import get_db
from mirs.data.schemas import ImageCreateResponse, ImageSearchResult, ImageModel, ImageMetadataModel
from mirs.data import models
from mirs.utils import get_image_metadata
from sklearn.metrics.pairwise import cosine_similarity
from pprint import pprint
from mirs.conf.config import config
from mirs.utils import resolve_response, QueryException


VLM = 'llava-llama3:latest'
LANG_MODEL = 'llama3.1:latest'

conf = {
    'bootstrap.servers': config.get_kafka_bootstrap_servers(), 
    'client.id': socket.gethostname()
}
producer = Producer(conf)
router = APIRouter()
embeddings_dim = config.clip.get_embeddings_dim()

def get_system_message() -> dict[str, Any]:
    
    system_prompt_file = os.path.join(config.root_dir, 'prompts', 'query_system_message.txt')
    with open(system_prompt_file, 'r') as stream:
        content = stream.read()

    return { "role": "system", "content": content}


# def get_system_message() -> dict[str, Any]:
#     content = """
# You are an image retrieval assistant.Your responsibility is to generate a JSON object based on user context enclosed with <context></context>.The context consists of a user text and 0 to 3 images. 
# Also make sure that you carefully adhere to the following requirements.

# Requirements:
#  - The images are enclosed with <image order=0>...</image>, <image order=1>...</image>, etc.
#  - The user text is enclosed with <user-text></user-text>, it can be an instruction, a text with key information or both.
#  - the user text may attempt to look up information from images, such as color, activity, people and environment etc.
#  - Ensure the images are in the correct order.
#  - Ensure the query string "q" is brief and concise.
#  - Generate a JSON object containing an error message if a conflict occurs.
#  - Convert a date or datetime object to the 'YYYY-MM-DD HH:mm:ss' format.
#  - Strictly adhere to the given schema.

# Search query schema:
# ```json
# {
#   "q": <string|null>,
#   "taken_at": <string|null>,
#   "taken_by": <string|null>,
#   "created_at": <string|null>,
#   "updated_at": <string|null>
# }
# ```
# Error schema:
# ```json
# {  “error”: <string> }
# ```

# """
#     return {
#         "role": "system",
#         "content": content,
#     }


wrap_image = lambda i, x: f'<image order={i}>{x}</image>'

async def describe_image(client: ollama.AsyncClient, images: List[io.BytesIO]):
    for index, im in enumerate(images):
        response = await client.chat(
            VLM,
            messages=[
                {
                    "role": "user",
                    "content": "What's in this image?",
                    "images": [im]
                }
            ],
        )
        yield wrap_image(index, response['message']['content'])

async def get_hybrid_semantic_query(client: ollama.AsyncClient, text: str, images: List[io.BytesIO]) -> Dict[str, Any]:
    messages = [get_system_message()]

    image_descriptions = []
    async for image_description in describe_image(client, images):
        image_descriptions.append(image_description)
    
    image_descriptions = '\n'.join(image_descriptions)
    user_context = f"""
<context>
    {image_descriptions}
    <user-text>{text}</user-text>
</context>

Output JSON Object:

"""
    
    messages.append({ "role": "user", "content": user_context})
    pprint(messages)
    response = await client.chat(LANG_MODEL, messages=messages)
    print(response['message']['content'])
    return resolve_response(response['message']['content'])

async def rerank(client: ollama.AsyncClient, q: str, images: List[ImageModel]) -> List[ImageModel]:

    if len(images) <= 1:
        return images
    
    for image in images:
        if not image.captions:
            continue

        texts = [q] + image.captions
        response = await client.embed(model=LANG_MODEL, input=texts)
        q_embeds = response['embeddings'][0]
        image_embeds = response['embeddings'][1:]
        image.similarity = np.max(cosine_similarity([q_embeds], image_embeds))
        
    sorted_images = sorted(images, key=lambda x: x.similarity, reverse=True)
    return sorted_images


async def rerank_1(client: ollama.AsyncClient, q: str, images: List[ImageModel]) -> List[ImageModel]:

    if len(images) <= 1:
        return images
    
    for image in images:
        if not image.captions:
            continue
        
        all_captions = ''.join(image.captions)
        texts = [q, all_captions]
        response = await client.embed(model=LANG_MODEL, input=texts)
        q_embeds = response['embeddings'][0]
        image_embeds = response['embeddings'][1:]
        image.similarity = np.max(cosine_similarity([q_embeds], image_embeds))
        
    sorted_images = sorted(images, key=lambda x: x.similarity, reverse=True)
    return sorted_images

async def rerank_2(client: ollama.AsyncClient, q: str, images: List[ImageModel]) -> List[ImageModel]:

    if len(images) <= 1:
        return images
    
    for image in images:
        if not image.captions:
            continue
        
        texts = [q] + image.captions
        response = await client.embed(model=LANG_MODEL, input=texts)
        q_embeds = response['embeddings'][0]
        image_embeds = response['embeddings'][1:]
        image.similarity = np.average(cosine_similarity([q_embeds], image_embeds))
        
    sorted_images = sorted(images, key=lambda x: x.similarity, reverse=True)
    return sorted_images

async def rerank_3(client: ollama.AsyncClient, q: str, images: List[ImageModel]) -> List[ImageModel]:

    if len(images) <= 1:
        return images
    
    for image in images:
        if not image.captions:
            continue
        
        texts = [q] + image.captions
        response = await client.embed(model=LANG_MODEL, input=texts)
        q_embeds = response['embeddings'][0]
        captions_embeds = response['embeddings'][1:]
        clipped_scores = np.clip(cosine_similarity([q_embeds], captions_embeds), image.similarity, 1.)
        image.similarity = np.average(clipped_scores)
        
    sorted_images = sorted(images, key=lambda x: x.similarity, reverse=True)
    return sorted_images

# async def rerank_4(q: str, images: List[ImageModel]) -> List[ImageModel]:

#     if len(images) <= 1:
#         return images
    
#     for image in images:
#         if not image.captions:
#             continue
        
#         texts = [q] + image.captions
#         async with httpx.AsyncClient() as http:
#             data: httpx.Response = await http.post(config.get_embeddings_api_url() + '/texts', follow_redirects=True, json=texts)
#             embeddings = np.array(data.json(), dtype=np.float32)

#             q_embeds = embeddings[0]
#             captions_embeds = embeddings[1:]
#         clipped_scores = np.clip(cosine_similarity([q_embeds], captions_embeds), image.similarity, 1.)
#         image.similarity = np.average(clipped_scores)
        
#     sorted_images = sorted(images, key=lambda x: x.similarity, reverse=True)
#     return sorted_images


@router.post('/search', response_model=ImageSearchResult)
async def search(
        prompt: Annotated[
            Optional[str], Form()
        ] = None,
        images: Annotated[
            list[UploadFile], File(description='images to search with') 
        ] = [],
        offset: int = 0,
        limit: int = 20,
        db: Session = Depends(get_db)
):
    print('Received request -->\n', 'prompt:', prompt, '\n# of images:', len(images), '\noffset:', offset, '\nlimit:', limit)
    
    if not prompt and not images:
        raise ValueError('Prompt or images must be provided')

    use_default_prompt = False
    if not prompt:
        use_default_prompt = True
        prompt = 'Generate the query based on the images provided'

    image_semantic_retrieval_only = use_default_prompt and len(images) == 1

    if not image_semantic_retrieval_only:
        client = ollama.AsyncClient()
        images_in_bytes = [io.BytesIO(await image.read()) for image in images]

        query = await get_hybrid_semantic_query(client, prompt, images_in_bytes)
        
        if 'error' in query:
            raise QueryException(message=query['error'])

        if 'limit' in query and query['limit'] is not None:
            limit = query['limit']
        if 'offset' in query and query['offset'] is not None:
            offset = query['offset']

    async with httpx.AsyncClient() as http:
        q_embeddings = None
        if not image_semantic_retrieval_only:
            data: httpx.Response = await http.post(config.get_embeddings_api_url(), follow_redirects=True, data={'text':  f'a photo of {query["q"]}'})
            q_embeddings = np.array(data.json(), dtype=np.float32)
        else:
            image_files = [('images', (x.filename, x.file, x.content_type)) for x in images]
            data: httpx.Response = await http.post(config.get_embeddings_api_url(), follow_redirects=True, 
                                                     files=image_files)            
            image_embeddings_list: List[List[float]] = data.json()
            q_embeddings = np.array(image_embeddings_list[0], dtype=np.float32)
            
        result_set = db.query(models.Image) \
                .filter(models.Image.embeddings != None) \
                .order_by(models.Image.embeddings.cosine_distance(q_embeddings.tolist())) \
                .offset(offset) \
                .limit(limit)
        db.commit()

        _images = []
        for row in result_set:
            _real_filename = os.path.basename(row.path)

            metadata = ImageMetadataModel(
                device_make=row.device_make,
                device_model=row.device_model,
                artist=row.artist,
                taken_at=row.taken_at,
                original_taken_at=row.original_taken_at,
                gps_latitude=row.gps_latitude,
                gps_latitude_ref=row.gps_latitude_ref,
                gps_longitude=row.gps_longitude,
                gps_longitude_ref=row.gps_longitude_ref,
                gps_altitude=row.gps_altitude,
                gps_altitude_ref=row.gps_altitude_ref,
            )
            
            _images.append(ImageModel(
                    id=row.id, 
                    similarity=cosine_similarity([q_embeddings], [row.embeddings]),
                    filename=row.filename,
                    captions=row.captions,
                    tags=row.tags,
                    metadata=metadata,
                    created_at=row.created_at,
                    updated_at=row.updated_at,
                    url=config.get_images_url(_real_filename),  
                    content_type=row.content_type))
            
    if not image_semantic_retrieval_only:
        data = await rerank_3(client, query['q'], _images)
    else:
        data = _images
    return ImageSearchResult(limit=limit, offset=offset, data=data)


# @router.post('/search', response_model=ImageSearchResult)
# async def search(
#         prompt: Annotated[
#             Optional[str], Form()
#         ] = None,
#         images: Annotated[
#             list[UploadFile], File(description='images to search with') 
#         ] = [],
#         offset: int = 0,
#         limit: int = 12,
#         db: Session = Depends(get_db)
# ):
#     _images = []
#     async with httpx.AsyncClient() as client:
#         image_embeddings: np.ndarray = np.array([])
#         text_embeddings: np.ndarray = np.array([])

#         if prompt:
#             data: httpx.Response = await client.post(config.get_embeddings_api_url(), follow_redirects=True, data={'text':  prompt})
#             text_embeddings_arr = data.json()
#             text_embeddings = np.array(text_embeddings_arr, dtype=np.float32)
        
#         if images:
#             image_files = [('images', (x.filename, x.file, x.content_type)) for x in images]
#             data: httpx.Response = await client.post(config.get_embeddings_api_url(), follow_redirects=True, 
#                                                      files=image_files)
#             image_embeddings_list: List[List[float]] = data.json()
            
#             for _image_embeddings in image_embeddings_list:
#                 if image_embeddings.size == 0:
#                     image_embeddings = np.array(_image_embeddings, dtype=np.float32)
#                     continue
#                 image_embeddings += _image_embeddings

#         embeddings: np.ndarray
  
#         if text_embeddings.size == embeddings_dim and image_embeddings.size == embeddings_dim:
#             embeddings = text_embeddings + image_embeddings
#         elif text_embeddings.size == embeddings_dim:
#             embeddings = text_embeddings
#         elif image_embeddings.size == embeddings_dim:
#             embeddings = image_embeddings
#         else:
#             raise ValueError('No text or image embeddings available')        

#         result_set = db.query(models.Image) \
#                 .filter(models.Image.embeddings != None) \
#                 .order_by(models.Image.embeddings.cosine_distance(embeddings.tolist())) \
#                 .offset(offset) \
#                 .limit(limit)
#         db.commit()
    
#         for row in result_set:
#             _real_filename = os.path.basename(row.path)

#             metadata = ImageMetadataModel(
#                 device_make=row.device_make,
#                 device_model=row.device_model,
#                 artist=row.artist,
#                 taken_at=row.taken_at,
#                 original_taken_at=row.original_taken_at,
#                 gps_latitude=row.gps_latitude,
#                 gps_latitude_ref=row.gps_latitude_ref,
#                 gps_longitude=row.gps_longitude,
#                 gps_longitude_ref=row.gps_longitude_ref,
#                 gps_altitude=row.gps_altitude,
#                 gps_altitude_ref=row.gps_altitude_ref,
#             )
            
#             _images.append(ImageModel(
#                     id=row.id, 
#                     similarity=cosine_similarity([embeddings], [row.embeddings])[0][0],
#                     filename=row.filename,
#                     captions=row.captions,
#                     tags=row.tags,
#                     metadata=metadata,
#                     created_at=row.created_at,
#                     updated_at=row.updated_at,
#                     url=config.get_images_url(_real_filename),  
#                     content_type=row.content_type))
            
#     return ImageSearchResult(limit=limit, offset=offset, data=_images)

@router.post('', response_model=ImageCreateResponse)
async def upload(image: UploadFile, captions: List[str] = [], tags: List[str] = [], db: Session = Depends(get_db)):

    # Ensure images directory exists
    images_dir = config.get_images_dir()
    os.makedirs(images_dir, exist_ok=True)
    
    contents = await image.read()
    _, ext = os.path.splitext(image.filename)
    image_hash = hashlib.sha512(contents).hexdigest()
    image_filename = image_hash + ext    
    image_file_path = os.path.join(images_dir, image_filename)

    response_image = ImageCreateResponse(
                        id=image_hash, 
                        filename=image.filename, 
                        url=config.get_images_url(image_filename),
                        content_type=image.content_type)
    
    if not os.path.exists(image_file_path):
        with open(image_file_path, 'wb') as image_file:
            image_file.write(contents)

        metadata = get_image_metadata(image_file_path)

        new_image = models.Image(
            id=image_hash,
            filename=image.filename,
            content_type=image.content_type,
            captions=captions,
            tags=tags,
            # model=config.clip.get_model(),
            # pretrained=config.clip.get_pretrained(),
            # Metadata from EXIF
            device_make=metadata.device_make,
            device_model=metadata.device_model,
            artist=metadata.artist,
            taken_at=metadata.taken_at,
            original_taken_at=metadata.original_taken_at,
            gps_latitude=metadata.gps_latitude,
            gps_latitude_ref=metadata.gps_latitude_ref,
            gps_longitude=metadata.gps_longitude,
            gps_longitude_ref=metadata.gps_longitude_ref,
            gps_altitude=metadata.gps_altitude,
            gps_altitude_ref=metadata.gps_altitude_ref,

            path=os.path.join(config.get_relative_images_dir(), image_filename),
        )
        db.add(new_image)
        db.commit()
        # Produce message to Kafka

    def on_delivery(err, msg):
        if err:
            print('Message delivery failed: {}'.format(err))
        else:
            print('Message delivered to {} [{}]'.format(msg.topic(), msg.partition()))
    producer.produce(topic='image-events', key='created', value=image_hash, on_delivery=on_delivery)
    producer.flush()
    return response_image

@router.get('/{image_id}/tags', status_code=status.HTTP_200_OK)
async def get_image_tags(image_id: str, db: Session = Depends(get_db)):
    single_result = db.query(models.Image.tags).filter(models.Image.id == image_id).one_or_none()
    return single_result.tags if single_result else []


@router.post('/{image_id}/tags', status_code=status.HTTP_200_OK)
async def update_image_tags(image_id: str, db: Session = Depends(get_db)):
    image = db.query(models.Image).filter(models.Image.id == image_id).one_or_none()
    if not image:
        raise ValueError(f'Image with id {image_id} not found')

    client = ollama.AsyncClient()
    if len(image.captions) > 0:
        formatted_captions = '\n'.join([f'- {caption}' for caption in image.captions])
        prompt = f'Generate at most 10 tags referring to the following descriptions:\n\n{formatted_captions}\n\nResponse format MUST strictly follow the below example:\n["tag1", "tag2",...]\n\Here are the generated tags:\n'
        result =  await client.generate(LANG_MODEL, prompt=prompt)
        print('---------------- result: ', result)
        # tags = unsafe_extract_response(result['response'])
        tags = json.loads(result['response'])

    if tags and len(tags) > 0:
        image.tags = tags
        db.commit()
        return image.tags
    else:
        return []
    
def unsafe_extract_response(text: str) -> List[str]:
    p = r"```(?:\w+\s+)?(.*?)```"
    matches = re.findall(p, text, re.DOTALL)
    blocks = [block.strip() for block in matches]
    return json.loads(blocks[0])

# @router.post('/{image_id}/tags', status_code=status.HTTP_200_OK)
# async def update_image_tags(image_id: str, db: Session = Depends(get_db)):
#     image = db.query(models.Image).filter(models.Image.id == image_id).one_or_none()
#     if not image:
#         raise ValueError(f'Image with id {image_id} not found')

#     llm = LanguageModelProxy(config.llm)
#     if len(image.captions) > 0:
#         formatted_captions = '\n'.join([f'- {caption}' for caption in image.captions])
#         prompt = f'Generate at most 10 tags referring to the following descriptions:\n\n{formatted_captions}\n\nTags in json array:'
#         result =  await llm.complete(prompt)
#         print('---------------- result: ', result)
#         tags = json.loads(result['choices'][0]['message']['content'])

#     if tags and len(tags) > 0:
#         image.tags = tags
#         db.commit()
#         return image.tags
#     else:
#         return []
    

# @router.post('/batch', status_code=status.HTTP_200_OK)
# async def upload_batch(images: List[UploadFile] = [], db: Session = Depends(get_db)):
#     for image in images:
#         await upload(image, db=db)
#     return {'message': 'ok'}