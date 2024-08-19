import logging

# logging.basicConfig(level=logging.DEBUG)

import httpx
import uvicorn
import asyncio

from typing import List
from contextlib import asynccontextmanager
from fastapi import FastAPI, status, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from mirs.api import api_router
from mirs.data import models
from mirs.data.database import engine, get_db
from mirs.data.crud import update_image_embeddings
from aiokafka import AIOKafkaConsumer
from mirs.conf.config import config
from mirs.utils import QueryException



async def on_embedding_message(msg):
    # We should not fail the consumer if there is an error updating the image embeddings
    try:
        image_id = msg.value.decode('utf-8')
        print('Received message: {}'.format(image_id))
        async with httpx.AsyncClient() as client:
            print('Sending request to embeddings service')
            data: httpx.Response = await client.post(config.get_embeddings_api_url(), follow_redirects=True,    
                                        data={'image_id': image_id})
            image_embeddings: List[float] = data.json()
            db = next(get_db())
            update_image_embeddings(db, image_id, image_embeddings)
    except Exception as e:
        print('Error updating image embeddings: ', e)

async def embedding():
    consumer = AIOKafkaConsumer(
                    'image-events', 
                    bootstrap_servers=config.get_kafka_bootstrap_servers(), 
                    group_id='embeddings')
    
    await consumer.start()
    print('--------------------------- Embedding consumer started ---------------------------')
    try:
        async for msg in consumer:
            await on_embedding_message(msg)
    except Exception as e:
        print('Embedding consumer error: ', e)
    finally:
       print('Closing embedding consumer...')
       await consumer.stop()

# async def extract_exif():
#     consumer = AIOKafkaConsumer(
#                     'image-events', 
#                     bootstrap_servers=config.get_kafka_bootstrap_servers(), 
#                     group_id='exif')

#     await consumer.start()
#     try:
#         async for msg in consumer:
#             print(msg)
#     except Exception as e:
#         print('EXIF consumer error: ', e)
#     finally:
#        print('Closing EXIF consumer...')
#        await consumer.stop()

async def on_tagging_message(msg):
    try:
        image_id = msg.value.decode('utf-8')
        print('Received message: {}'.format(image_id))

        async with httpx.AsyncClient() as client:
            print('Sending request to tagging service')

            tags_api_url = config.server.get_tags_api_url(image_id)
            data: httpx.Response = await client.post(tags_api_url, follow_redirects=True)
            print(f'Tags({image_id}): ', data.json())
    except Exception as e:
        print('Error updating image tags: ', e)
        
async def tagging():
    consumer = AIOKafkaConsumer(
                    'image-events', 
                    bootstrap_servers=config.get_kafka_bootstrap_servers(), 
                    group_id='tagging')
    await consumer.start()
    print('--------------------------- Tagging consumer started ---------------------------')
    try:
        async for msg in consumer:
            await on_tagging_message(msg)
    except Exception as e:
        print('Tagging consumer error: ', e)
    finally:
       print('Closing tagging consumer...')
       await consumer.stop()

@asynccontextmanager
async def lifespan(app: FastAPI):
    models.Base.metadata.create_all(bind=engine)
    asyncio.get_event_loop().create_task(embedding())
    # asyncio.get_event_loop().create_task(extract_exif())
    asyncio.get_event_loop().create_task(tagging())
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=['*'])

@app.exception_handler(QueryException)
async def query_exception_handler(_: Request, err: QueryException):
    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={
            "code": status.HTTP_400_BAD_REQUEST,
            "message": err.message})


@app.get('/health', status_code=status.HTTP_200_OK)
async def health():
    return {'status': 'ok'}

app.include_router(api_router, prefix='/api')

async def main():
    server_conf = uvicorn.Config('server:app', 
                                 host=config.server.get_host(), 
                                 port=config.server.get_port(),  
                                 reload=True, 
                                 log_level=config.get_log_level())
    server = uvicorn.Server(server_conf)
    await server.serve()

if __name__ == '__main__':
    asyncio.run(main())
