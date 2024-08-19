from fastapi import APIRouter
from .v1 import images
from .v1 import embeddings

api_router = APIRouter()

# API v1
api_router.include_router(images.router, prefix='/v1/images')
api_router.include_router(embeddings.router, prefix='/v1/embeddings')

# API v2