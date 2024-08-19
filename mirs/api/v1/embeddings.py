import os
import io
from fastapi import APIRouter, File, Form, UploadFile, status, Depends
from mirs.data.database import get_db
from sqlalchemy.orm import Session
from PIL import Image
from mirs.ai.models import clip_model
from mirs.data import models
from typing import Annotated, Any, Optional

from pydantic import BaseModel

class EmbeddingsRequestBody(BaseModel):
    type: str
    value: Any

router = APIRouter()

def get_image_embeddings(image: Image.Image):
    image_embeddings = clip_model.get_image_embeddings(image)
    return image_embeddings.cpu().numpy().tolist()[0]

@router.post('', status_code=status.HTTP_200_OK)
async def get_embeddings(
        text: Annotated[
            Optional[str], Form()
        ] = None,
        image_id: Annotated[
            Optional[str], Form()
        ] = None,
        images: Annotated[
            list[UploadFile], File(description='Single or multiple images to embeddings') 
        ] = [], db: Session = Depends(get_db)):
    
    if text:
        text_features = clip_model.get_text_embeddings([text])
        return text_features.cpu().numpy().tolist()[0]
    elif image_id:
        image = db.query(models.Image).filter(models.Image.id == image_id).one_or_none()
        if not image:
            raise ValueError(f'Image({image_id}) does not exist')
        image_file_path = os.path.abspath(image.path)
        return get_image_embeddings(Image.open(image_file_path))
    elif images:
        embeddings = []
        for image in images:
            contents =  await image.read()
            embeddings.append(get_image_embeddings(Image.open(io.BytesIO(contents))))
        return embeddings
    else:
        raise ValueError('text or images must be provided')

