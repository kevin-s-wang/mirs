from sqlalchemy.orm import Session
from . import models, schemas
from typing import List

def get_image(db: Session, image_id: str) -> models.Image:
    return db.query(models.Image).filter(models.Image.id == image_id).first()

def get_images(db: Session, skip: int = 0, limit: int = 100) -> List[models.Image]:
    return db.query(models.Image).offset(skip).limit(limit).all()

def create_image(db: Session, image: schemas.ImageCreate) -> models.Image:
    
    db_image_file = models.ImageFile(id=image.ref, path=image.path, content_type=image.content_type)
    db.add(db_image_file)

    db_image = models.Image(filename=image.filename, ref=image.ref)
    db.add(db_image)

    db.commit()
    # db.refresh(db_image)
    return db_image


def update_image_embeddings(db: Session, image_id: str, embeddings: list[float]) -> None:
    db_image = db.query(models.Image).filter(models.Image.id == image_id).first()
    db_image.embeddings = embeddings
    db.commit()
  

