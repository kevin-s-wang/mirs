from sqlalchemy import Column, DateTime, String,  Text, text, ARRAY, Float, CHAR, Integer
from pgvector.sqlalchemy import Vector
from mirs.conf.config import config

from .database import Base

SERVER_NOW = text("now()")

class Image(Base):
    __tablename__ = 'images'

    id = Column(String(128), primary_key=True, index=True)
    path = Column(String(255), nullable=False)
    filename = Column(String(255), nullable=False)
    # Supported: [image/gif, image/jpeg, image/png]
    content_type = Column(String(64), nullable=False)
    # model for generating embeddings (openclip/clip)
    clip_model = Column(String(128), default=config.clip.get_model())
    clip_pretrained = Column(String(128), default=config.clip.get_pretrained())
    clip_emebddings_dim = Column(Integer, default=config.clip.get_embeddings_dim())
    embeddings = Column(Vector(config.clip.get_embeddings_dim()))

    tags = Column(ARRAY(String(255)))
    captions = Column(ARRAY(Text()))

    # EXIF metadata
    device_make = Column(String(255))
    device_model = Column(String(255))
    artist = Column(String(64))
    taken_at = Column(DateTime)
    original_taken_at = Column(DateTime)
    gps_latitude = Column(Float)
    gps_latitude_ref = Column(CHAR(1))
    gps_longitude = Column(Float)
    gps_longitude_ref = Column(CHAR(1))
    gps_altitude = Column(Float)
    gps_altitude_ref = Column(CHAR(1))

    created_at = Column(DateTime, nullable=False, server_default=SERVER_NOW)
    updated_at = Column(DateTime, nullable=False, server_default=SERVER_NOW, server_onupdate=SERVER_NOW)

