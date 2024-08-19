from pydantic import BaseModel
from typing import List
from datetime import datetime

class ImageBase(BaseModel):
    id: str
    filename: str
    path: str
    content_type: str


class ImageCreateResponse(BaseModel):
    id: str
    filename: str
    url: str
    content_type: str
    tags: List[str] = []
    captions: List[str] = []


class ImageCreate(ImageBase):
    pass


class ImageMetadataModel(BaseModel):
    device_make: str | None
    device_model: str | None
    artist: str | None
    taken_at: datetime | None
    original_taken_at: datetime | None
    gps_latitude: float | None
    gps_latitude_ref: str | None
    gps_longitude: float | None
    gps_longitude_ref: str | None
    gps_altitude: float | None
    gps_altitude_ref: str | None

class ImageModel(BaseModel):
    id: str
    filename: str
    url: str
    content_type: str
    similarity: float
    tags: List[str] = []
    captions: List[str] = []
    
    metadata: ImageMetadataModel

    created_at: datetime
    updated_at: datetime


class ImageSearchResult(BaseModel):
    limit: int = 12
    offset: int = 0
    data: List[ImageModel] = []

