import os
import re
import json
from dataclasses import dataclass
from datetime import datetime
from PIL import Image, ExifTags
from typing import Dict, List, Any


@dataclass
class ImageMetadata:
    device_make: str
    device_model: str
    artist: str
    taken_at: datetime
    original_taken_at: datetime
    gps_latitude: float
    gps_latitude_ref: str
    gps_longitude: float
    gps_longitude_ref: str
    gps_altitude: float
    gps_altitude_ref: str


def parse_exif_datetime(exif_datetime: str) -> datetime | None:
    parsed : datetime | None = None
    if exif_datetime:
        try:
            parsed = datetime.strptime(exif_datetime, '%Y:%m:%d %H:%M:%S')
        except ValueError:
            pass
    return parsed
    

def get_image_metadata(filename: str) -> ImageMetadata:
    if not os.path.exists(filename):
        raise FileNotFoundError(f'{filename} not found')
    
    im = Image.open(filename)
    exif_info = im.getexif()
    
    metadata = ImageMetadata(
        device_make=exif_info.get(ExifTags.Base.Make),
        device_model=exif_info.get(ExifTags.Base.Model),
        artist=exif_info.get(ExifTags.Base.Artist),
        taken_at=parse_exif_datetime(exif_info.get(ExifTags.Base.DateTime)),
        original_taken_at=parse_exif_datetime(exif_info.get(ExifTags.Base.DateTimeOriginal)),
        gps_latitude=exif_info.get(ExifTags.GPS.GPSLatitude),
        gps_latitude_ref=exif_info.get(ExifTags.GPS.GPSLatitudeRef),
        gps_longitude=exif_info.get(ExifTags.GPS.GPSLongitude),
        gps_longitude_ref=exif_info.get(ExifTags.GPS.GPSLongitudeRef),
        gps_altitude=exif_info.get(ExifTags.GPS.GPSAltitude),
        gps_altitude_ref=exif_info.get(ExifTags.GPS.GPSAltitudeRef),
    )
    return metadata


def load_flickr8k_metadata(datasets_dir: str) -> Dict:
    flickr8k_metadata_filepath = os.path.join(datasets_dir, 'dataset_flickr8k.json')
    with open(flickr8k_metadata_filepath, 'r') as f:
        return json.load(f)
    

def extract_json_objects(markdown: str) -> List[Any]:
    """
    Extract JSON objects from a given markdown string enclosed between ````json\n` and ````\n`.

    Args:
        markdown_str (str): The input markdown string.

    Returns:
        List[Any]: A list of extracted JSON objects parsed as Python data types.
    """
    pattern = r'```\s*json\s*(.*?)\s*```'
    matches = re.findall(pattern, markdown, re.MULTILINE | re.DOTALL)

    json_objects = []
    for match in matches:
        try:
            json_obj = json.loads(match.strip())
            json_objects.append(json_obj)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON found: {match.strip()}. Error: {e}")
            continue

    return json_objects


def resolve_response(markdown: str) -> Any:
    """
    Extract the first JSON object from a given markdown string enclosed between ````json\n` and ````\n`.

    Args:
        markdown (str): The input markdown string.

    Returns:
        Any: The extracted JSON object parsed as a Python data type.
    """

    default_error_response = { 'error': 'No valid JSON object found in the response.'}
    
    json_objects = extract_json_objects(markdown)
    if not json_objects:
        return default_error_response
    
    for json_object in json_objects:
        if 'q' in json_object and json_object['q'] is not None:
            return json_object
        
        if 'error' in json_object:
            return json_object
        
    return default_error_response

class QueryException(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
