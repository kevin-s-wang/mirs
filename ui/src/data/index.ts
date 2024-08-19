export interface ImageMetadata {
    device_make: string;
    device_model: string;
    artist: string;
    taken_at: Date;
    original_taken_at: Date;
    gps_latitude: number;
    gps_latitude_ref: string;
    gps_longitude: number;
    gps_longitude_ref: string;
    gps_altitude: number;
    gps_altitude_ref: string;
}

export interface Image {
    id: string,
    url: string;
    filename: string;
    alt?: string;
    similarity: number;
    created_at: string;
    updated_at: string;
    metadata: ImageMetadata;
    tags: string[];
    captions: string[];
}