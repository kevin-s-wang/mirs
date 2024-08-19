import React from "react";
import { Card, CardMedia, Grid, IconButton } from "@mui/material";
import DeleteIcon from '@mui/icons-material/Delete';


interface ImagePreviewProps {
    images: File[];
    onRemove: (index: number) => void;
}

const ImagePreview = ({ images, onRemove }: ImagePreviewProps) => {
    return ( 
        images.length > 0 ? 
        <Grid container wrap='nowrap' gap={2} >
        {
        images.map((image, index) => (
            <Card key={`preview-image-${index}`} sx={{position: 'relative', mt: 2, p: 1, boxShadow: 'rgba(0, 0, 0, 0.05) 0px 0px 0px 1px',  width: 155, 
            '&:hover': {
                '& .delete': {
                visibility: 'visible',
                backgroundColor: 'rgba(0, 0, 0, 0.5)',

                },
                '& .preview-image-item': {
                opacity: 0.5,
                }

            }}}>

                <CardMedia className='preview-image-item' component='img' image={URL.createObjectURL(image)} sx={{objectFit: 'cover', height: 90}} alt={image.name} />
                <IconButton size='small' 
                    sx={{position: 'absolute', top: 0, right: 0, visibility: 'hidden' }} 
                    className='delete' type='button'  aria-label="Remove" onClick={() => onRemove(index)}>
                    <DeleteIcon sx={{color: 'white'}} fontSize='small' />
                </IconButton>
            </Card>
            ))
        }
        </Grid>
        : null
    );
};

export default ImagePreview;