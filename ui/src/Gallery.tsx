import React, { useState } from "react";
import { Button, Card, CardActionArea, CardContent, CardMedia, Container, Grid, Modal, Paper, Typography } from "@mui/material";
import { Image } from "./data";
import ImageDetail from "./ImageDetail";
import LocalOfferOutlinedIcon from '@mui/icons-material/LocalOfferOutlined';


const modalStyles = {
    position: 'absolute',
    width: '100%',
    height: '100%',
    padding: '13px',
    bgcolor: 'background.paper',
};

interface GalleryProps {
    images: Image[];
    onLoadMore?: () => Promise<void>;
}

const Gallery = ( { images, onLoadMore }: GalleryProps ) => {

    const onImageClick = (image: Image) => setClickedImage(image);
    const onModalClose = () => setClickedImage(null);

    const [ clickedImage, setClickedImage ] = useState<Image | null>(null);
    return (
        images.length > 0 ?
        <Grid container spacing={{ xs: 2, md: 3 }} columns={{ xs: 4, sm: 8, md: 12 }} sx={{overflowY: 'auto', padding: '13px'}} >
        {
        images.map(image => (
            <Grid item xs={2} sm={4} md={3} key={'grid-' + image.id}>
                <Card key={image.id} sx={{border: 'none', boxShadow: 'rgba(33, 35, 38, 0.1) 0px 10px 10px -10px'}}>
                    <CardActionArea onClick={() => onImageClick(image)}>
                    <CardMedia component='img' image={image.url} sx={{objectFit: 'cover', height: 200, '&:hover': { opacity: 0.5}}} alt={image.alt} />
                    </CardActionArea>
                    <CardContent>
                      <Typography variant="body2" color="text.secondary" sx={{textAlign: 'right'}}>Similarity: {image.similarity.toPrecision(4)}</Typography>
                      <Typography gutterBottom variant="body1" component="div">{image.captions[0]}</Typography>
                      {
                        image.tags.length > 0 ?
                          <div className="tags" style={{display: 'flex', flexDirection: 'row'}}>
                            <LocalOfferOutlinedIcon titleAccess="Tags" sx={{fontSize: '20px', marginRight: '5px', color: '#666'}} />
                            <Typography variant="body2" color="text.secondary">{image.tags.join(', ')}</Typography>
                          </div>
                          : null
                      }
                    </CardContent>
                </Card>
            </Grid>)
        )
        }

   
        <Grid item xs={12} sm={12} md={12} sx={{textAlign: 'center'}}>
            <Button variant="contained"  onClick={onLoadMore}>Load More</Button>
        </Grid>

        <Modal open={clickedImage !== null} onClose={onModalClose}>
          <Paper sx={modalStyles}>
            <ImageDetail image={clickedImage} onClose={onModalClose} />
          </Paper>
        </Modal>
      </Grid>
      : 
      <Container sx={{height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center'}}>
        <Typography variant="body2" color="text.secondary">No images found.</Typography>
      </Container>
    );
};

export default Gallery;