import React from "react";
import { Image } from "./data";
import { Box, Chip, IconButton, Link, Stack, Table, TableBody, TableCell, TableContainer, TableRow, Typography } from "@mui/material";
import CloseIcon from '@mui/icons-material/Close';



interface ImageDetailProps {
    image: Image | null;
    onClose?: () => void;
}
const NOOP = () => {};

const ImagePropertySectionHeader = ({title}: {title: string}) => 
        <Typography variant="h6" sx={{marginBottom: '8px'}}>{title}</Typography>;

const ImageDetail = ({ image, onClose = NOOP }: ImageDetailProps) => {

    return (
        image ? 
        <Stack className="image-detail">
            <Box sx={{textAlign: 'right'}}>
                <IconButton type='button' aria-label="Close Modal" onClick={onClose}>
                    <CloseIcon />
                </IconButton>
            </Box>
            <Stack flexDirection='row' sx={{width: '100%', height: '100%'}}>
                <Box sx={{flexGrow: 1, alignSelf: 'center', textAlign: 'center', p: '8px'}}>
                    <img src={image.url}
                        style={{objectFit: 'cover'}}
                        alt={image.filename}
                        loading="lazy"/> 
                </Box>
                
                <Stack sx={{minWidth: 400, maxWidth: '40%', marginLeft: 'auto'}} gap={4}>
                    <Box className='basic'>
                        <TableContainer>
                            <ImagePropertySectionHeader title="Basic" />
                            <Table size="small">
                                <TableBody>
                                    <TableRow>
                                        <TableCell><Typography variant="body1" sx={{color: '#666'}}>Filename</Typography></TableCell>
                                        <TableCell><Link href={image.url} title="Open image in new tab" underline="hover" target="_blank" rel="noreferrer" >{image.filename}</Link></TableCell>
                                    </TableRow>
                                    <TableRow>
                                        <TableCell><Typography variant="body1" sx={{color: '#666'}}>Similarity</Typography></TableCell>
                                        <TableCell><Typography variant="body1" sx={{color: '#666'}}>{image.similarity.toPrecision(4)}</Typography></TableCell>
                                    </TableRow>
                                    <TableRow>
                                        <TableCell><Typography variant="body1" sx={{color: '#666'}}>Created at</Typography></TableCell>
                                        <TableCell><Typography variant="body1" sx={{color: '#666'}}>{image.created_at}</Typography></TableCell>
                                    </TableRow>
                                    <TableRow>
                                        <TableCell><Typography variant="body1" sx={{color: '#666'}}>Last updated at</Typography></TableCell>
                                        <TableCell><Typography variant="body1" sx={{color: '#666'}}>{image.updated_at}</Typography></TableCell>
                                    </TableRow>
                                </TableBody>
                            </Table>
                        </TableContainer>
                    </Box>
                    <Box className='tags'>
                        <ImagePropertySectionHeader title="Tags" />
                        <Box className="tags-content" sx={{gap: 1, display: 'flex', flexDirection: 'row'}}>
                            {
                                image.tags.length > 0 ? 
                                image.tags.map((tag, index) => (<Chip key={`tag-${image.id}-${index}`} variant="outlined" color="primary" size="small" label={tag} /> ))
                                :
                                <Typography style={{color: '#666', marginRight: '13px'}}>No tags</Typography>
                            }   
                        </Box>

                    </Box>
                    <Box className='captions'>
                        <ImagePropertySectionHeader title="Captions" />
                        <Box className="captions-content">
                            {
                                image.captions.length > 0 ? 
                                image.captions.map((caption, index) => (<Typography key={`caption-${image.id}-${index}`} variant='body1'  sx={{color: '#666'}}>{index+1}. {caption}</Typography>))
                                : 
                                <Typography style={{color: '#666', marginRight: '13px'}}>No captions</Typography>
                            }
                        </Box>
                    </Box>

                    <Box className='more-info'>
                        <TableContainer>
                        <ImagePropertySectionHeader title="More information" />
                            {
                                image.metadata ?
                                <Table size="small">
                                    <TableBody>
                                        <TableRow>
                                            <TableCell><Typography variant="body1" sx={{color: '#666'}}>Device make</Typography></TableCell>
                                            <TableCell><Typography variant="body1" sx={{color: '#666'}}>{image.metadata.device_make ?? 'na'}</Typography></TableCell>                                    </TableRow>
                                        <TableRow>
                                            <TableCell><Typography variant="body1" sx={{color: '#666'}}>Device model</Typography></TableCell>
                                            <TableCell><Typography variant="body1" sx={{color: '#666'}}>{image.metadata.device_model ?? 'na'}</Typography></TableCell>
                                        </TableRow>
                                        <TableRow>
                                            <TableCell><Typography variant="body1" sx={{color: '#666'}}>Artist</Typography></TableCell>
                                            <TableCell><Typography variant="body1" sx={{color: '#666'}}>{image.metadata.artist ?? 'na'}</Typography></TableCell>
                                        </TableRow>
                                        <TableRow>
                                            <TableCell><Typography variant="body1" sx={{color: '#666'}}>GPS altitude</Typography></TableCell>
                                            <TableCell><Typography variant="body1" sx={{color: '#666'}}>{image.metadata.gps_altitude ?? 'na'}</Typography></TableCell>
                                        </TableRow>
                                        <TableRow>
                                            <TableCell><Typography variant="body1" sx={{color: '#666'}}>GPS altitude ref</Typography></TableCell>
                                            <TableCell><Typography variant="body1" sx={{color: '#666'}}>{image.metadata.gps_altitude_ref ?? 'na'}</Typography></TableCell>
                                        </TableRow>
                                        <TableRow>
                                            <TableCell><Typography variant="body1" sx={{color: '#666'}}>GPS longitude</Typography></TableCell>
                                            <TableCell><Typography variant="body1" sx={{color: '#666'}}>{image.metadata.gps_longitude ?? 'na'}</Typography></TableCell>
                                        </TableRow>
                                        <TableRow>
                                            <TableCell><Typography variant="body1" sx={{color: '#666'}}>GPS longitude ref</Typography></TableCell>
                                            <TableCell><Typography variant="body1" sx={{color: '#666'}}>{image.metadata.gps_longitude_ref ?? 'na'}</Typography></TableCell>
                                        </TableRow>
                                        <TableRow>
                                            <TableCell><Typography variant="body1" sx={{color: '#666'}}>GPS latitude</Typography></TableCell>
                                            <TableCell><Typography variant="body1" sx={{color: '#666'}}>{image.metadata.gps_latitude ?? 'na'}</Typography></TableCell>
                                        </TableRow>
                                        <TableRow>
                                            <TableCell><Typography variant="body1" sx={{color: '#666'}}>GPS latitude ref</Typography></TableCell>
                                            <TableCell><Typography variant="body1" sx={{color: '#666'}}>{image.metadata.gps_latitude_ref ?? 'na'}</Typography></TableCell>
                                        </TableRow>
                                    </TableBody>
                                </Table>
                                :
                                <Typography style={{color: '#666', marginRight: '13px'}}>No more information</Typography>
                            }
                        </TableContainer>
                    </Box>
                </Stack>
            </Stack>
      </Stack>
      : null
    );
};

export default ImageDetail;