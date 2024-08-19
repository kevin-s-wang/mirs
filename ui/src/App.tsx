import React, { useRef, useState } from 'react';
import Typography from '@mui/material/Typography';
import Box from '@mui/material/Box';
import SearchIcon from '@mui/icons-material/Search';
import { Divider, Fade, IconButton, InputBase, Paper, Popper, Stack, Tooltip,  Badge, Dialog, DialogTitle, DialogContent, DialogContentText, DialogActions, Button, Modal, LinearProgress } from '@mui/material';
import CameraAltIcon from '@mui/icons-material/CameraAlt';
import CloseIcon from '@mui/icons-material/Close';
import CollectionsIcon from '@mui/icons-material/Collections';

import HelpIcon from '@mui/icons-material/Help';
import ImagePreview from './ImagePreview';
import { Image } from './data';
import Gallery from './Gallery';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';

const mainStyles = { 
    background: '#efefef', 
    height: '100%', 
    width: '100%', 
    overflow: 'hidden'
};

const navbarStyles = { 
  height: '80px', 
  lineHeight: '80px',
  width: '100%', 
  padding: '0 13px',
  backgroundColor: '#fff',
  zIndex: 100,
  boxShadow:'rgba(33, 35, 38, 0.1) 0px 10px 10px -10px',
};

const brandStyles = {
  width: 260,
  height: '80px',
  padding: '10px 0',
  userSelect: 'none',
};

const logoStyles = {
  backgroundColor: 'primary.main',
  width: 120,
  height: '60px',
  lineHeight: '60px',
  color: '#fff',
  fontWeight: 'bold',
  fontSize: '2rem',
  textAlign: 'center',
};

interface PageInfo {
  limit: number;
  offset: number;
}

const IMAGES_API_BASE_URL = 'http://127.0.0.1:8889/api/v1/images';

const App = () => {

  const [imageFiles, setImageFiles] = useState<File[]>([]);
  const [images, setImages] = useState<Image[]>([]);
  const [helpDialogOpen, setHelpDialogOpen] = useState<boolean>(false);
  const [uploadDialogOpen, setUploadDialogOpen] = useState<boolean>(false);
  const pageInfo = useRef<PageInfo>({limit: 12, offset: 0});
  const [textPrompt, setTextPrompt] = useState<string>('');
  const [anchorEl, setAnchorEl] = React.useState<null | HTMLElement>(null);
  const [erorrMessage, setErrorMessage] = useState<string|null>(null);
  const [displayProgress, setDisplayProgress] = useState<boolean>(false);

  const handleClick = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(anchorEl ? null : event.currentTarget);
  };

  const inputRef = useRef<HTMLInputElement>(null);
  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    const files = Array.from(event.dataTransfer.files);
    setImageFiles([...imageFiles, ...files]);
};

const handleFileInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || []);    
    setImageFiles([...imageFiles, ...files]);
};

  const handleRemoveImage = (index: number) => {
      const newImages = [...imageFiles];
      newImages.splice(index, 1);
      setImageFiles(newImages);
      if(inputRef.current) {
        const data = new DataTransfer();
        newImages.forEach((image) => {
          data.items.add(image);
        });
        inputRef.current.files = data.files;
      }
  };
  const open = Boolean(anchorEl);
  const id = open ? 'search-with-images' : undefined;

  const search = async (e?: any, reset = false) => {
    e?.preventDefault();
    
    if(reset) {
      pageInfo.current.offset = 0;
      pageInfo.current.limit = 20;
    }

    const searchImages = async (files: File[]) => {
      
      const prompt = textPrompt.trim();
      if (files.length === 0 && !prompt) {
        console.error('No images or text prompt provided!');
        return;
      }
      const formData = new FormData();
      files.forEach((file) => {
        formData.append('images', file);
      });

      if(prompt) {
        formData.append('prompt', prompt);
      }
      
      setDisplayProgress(true);

      const response = await fetch(`${IMAGES_API_BASE_URL}/search?limit=${pageInfo.current.limit}&offset=${pageInfo.current.offset}`, {
        method: 'POST',
        body: formData,
      });
      if(response.ok) {
        const result = await response.json();
        if(reset) {
          setImages([...result.data]);
        } else {
          setImages([...images, ...result.data]);
        }
        pageInfo.current.offset += result.data.length;
      } 
      else if (response.status === 400) {        
        const err = await response.json();
        setErrorMessage(err.message);
        setImages([]);
      } else {
        setErrorMessage(response.statusText);
        setImages([]);
      }
      setDisplayProgress(false);
      
    };
    searchImages(imageFiles).catch(err => {
      setImages([]);
      setErrorMessage(err.message);
    });
  };

  const onClearInput = () => {
    pageInfo.current.offset = 0;
    pageInfo.current.limit = 20;
    setTextPrompt('');
    setImages([]);
    setDisplayProgress(false);
  };

  return (
    <Stack className='main' direction='column' sx={mainStyles}>
      <Box sx={{width: '100%', backgroundColor: '#fff'}} >
          <LinearProgress sx={{visibility: displayProgress ? 'visible' : 'hidden'}} />
      </Box>
      <Stack className='navbar' direction='row' sx={navbarStyles}>
        <Box className='brand' sx={brandStyles} >
            <Box className='logo' sx={logoStyles}>
                <div className='text' title='MIRS: Muiltimal Image Retrieval System'>MIRS</div>
            </Box>
        </Box>
        <Box className='searchbar' sx={{padding: '15px 0', height: '80px', width: '100%'}} >
        <Paper component="form"
              onSubmit={e => search(e, true)}
              sx={{ p: '2px 4px', 
              display: 'flex', 
              alignItems: 'center', 
              width: '100%',
              boxShadow: 'none', 
              backgroundColor: '#efefef',
              height: '50px',
              borderRadius: '25px',
               }}>
          {/* <IconButton sx={{ p: '10px' }} aria-label="menu">
            <MenuIcon />
          </IconButton> */}

          <InputBase
            sx={{ ml: 1, flex: 1, marginLeft: '25px',
          }}
            placeholder="Your prompt here!"
            value={textPrompt}
            onChange={(e: any) => setTextPrompt(e.target.value || '')}
            inputProps={{ 'aria-label': 'multimodal image retrieval' }} />

              <IconButton type='button' 
                  sx={{ p: '10px', visibility: textPrompt.trim() ? 'visible' : 'hidden' }} 
                  aria-label="Clear"
                  onClick={onClearInput}>
                <CloseIcon />
              </IconButton>
          <Divider sx={{ height: 28, m: 0.5 }} orientation="vertical" />

          <Tooltip arrow title="Search with image(s)">
                <IconButton aria-describedby={id} type='button' sx={{ p: '10px' }} aria-label="Visual Prompt" onClick={handleClick}>
                  <Badge badgeContent={imageFiles.length} color="primary">
                    <CameraAltIcon />
                  </Badge>
              </IconButton>  
          </Tooltip>
          <Popper id={id} open={open} anchorEl={anchorEl} placement='bottom' transition sx={{zIndex: 200}}
            // modifiers={[{
            //   name: 'arrow',
            //   enabled: true,
            //   options: {
            //     element: arrowRef,
            //   },
            // },]}
          >
        {({ TransitionProps }) => (
          <Fade {...TransitionProps} timeout={350}>
            <Paper sx={{p: 2, boxShadow: 'rgba(17, 17, 26, 0.1) 0px 8px 24px, rgba(17, 17, 26, 0.1) 0px 16px 56px, rgba(17, 17, 26, 0.1) 0px 24px 80px'}}>
              <div style={{padding: '13px', width: 500, height: 240, border: '1px dashed #ccc', textAlign: 'center'}}
                  onDrop={handleDrop}
                  onDragOver={e => e.preventDefault()}>
                  <CollectionsIcon color='disabled' sx={{fontSize: 50}}/>
                  <Typography sx={{ p: 2, textAlign: 'left'}}>Drag and drop images here or&nbsp;
                  <span style={{color: 'blue', cursor: 'pointer'}} role="button" onClick={e => {
                          e.preventDefault();
                          inputRef.current?.showPicker();
                      }}>click to select images<input ref={inputRef} type='file' hidden accept='image/*' multiple onChange={handleFileInputChange}/></span>
                  </Typography>
              </div>
              <ImagePreview images={imageFiles} onRemove={handleRemoveImage} />
            </Paper>
          </Fade>
        )}
      </Popper>

          <IconButton type="button" color='primary' sx={{ p: '10px' }} aria-label="Search" onClick={e => search(e, true)}>
            <SearchIcon />
          </IconButton>
        </Paper>
        </Box>
        <Box className='tools' sx={{marginLeft: 'auto', p: '13px', width: 100, display: 'flex', direction: 'row-reverse'}}>
          
          <IconButton type='button' sx={{ height: 'max-content', alignSelf: 'center' }} aria-label="Help" onClick={() => setUploadDialogOpen(true)}>
              <CloudUploadIcon /> 
          </IconButton>
          <IconButton type='button' sx={{ height: 'max-content', alignSelf: 'center' }} aria-label="Help" onClick={() => setHelpDialogOpen(true)}>
              <HelpIcon />
          </IconButton>
          
          <Dialog 
              open={helpDialogOpen} 
              onClose={() => setHelpDialogOpen(false)}
              aria-labelledby="help-dialog-title"
              aria-describedby="help-dialog-description">
              <DialogTitle id="help-dialog-title">Multimodal Image Retrieval System</DialogTitle>
              <DialogContent>
                <DialogContentText id="help-dialog-description">
                  This system is fully designed and developed by Kevin Wang for his master thesis at UCAS. Any unauthorized use of this system is strictly prohibited.
                </DialogContentText>
              </DialogContent>
              <DialogActions>
                <Button onClick={() => setHelpDialogOpen(false)}>Close</Button>
              </DialogActions>
            </Dialog>

            <Dialog 
              open={!!erorrMessage} 
              onClose={() => setErrorMessage(null)}
              aria-labelledby="error-dialog-title"
              aria-describedby="error-dialog-description">
              <DialogTitle id="error-dialog-title">ERROR</DialogTitle>
              <DialogContent>
                <DialogContentText id="error-dialog-description">
                {erorrMessage}
                </DialogContentText>
              </DialogContent>
              <DialogActions>
                <Button onClick={() => setErrorMessage(null)}>Close</Button>
              </DialogActions>
            </Dialog>

        </Box>

      </Stack>
      <Gallery images={images} onLoadMore={search}/>
    </Stack>
  );
};

export default App;