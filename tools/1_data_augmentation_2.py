import sys
sys.path.append('../mirs')
import os
import json
import re
from tqdm import tqdm
from mirs.conf.config import config
from PIL import Image, ImageFilter
from typing import Dict, List
import numpy as np
from pprint import pprint
from mirs.ai.models import clip_model
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

max_error = 0.001
max_error_proportion = 0.1

def load_flickr8k_metadata(datasets_dir: str) -> Dict:
    flickr8k_metadata_filepath = os.path.join(datasets_dir, 'dataset_flickr8k.json')

    with open(flickr8k_metadata_filepath, 'r') as f:
        return json.load(f)
    
def load_flickr8k_augmentation_metadata(datasets_dir: str, augment_method: str) -> List:
    metadata_filepath = os.path.join(datasets_dir, 'flickr8k-augmentation', f'flickr8k_{augment_method}.json')
    print('Flickr8k augmentation metadata:', metadata_filepath)
    if os.path.exists(metadata_filepath):
        try:
            with open(metadata_filepath, 'r') as stream:
                return json.load(stream)
        except Exception as e:
            print(e)
    return []
    
    
def gaussian_noise(size, seed=1, mean=0, sigma=5):
    gen = np.random.default_rng(seed)
    return gen.normal(mean, sigma, size)

augment_methods = ['gaussian_noise', 'gaussian_blur']

if __name__ == '__main__':
    augment_method = 'gaussian_noise'
    if augment_method not in augment_methods:
        raise ValueError('Supported augmentation methods are:', ','.join(augment_methods))
    
    datasets_dir = os.path.join(config.root_dir, '.dev/datasets')
    flickr8k_augmentation_metadata = load_flickr8k_augmentation_metadata(datasets_dir, augment_method)
    flickr8k_metadata = load_flickr8k_metadata(datasets_dir)

    flickr8k_images = flickr8k_metadata['images'][7843:7844]
    
    augmented_images = []
    for image in tqdm(flickr8k_images):
        image_filepath = os.path.join(datasets_dir, 'Flicker8k_Dataset', image['filename'])

        im = Image.open(image_filepath)

        captions = [re.sub('\s+\.?$', '.', sentence['raw']) for sentence in image['sentences']]
        
        new_image = {
            'filename': image['filename'],
            'noisy_images': []
        }
        np_im = np.array(im)
        for step in range(20):
            new_filename = f"{step+1} - {image['filename']}"
            np_im = np_im + gaussian_noise(np_im.shape)

            np_im = np.clip(np_im, 0, 255).astype(np.uint8)

            noisy_im = Image.fromarray(np_im)
            noisy_im.save(new_filename)

            noisy_image = {
                'filename': new_filename,
                'similarities': []
            }

            for caption in captions:
                original_similarity = clip_model.get_image_text_similarity(im, caption).item()
                augmented_similarity = clip_model.get_image_text_similarity(noisy_im, caption).item()
                improved_similarity =  augmented_similarity - original_similarity

                noisy_image['similarities'].append({
                    'caption': caption,
                    'original_similarity': original_similarity,
                    'augmented_similarity': augmented_similarity,
                    'improved_similarity': improved_similarity,
                })
            new_image['noisy_images'].append(noisy_image)
                # print('-' * 120)
                # print('Original Similarity:', original_similarity, '\nAugmented Similarity 2:', augmented_similarity)
                # print('Error:', error, 'Exceed allowed error:', error > max_error)
        augmented_images.append(new_image)

    with open('test.json', 'w') as stream:
        json.dump(augmented_images, stream)

    first_image = augmented_images[0]
    original_similarities_1 = []
    original_similarities_2 = []
    original_similarities_3 = []
    original_similarities_4 = []
    original_similarities_5 = []

    augmented_similarities_1 = []
    augmented_similarities_2 = []
    augmented_similarities_3 = []
    augmented_similarities_4 = []
    augmented_similarities_5 = []
    print('Len: ', len(first_image['noisy_images']))
    for noisy_image in first_image['noisy_images']:
        similarities = noisy_image['similarities']
        original_similarities_1.append(similarities[0]['original_similarity'])
        augmented_similarities_1.append(similarities[0]['augmented_similarity'])

        original_similarities_2.append(similarities[1]['original_similarity'])
        augmented_similarities_2.append(similarities[1]['augmented_similarity'])

        original_similarities_3.append(similarities[2]['original_similarity'])
        augmented_similarities_3.append(similarities[2]['augmented_similarity'])

        original_similarities_4.append(similarities[3]['original_similarity'])
        augmented_similarities_4.append(similarities[3]['augmented_similarity'])

        original_similarities_5.append(similarities[4]['original_similarity'])
        augmented_similarities_5.append(similarities[4]['augmented_similarity'])


    x = np.arange(1, 21)

    plt.title(first_image['filename'])
    plt.xticks(x)
    plt.xlabel('t')
    plt.ylabel('Cosine Similarity')
    plt.plot(x, original_similarities_1, 'b:') 
    plt.plot(x, original_similarities_2, 'g:')
    plt.plot(x, original_similarities_3, 'r:')
    plt.plot(x, original_similarities_4, 'c:') 
    plt.plot(x, original_similarities_5, 'm:')   

    plt.plot(x, augmented_similarities_1, 'b') 
    plt.plot(x, augmented_similarities_2, 'g') 
    plt.plot(x, augmented_similarities_3, 'r') 
    plt.plot(x, augmented_similarities_4, 'c') 
    plt.plot(x, augmented_similarities_5, 'm')
    lines = [
        Line2D([0], [0], linestyle=':'),
        Line2D([0], [0], linestyle='-'),
    ]
    plt.legend(lines, ['baseline', 'w/ guassian noise'])

    plt.show()
    # pprint(augmented_images)
        #noisy_im = im.copy()


        # for radius in range(3):
        #     if radius != 0:
        #         noisy_im = im.filter(ImageFilter.GaussianBlur(radius))

        #     print(radius+1, '+' * 120)
        #     images_similarity = clip_model.get_images_similarity(im, noisy_im).item()

        #     print('Images similarity:', images_similarity)
            
        #     for caption in captions:
        #         original_similarity = clip_model.get_image_text_similarity(im, caption).item()
        #         augmented_similarity = clip_model.get_image_text_similarity(noisy_im, caption).item()
        #         error = original_similarity - augmented_similarity

        #         print('-' * 120)
        #         print('Original Similarity:', original_similarity, '\nAugmented Similarity 2:', augmented_similarity)
        #         print('Error:', error, 'Exceed allowed error:', error > max_error)
        # im.show()
        # noisy_im.show()



