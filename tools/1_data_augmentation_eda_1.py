import sys
sys.path.append('../mirs')

import os
import math
import json
from typing import List, Dict
from mirs.conf import config
import numpy as np
# from pprint import pprint
import matplotlib.pyplot as plt

def load_augmented_flickr8k_metadata() -> List:
    metadata_filepath = os.path.join(config.datasets_dir, 'dataset_flickr8k_augmented_new.json')
    if os.path.exists(metadata_filepath):
        try:
            with open(metadata_filepath, 'r') as stream:
                return json.load(stream)
        except Exception as e:
            print(e)
    return []

def load_flickr8k_metadata() -> Dict:
    metadata_filepath = os.path.join(config.datasets_dir, 'dataset_flickr8k.json')
    with open(metadata_filepath, 'r') as stream:
        return json.load(stream)


if __name__ == '__main__':
    flickr8k_metadata = load_flickr8k_metadata()
    flickr8k_images = flickr8k_metadata['images']

    augmented_metadata = load_augmented_flickr8k_metadata()
    
    print('# of records:', len(augmented_metadata))

    image_original_similarities = []
    image_rephrased_similarities = []
    original_rephrased_similarities = []

    n_betters = 0
    n_worses = 0
    n_equivs = 0
    distances = []
    for augmented in augmented_metadata:
        for caption in augmented['captions']:
            s = caption['image_rephrased_similarity'] - caption['image_original_similarity']
            distances.append(s)
            if  s > 0 :
                n_betters += 1
            elif s <= 0 and s >= -0.05:
                n_equivs += 1
            else:
                n_worses += 1
            
            image_original_similarities.append(caption['image_original_similarity'])
            image_rephrased_similarities.append(caption['image_rephrased_similarity'])
            original_rephrased_similarities.append(caption['original_rephrased_similarity'])

    # x = np.arange(0, len(image_original_similarities))
    # plt.plot(x, image_original_similarities)
    # plt.plot(x, image_rephrased_similarities)
    # plt.plot(x, original_rephrased_similarities)
    # plt.boxplot([
    #         image_original_similarities, 
    #         image_rephrased_similarities, 
    #         original_rephrased_similarities
    #     ],
    #     vert=True,
    #     patch_artist=True,
    #     labels=['IO', 'IR', 'OR'])

    # plt.boxplot(distances, vert=True, patch_artist=True)
    print('n_betters:', n_betters, 'percentage:', (n_betters / 40000)*100)
    print('n_equivs:', n_equivs, 'percentage:', (n_equivs / 40000)*100 )
    print('n_worses:', n_worses, 'percentage:', (n_worses / 40000)*100)
    plt.pie([(n_betters / 40000)*100, (n_equivs / 40000)*100, (n_worses / 40000)*100], 
            autopct='%1.1f%%',
            labels=['Better', 'Equivalent', 'Worse'] )
    plt.title('Caption Augmentation Report (First epoch, 40k captions)')
    plt.show()