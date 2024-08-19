import sys
sys.path.append('../mirs')

import os
import math
import json
from typing import List, Dict
from mirs.conf import config
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy import stats

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
    flickr8k_augmented_metadata = load_augmented_flickr8k_metadata()

    
    original_vocab = set()
    original_caption_lengths = []
    for image in flickr8k_metadata['images']:
        for sentence in image['sentences']:
            original_caption_lengths.append(len(sentence['tokens']))
            for token in sentence['tokens']:
                original_vocab.add(token)
    print('Original vocab size:', len(original_vocab))

    rephrased_vocab = set()
    rephrased_caption_lengths = []
    print(len(flickr8k_augmented_metadata))
    n_captions = 0
    for augmented in flickr8k_augmented_metadata:
        if len(augmented['captions']) != 5:
            print(augmented['image'], '->', len(augmented['captions']))
        for caption in augmented['captions']:
            n_captions += 1
            tokens = re.sub(r'[^\w\s]', '', caption['rephrased']).split()
            rephrased_caption_lengths.append(len(tokens))
            for token in tokens:
                rephrased_vocab.add(token)

    print('Rephrased vocab size:', len(rephrased_vocab))
    print('n_captions:', n_captions)
    n_new_vocab = 0
    for word in rephrased_vocab:
        if word not in original_vocab:
            n_new_vocab += 1

    print('New vocabs:', n_new_vocab)
    # x = np.arange(0, len(rephrased_caption_lengths))
    print(stats.describe(rephrased_caption_lengths))
    print(stats.describe(original_caption_lengths))

    plt.boxplot([original_caption_lengths, rephrased_caption_lengths], vert=True, patch_artist=True, labels=['Original', 'Rephrased'])    
    plt.ylabel('# of tokens per caption')
    # plt.title('Comparison of token counts in original vs rephrased captions')
    plt.show()



