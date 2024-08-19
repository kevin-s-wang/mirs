import os
import sys
import shutil

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from mirs.conf import config
from mirs.utils import load_flickr8k_metadata

if __name__ == '__main__':
    datasets_dir = os.path.join(config.dev_dir, 'datasets')
    flickr8k_dataset_dir = os.path.join(datasets_dir, 'flickr8k')

    flickr8k_metadata = load_flickr8k_metadata(datasets_dir)

    flickr8k_images = flickr8k_metadata['images']
    testdata_dir = os.path.join(config.dev_dir, 'testdata')
    os.makedirs(testdata_dir, exist_ok=True)

    for image in flickr8k_images:
        for topic in ('climbing', 'hiking', 'paddling', 'biking'):
            os.makedirs(os.path.join(testdata_dir, topic), exist_ok=True)
            if any(topic in sentence['raw'] for sentence in image['sentences']):
                image_filename = image['filename']
                image_filepath = os.path.join(datasets_dir, 'Flicker8k_Dataset', image_filename)
                shutil.copyfile(image_filepath, os.path.join(testdata_dir, topic, image_filename))
                break
