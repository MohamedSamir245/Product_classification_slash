### We create a bunch of helpful functions throughout the course.
### Storing them here so they're easily accessible.

import tensorflow as tf

import pandas as pd

def get_paths_and_labels(root_path):
    image_paths = root_path.glob('*/*')
    image_paths = [str(path) for path in image_paths]
    labels = [path.split('/')[-2] for path in image_paths]
    return pd.DataFrame({'image_path': image_paths, 'label': labels})


