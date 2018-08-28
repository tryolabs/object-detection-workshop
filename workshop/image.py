import numpy as np
import os

from PIL import Image


def open_all_images(path):
    """Opens all images located in `path`.

    Returns:
        Dictionary that maps the base name (with no extension) to the
        `numpy.ndarray` corresponding to the image.
    """
    images = {}
    for filename in os.listdir(path):
        curr_path = os.path.join(path, filename)
        if not os.path.isfile(curr_path):
            continue

        name, _ = os.path.splitext(os.path.basename(curr_path))
        images[name] = open_image(curr_path)

    return images


def open_image(path):
    path = os.path.expanduser(path)
    raw_image = Image.open(path)
    image = np.expand_dims(raw_image.convert('RGB'), axis=0)
    return image


def to_image(image_array):
    return Image.fromarray(np.squeeze(image_array, axis=0))
