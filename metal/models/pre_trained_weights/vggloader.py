import numpy as np
import pathlib
from PIL import Image
import requests


LABELS_URL = 'https://s3.amazonaws.com/outcome-blog/imagenet/labels.json'
response = requests.get(LABELS_URL)
labels = {int(key): value for key, value in response.json().items()}
load_vgg_weights_labels = np.load(str(pathlib.Path(__file__).parent.absolute())+"/"+'VGGBASE.npz'), labels

def prepare(image, size=(224, 224)):
    if isinstance(image, np.ndarray):
        if image.ndim == 3:
            if image.shape[0] == 1:
                image = image[0, :, :]
            elif image.shape[0] == 3:
                image = image.transpose((1, 2, 0))
        image = Image.fromarray(image.astype(np.uint8))
    image = image.convert('RGB')
    if size:
        image = image.resize(size)
    image = np.asarray(image, dtype="float32")
    image = image[:, :, ::-1]
    image -= np.array(
        [103.939, 116.779, 123.68], dtype="float32")
    image = image.transpose((2, 0, 1))
    return image

vggweights = load_vgg_weights
