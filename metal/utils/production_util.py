import resource
import sys
import urllib.request
import PIL
from PIL import Image
from metal.utils.data_manipulation import normalize
import numpy as np
import dill
from autograd.parameter import Parameter


print('resource limit ',resource.getrlimit(resource.RLIMIT_STACK))
print('recursion limit ',sys.getrecursionlimit())


def save_model(filename,model, max_rec=100000):
    max_rec = max_rec
    sys.setrecursionlimit(max_rec)
    with open(filename+'.pkl', 'wb') as file:
        dill.dump(model.layers, file)

def load_model(filename):
    with open(filename+'.pkl', 'rb') as file:
        layers = dill.load(file)
    return layers

def fetch_img(url,shape):
    I = Image.open(urllib.request.urlopen(url)).convert('L')
    I = np.asarray(I.resize((shape[-1],shape[-2]), PIL.Image.LANCZOS)).reshape(*shape)
    I = np.invert(I)
    I = normalize(I)
    return I

def _forward_pass(X, training=True, model_name=None):
    """ Calculate the output of the NN """
    layers_ = load_model(model_name)
    layer_output = Parameter(X,False)
    for layer in layers_:
        layer_output = layer.forward_pass(layer_output, training)
    return layer_output
