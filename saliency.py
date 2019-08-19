from keras.models import load_model, Model
from keras import backend as K
from vis.utils import utils
from keras import activations
import numpy as np
from layer import CharacterEmbeddingLayer
from keras.utils import np_utils
from matplotlib import pyplot as plt


def calculate_saliency(conf, sample, label):
    path = conf["paths"]["model_path"]
    layer_dict = {'CharacterEmbeddingLayer': CharacterEmbeddingLayer}
    original_model = load_model(path, custom_objects=layer_dict)
    sample = np_utils.to_categorical(sample)

    layer_idx = utils.find_layer_idx(original_model, 'preds')
    original_model.layers[layer_idx].activation = activations.linear

    K.set_learning_phase(0)
    input_tensors = [original_model.input]
    gradients = original_model.optimizer.get_gradients(original_model.output[0][int(label[1])], original_model.layers[1].input)
    compute_gradients = K.function(inputs=input_tensors, outputs=gradients)
    matrix = compute_gradients([sample.reshape(1, 1, 150, 67)])[0][0]

    return np.mean((matrix.reshape(150, 67)), axis=1)
    #return zscore(matrix.reshape(150, 67))
    #return matrix.reshape(150, 67)


def generate_heatmap(conf, saliency, text, path=None):
    saliency[saliency < 0.0] = 0.0
    saliency = zscore(saliency.reshape(15, 10))
    #saliency = np.array([saliency for n in saliency])

    plt.figure(figsize=(7, 5))
    plt.imshow(saliency, interpolation='nearest', cmap='jet')

    ys, xs = np.meshgrid(range(saliency.shape[0]), range(saliency.shape[1]), indexing='ij')
    for (x, y, c) in zip(xs.flatten(), ys.flatten(), text):
        plt.text(x, y, c, horizontalalignment='center', verticalalignment='center', )

    plt.colorbar()
    if path is None:
        path = conf['paths']['saliency_path']
    plt.savefig(path)


# min max normalization
def min_max(x, axis=None):
    _max = np.max(x)
    _min = np.min(x)
    return (x - _min) / (_max - _min)

# z-score normalization
def zscore(x, axis = None):
    xmean = x.mean(axis=axis, keepdims=True)
    xstd  = np.std(x, axis=axis, keepdims=True)
    zscore = (x-xmean)/xstd
    return zscore

