from keras.models import load_model, Model
from keras import backend as K
from vis.utils import utils
from vis.visualization import visualize_saliency
from keras import activations
import numpy as np
from keras.utils import np_utils
from matplotlib import pyplot as plt
import matplotlib
from layer import CharacterEmbeddingLayer
from preprocess import id_list_to_characters


class GradientSaliency:
    def __init__(self, model, label):
        # Define the function to compute the gradient
        layer_idx = utils.find_layer_idx(model, 'final')
        output_index = int(label[1])
        print(output_index)
        input_tensors = [model.input]
        gradients = model.optimizer.get_gradients(model.output[0][output_index], model.input)
        self.compute_gradients = K.function(inputs=input_tensors, outputs=gradients)

    def get_mask(self, sample):
        # Execute the function to compute the gradient
        sample = np_utils.to_categorical(sample)
        x_value = sample.reshape(1, 1, 150, 67)
        gradients = self.compute_gradients([x_value])[0][0]

        return gradients


def calculate_saliency(conf, sample, label):
    path = conf["paths"]["model_path"]
    layer_dict = {'CharacterEmbeddingLayer': CharacterEmbeddingLayer}
    model = load_model(path, custom_objects=layer_dict)

    saliency = GradientSaliency(model, label)
    matrix = saliency.get_mask(sample)

    return np.mean((matrix.reshape(150, 67)), axis=1)

def calculate_saliency_with_vis(conf, sample, label):
    path = conf["paths"]["model_path"]
    layer_dict = {'CharacterEmbeddingLayer': CharacterEmbeddingLayer}
    model = load_model(path, custom_objects=layer_dict)

    layer_idx = utils.find_layer_idx(model, 'preds')
    class_idx = [int(label[1])]
    sample = np_utils.to_categorical(sample)
    x_value = sample.reshape(150, 67, 1)
    grads = visualize_saliency(model, layer_idx, filter_indices=class_idx, seed_input=x_value)

    return np.mean((grads.reshape(150, 67)), axis=1)

def generate_heatmap(conf, saliency, id_list, label, keyword='None', path=None, z_norm=False):
    # preprocess
    if z_norm:
        saliency[saliency < 0.0] = 0.0
        saliency = zscore(saliency.reshape(15, 10))
    else:
        saliency = (saliency.reshape(15, 10))
    text = id_list_to_characters(id_list)

    f = plt.figure(figsize=(7, 5))

    # configure figure elements
    orig_cmap = matplotlib.cm.Oranges
    shifted_cmap = shiftedColorMap(orig_cmap, start=0.15, midpoint=0.5, stop=0.85, name='shifted')
    # plt.imshow(saliency, interpolation='nearest', cmap=shifted_cmap)
    _max = np.max(saliency) * 1.414
    plt.imshow(saliency, interpolation='nearest', cmap=orig_cmap, vmax=_max, vmin=0.0)

    ys, xs = np.meshgrid(range(saliency.shape[0]), range(saliency.shape[1]), indexing='ij')
    for (x, y, c) in zip(xs.flatten(), ys.flatten(), text):
        plt.text(x, y, c, horizontalalignment='center', verticalalignment='center', )

    #label = 'positive' if label[0] == 0 else 'negative'
    #plt.title('keyword = {}, label = {}'.format(keyword, label))
    plt.title(conf["experiment_name"])

    #plt.colorbar()

    # save figure
    if path is None:
        path = conf['paths']['saliency_path']
    print(path)
    plt.savefig(path)

    f.clear()
    plt.close(f)


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


def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap