import os
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
from matplotlib import animation
from tqdm import tqdm


class GradientSaliency:
    def __init__(self, model, label, limit_characters):
        # Define the function to compute the gradient
        self.limit_characters = limit_characters
        out_layer_idx = utils.find_layer_idx(model, 'final')
        in_layer_idx = utils.find_layer_idx(model, 'embedding')
        output_index = int(label[0])
        input_tensors = [model.input]
        gradients = model.optimizer.get_gradients(model.layers[out_layer_idx].output[0][output_index],
                                                  model.layers[in_layer_idx].input)
        self.compute_gradients = K.function(inputs=input_tensors, outputs=gradients)

    def get_mask(self, sample):
        # Execute the function to compute the gradient
        # sample = np_utils.to_categorical(sample, 67)
        # x_value = sample.reshape(1, 150, 67, 1)
        x_value = sample.reshape(1, self.limit_characters, 1)
        gradients = self.compute_gradients([x_value])[0][0]

        return gradients

    def delete(self):
        del(self.compute_gradients)


def calculate_saliency(conf, sample, label, model=None):
    if model is None:
        path = conf["paths"]["model_path"]
        layer_dict = {'CharacterEmbeddingLayer': CharacterEmbeddingLayer}
        model = load_model(path, custom_objects=layer_dict)

    saliency = GradientSaliency(model, label, conf['preprocessing_parameters']['limit_characters'])
    matrix = saliency.get_mask(sample)
    saliency.delete()
    print("zoo")
    #matrix = np.zeros((150,67))
    del(saliency)

    return np.mean((matrix.reshape(150, 67)), axis=1)

def calculate_saliency_multi(conf, samples, labels, model):
    limit_characters = conf['preprocessing_parameters']['limit_characters']
    number_of_characters = conf['preprocessing_parameters']['number_of_characters']
    saliencies = {}
    for label in np.unique(labels, axis=0):
        idx = np.where(label == 1.0)[0][0]
        saliencies[idx] = GradientSaliency(model, list(label), limit_characters)

    results = []
    for sample, label in tqdm(zip(samples, labels)):
        idx = np.where(label == 1.0)[0][0]
        matrix = saliencies[idx].get_mask(sample)
        results.append(np.mean((matrix.reshape(limit_characters, number_of_characters)), axis=1))

    return results

def calculate_saliency_with_vis(conf, sample, label, model=None):
    if model is None:
        path = conf["paths"]["model_path"]
        layer_dict = {'CharacterEmbeddingLayer': CharacterEmbeddingLayer}
        model = load_model(path, custom_objects=layer_dict)

    layer_idx = utils.find_layer_idx(model, 'preds')
    # class_idx = [int(label[1])]
    class_idx = [int(label[1])]

    sample = np_utils.to_categorical(sample, 67)
    x_value = sample.reshape(150, 67, 1)
    grads = visualize_saliency(model, layer_idx, filter_indices=class_idx, seed_input=x_value)

    return np.mean((grads.reshape(150, 67)), axis=1)


def generate_heatmap(conf, saliency, id_list, epoch=None, path=None, z_norm=False):
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
    _max = np.max(saliency) * 1.414
    plt.imshow(saliency, interpolation='nearest', cmap=orig_cmap, vmax=_max, vmin=0.0)

    ys, xs = np.meshgrid(range(saliency.shape[0]), range(saliency.shape[1]), indexing='ij')
    for (x, y, c) in zip(xs.flatten(), ys.flatten(), text):
        plt.text(x, y, c, horizontalalignment='center', verticalalignment='center', )

    if epoch is None:
        plt.title(conf["experiment_name"])
    else:
        plt.title("{},epoch={}".format(conf["experiment_name"], epoch))


    #plt.colorbar()

    # save figure
    if path is None:
        path = conf['paths']['saliency_path']
    print(path)
    plt.savefig(path)

    f.clear()
    plt.close(f)


# generate one gif image
def generate_animation_heatmap(conf, case, saliency_list, id_list, label='', sm=False):
    # preprocess
    if sm:
        saliency_list = [softmax(saliency).reshape(16, 10) for saliency in saliency_list]
    else:
        saliency_list = [saliency.reshape(16, 10) for saliency in saliency_list]
    text = id_list_to_characters(id_list)
    case = case + 1

    f = plt.figure(figsize=(7, 5))

    # configure figure elements
    orig_cmap = matplotlib.cm.coolwarm
    _max = np.max(saliency_list) * 1.000

    def update(i):
        if i != 0:
            plt.cla()
        saliency = saliency_list[i]
        saliency[9][9] = 0.0
        epoch = i + 1
        ys, xs = np.meshgrid(range(saliency.shape[0]), range(saliency.shape[1]), indexing='ij')
        for (x, y, c) in zip(xs.flatten(), ys.flatten(), text):
            plt.text(x, y, c, horizontalalignment='center', verticalalignment='center', )
        im = plt.imshow(saliency, interpolation='nearest', cmap=orig_cmap, vmax=_max, vmin=-_max)
        plt.title("name={},case={},label={},epoch={}".format(conf["experiment_name"], case, int(label[1]), epoch))
        return [im]

    gif_dir = conf['paths']['saliency_dir_path'] + 'gif/'
    if not os.path.exists(gif_dir):
        os.mkdir(gif_dir)
    ani = animation.FuncAnimation(f, update, frames=len(saliency_list), interval=500, blit=True)
    ani.save('{}{}.gif'.format(gif_dir, str(case).zfill(3)), writer="imagemagick")

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

def softmax(x):
    f = np.exp(x)/np.sum(np.exp(x), keepdims = True)
    return f
