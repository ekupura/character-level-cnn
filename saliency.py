from keras.models import load_model, Model
from keras import backend as K
from vis.visualization import visualize_saliency
from vis.utils import utils
from keras import activations
from PIL import Image
import numpy as np

def saliency(conf, sample, label):
    path = conf["paths"]["model_path"]
    original_model = load_model(path)

    layer_idx = utils.find_layer_idx(original_model, 'preds')
    original_model.layers[layer_idx].activation = activations.linear
    model = utils.apply_modifications(original_model)

    K.set_learning_phase(0)
    input_tensors = [original_model.input]
    gradients = original_model.optimizer.get_gradients(original_model.output[0][int(label[1])], original_model.layers[2].input)
    compute_gradients = K.function(inputs=input_tensors, outputs=gradients)
    matrix = compute_gradients([sample.reshape(1, 1, 150)])[0][0]

    original_model.summary()
    print(sample)
    return matrix.reshape(150, 32)


def min_max(x, axis=None):
    _max = np.max(x)
    _min = np.min(x)
    return (x - _min) / (_max - _min)


def saliency_to_png(saliency):
    saliency[saliency < 0.0] = 0.0
    mat = min_max(saliency)
    """
    img = Image.fromarray(min_max(array), 'L')
    img.save('saliency2.png')
    """
    print(mat)
    m = np.mean(mat, axis=1)
    print(list(m))


