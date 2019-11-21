from keras.models import Model, load_model
from keras.layers import Input, Dense, Conv2D, Dropout, MaxPooling2D, Lambda, Embedding, Reshape, Activation, Flatten
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.utils import np_utils
import keras.backend as K
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
import numpy as np
import pickle
from layer import CharacterEmbeddingLayer
from architecture import simple
from callbacks import EpochSaliency, GifSaliency
from copy import deepcopy
import random

def callbacks(paths):
    # configure callback function
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    model_checkpoint = ModelCheckpoint(paths["model_path"], verbose=1, save_best_only=True)
    tensor_board = TensorBoard(log_dir=paths["log_dir_path"], histogram_freq=0)
    return [early_stopping, model_checkpoint, tensor_board]


def load_dataset(conf):
    with open(conf["paths"]["preprocessed_path"], "rb") as f:
        dataset = pickle.load(f)
    return dataset['x_train'], dataset['y_train'], dataset['x_test'], dataset['y_test']


def train_with_saliency(conf, architecture=simple, verbose=1):
    x, y, x_test, y_test = load_dataset(conf)
    # parameter
    batch_size = conf["train_parameters"]["batch_size"]
    epochs = conf["train_parameters"]["epochs"]
    # generate model
    model = architecture(conf)
    for layer in model.layers:
        layer.trainable = True
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.001, decay=0.0000),
                  metrics=['accuracy'])
    model.summary()
    #x, y = x[:10000], y[:10000]
    x_t, x_val, y_t, y_val = train_test_split(x, y, test_size=0.1, random_state=0)

    # random choice
    random.seed(0)
    indexes = random.sample(range(len(x_val)), 50)
    sample = deepcopy(np.array(x_val)[indexes])
    label = deepcopy(np.array(y_val)[indexes])

    x_t, x_val = x_t[x_t.shape[0] % batch_size:], x_val[x_val.shape[0] % batch_size:]
    y_t, y_val = y_t[y_t.shape[0] % batch_size:], y_val[y_val.shape[0] % batch_size:]
    x_t, x_val = x_t.reshape(*x_t.shape, 1), x_val.reshape(*x_val.shape, 1)


    epoch_saliency = GifSaliency(conf, sample, label)
    paths = conf["paths"]
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    model_checkpoint = ModelCheckpoint(paths["model_path"], verbose=1, save_best_only=True)
    tensor_board = TensorBoard(log_dir=paths["log_dir_path"], histogram_freq=0)
    cb = [epoch_saliency, early_stopping, model_checkpoint, tensor_board]

    result = model.fit(x_t, y_t, epochs=epochs, batch_size=batch_size, callbacks=cb,
                       verbose=verbose, validation_data=(x_val, y_val))
    model.save(conf["paths"]["model_path"])
    print(result.history)
    print("=======" * 12, end="\n\n\n")

    with open(conf['paths']['log_dir_path'] + 'result.pkl', 'wb') as f:
        pickle.dump(result.history, f)

    ev_result = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=verbose)
    print(ev_result)


def test(x, y, conf):
    batch_size = conf["train_parameters"]["batch_size"]
    model = load_model(conf["paths"]["model_path"])
    x = x.reshape(x.shape[0], 1, x.shape[1])
    x, y = x[x.shape[0] % batch_size:], y[y.shape[0] % batch_size:]
    print("Test Score: ", model.evaluate(x, y, batch_size=batch_size))

