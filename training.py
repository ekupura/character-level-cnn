from keras.models import Model, load_model
from keras.layers import Input, Dense, Conv2D, Dropout, MaxPooling2D, Lambda, Embedding, Reshape, Activation, Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.utils import np_utils
import keras.backend as K
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
import numpy as np
import pickle
from layer import CharacterEmbeddingLayer
from architecture import simple, classification_cnn, classification_dense
from callbacks import EpochSaliency, GifSaliency
from copy import deepcopy
import random
from keras.utils.training_utils import multi_gpu_model

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

def load_autoencoder_dataset(conf):
    with open(conf["paths"]["preprocessed_path"], "rb") as f:
        dataset = pickle.load(f)
    return dataset['x_train'], dataset['x_train'], dataset['x_test'], dataset['x_test']


def train_with_saliency(conf, architecture=simple, verbose=1, autoencoder=True):
    x, y, x_test, y_test = load_dataset(conf)
    # parameter
    batch_size = conf["train_parameters"]["batch_size"]
    epochs = conf["train_parameters"]["epochs"]
    # generate model
    model_original = architecture(conf)
    for layer in model_original.layers:
        layer.trainable = True
    multi_gpu = True
    if multi_gpu:
        batch_size = batch_size * 4
        model = multi_gpu_model(model_original, gpus=4)
        model_original.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        model_original.summary()
    else:
        model = model_original

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.summary()

    #x, y = x[:50000], y[:50000]
    #x_test, y_test = x_test[:50000], y_test[:50000]
    x_t, x_val, y_t, y_val = train_test_split(x, y, test_size=0.2, random_state=0)

    # random choice
    random.seed(0)
    indexes = random.sample(range(len(x_val)), 10)
    sample = deepcopy(np.array(x_val)[indexes])
    label = deepcopy(np.array(y_val)[indexes])

    x_t, x_val = x_t[x_t.shape[0] % batch_size:], x_val[x_val.shape[0] % batch_size:]
    y_t, y_val = y_t[y_t.shape[0] % batch_size:], y_val[y_val.shape[0] % batch_size:]
    x_t, x_val = x_t.reshape(*x_t.shape, 1), x_val.reshape(*x_val.shape, 1)

    # callbacks
    epoch_saliency = GifSaliency(conf, sample, label, gif=False)
    paths = conf["paths"]
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    model_checkpoint = ModelCheckpoint(paths["model_path"], verbose=1, save_best_only=True)
    tensor_board = TensorBoard(log_dir=paths["log_dir_path"], histogram_freq=0)
    # cb = [epoch_saliency, early_stopping, model_checkpoint, tensor_board]
    # cb = [early_stopping, model_checkpoint, tensor_board]
    cb = [early_stopping, tensor_board]

    # train
    result = model.fit(x_t, y_t, epochs=epochs, batch_size=batch_size, callbacks=cb,
                       verbose=verbose, validation_data=(x_val, y_val))
    model_original.save(conf["paths"]["model_path"])
    print(result.history)
    print("=======" * 12, end="\n\n\n")

    with open(conf['paths']['log_dir_path'] + 'result.pkl', 'wb') as f:
        pickle.dump(result.history, f)

    # test
    x_test = x_test.reshape(*x_test.shape, 1)
    score = model_original.evaluate(x=x_test, y=y_test, batch_size=1024, verbose=verbose)
    print(list(zip(model.metrics_names, score)))


def train_autoencoder(conf, architecture=simple, verbose=1):
    x, y, x_test, y_test = load_autoencoder_dataset(conf)

    # parameter
    batch_size = conf["train_parameters"]["batch_size"]
    epochs = conf["train_parameters"]["epochs"]
    # generate model
    model, ae, encoder = architecture(conf)
    for layer in model.layers:
        layer.trainable = True

    def original_loss(y_true, y_pred):
        return K.mean(y_pred, axis=-1)

    model.compile(loss=original_loss,
                  optimizer=Adam(lr=0.001),
                  metrics=[original_loss])
    model.summary()
    # x, y = x[:10000], y[:10000]
    x_t, x_val, y_t, y_val = train_test_split(x, y, test_size=0.25, random_state=0)

    x_t, x_val = x_t[x_t.shape[0] % batch_size:], x_val[x_val.shape[0] % batch_size:]
    y_t, y_val = y_t[y_t.shape[0] % batch_size:], y_val[y_val.shape[0] % batch_size:]
    x_t, x_val = x_t.reshape(*x_t.shape, 1), x_val.reshape(*x_val.shape, 1)
    y_t, y_val = y_t.reshape(*y_t.shape, 1), y_val.reshape(*y_val.shape, 1)

    paths = conf["paths"]
    early_stopping = EarlyStopping(monitor='val_loss', patience=1, verbose=1)
    tensor_board = TensorBoard(log_dir=paths["log_dir_path"], histogram_freq=0)
    cb = [early_stopping, tensor_board]

    result = model.fit(x_t, y_t, epochs=1, batch_size=32, callbacks=cb,
                       verbose=verbose, validation_data=(x_val, y_val))
    encoder.save(conf["paths"]["encoder_path"])

    print(result.history)
    print("=======" * 12, end="\n\n\n")

    with open(conf['paths']['log_dir_path'] + 'result.pkl', 'wb') as f:
        pickle.dump(result.history, f)


def train_classify_cnn(conf):
    x, y, x_test, y_test = load_dataset(conf)

    batch_size = conf["train_parameters"]["batch_size"]
    epochs = conf["train_parameters"]["epochs"]
    # generate model
    model = classification_cnn(conf)
    for layer in model.layers:
        layer.trainable = True
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=0.001),
                  metrics=['accuracy'])
    model.summary()
    encoder = load_model(conf["paths"]["encoder_path"])

    x_t, x_val, y_t, y_val = train_test_split(x, y, test_size=0.25, random_state=0)
    x_t, x_val = x_t[x_t.shape[0] % batch_size:], x_val[x_val.shape[0] % batch_size:]
    y_t, y_val = y_t[y_t.shape[0] % batch_size:], y_val[y_val.shape[0] % batch_size:]
    y_t, y_val = np.argmax(y_t, axis=1), np.argmax(y_val, axis=1)
    x_t, x_val = x_t.reshape(*x_t.shape, 1), x_val.reshape(*x_val.shape, 1)

    print("encoding...")
    x_t, x_val = encoder.predict(x_t, batch_size=128), encoder.predict(x_val, batch_size=128)

    paths = conf["paths"]
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    model_checkpoint = ModelCheckpoint(paths["classification_path"], verbose=1, save_best_only=True)
    tensor_board = TensorBoard(log_dir=paths["log_dir_path"], histogram_freq=0)
    cb = [early_stopping, model_checkpoint, tensor_board]

    result = model.fit(x_t, y_t, epochs=epochs, batch_size=batch_size, callbacks=cb,
                       verbose=1, validation_data=(x_val, y_val))
    model.save(conf["paths"]["classification_path"])
    print(result.history)
    print("=======" * 12, end="\n\n\n")

    with open(conf['paths']['log_dir_path'] + 'result.pkl', 'wb') as f:
        pickle.dump(result.history, f)

    x_test = x_test.reshape(*x_test.shape, 1)
    x_test = encoder.predict(x_test, batch_size=128)
    y_test = np.argmax(y_test)
    score = model.evaluate(x=x_test, y=y_test, batch_size=1, verbose=1)
    print(list(zip(model.metrics_names, score)))


def test(x, y, conf):
    batch_size = conf["train_parameters"]["batch_size"]
    model = load_model(conf["paths"]["model_path"])
    x = x.reshape(x.shape[0], 1, x.shape[1])
    x, y = x[x.shape[0] % batch_size:], y[y.shape[0] % batch_size:]
    print("Test Score: ", model.evaluate(x, y, batch_size=batch_size))

