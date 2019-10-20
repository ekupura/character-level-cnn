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

def callbacks(paths):
    # configure callback function
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    model_checkpoint = ModelCheckpoint(paths["model_path"], verbose=1, save_best_only=True)
    tensor_board = TensorBoard(log_dir=paths["log_dir_path"], histogram_freq=0)
    return [early_stopping, model_checkpoint, tensor_board]


def train(x, y, conf, architecture=simple):
    old_session = ktf.get_session()
    session = tf.Session('')
    ktf.set_session(session)
    ktf.set_learning_phase(1)
    for i in range(conf["train_parameters"]["n_folds"]):
        print("Training on Fold: ", i + 1)
        result = fit_and_evaluate(x, y, conf, architecture)
        with open(conf['paths']['log_dir_path'] + 'result' + str(i) + '.pkl', 'wb') as f:
            pickle.dump(result.history, f)
        print(result.history)
        print(type(result.history))
    ktf.set_session(old_session)


def fit_and_evaluate(x, y, conf, architecture=simple):
    # parameter
    batch_size = conf["train_parameters"]["batch_size"]
    epochs = conf["train_parameters"]["epochs"]
    print(x.shape)
    # generate model
    model = architecture(conf)
    for layer in model.layers:
        layer.trainable = True
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])
    model.summary()

    x_t, x_val, y_t, y_val = train_test_split(x, y, test_size=0.1, random_state=0)
    x_t, x_val = x_t[x_t.shape[0] % batch_size:], x_val[x_val.shape[0] % batch_size:]
    x_t, x_val = np_utils.to_categorical(x_t), np_utils.to_categorical(x_val)
    y_t, y_val = y_t[y_t.shape[0] % batch_size:], y_val[y_val.shape[0] % batch_size:]
    x_t, x_val = x_t.reshape(*x_t.shape, 1), x_val.reshape(*x_val.shape, 1)

    result = model.fit(x_t, y_t, epochs=epochs, batch_size=batch_size, callbacks=callbacks(conf["paths"]),
                       verbose=1, validation_data=(x_val, y_val))
    # print("Val Score: ", model.evaluate(x_val, y_val, batch_size=batch_size))
    print(K.gradients(model.output, model.input))
    model.save(conf["paths"]["model_path"])
    print("=======" * 12, end="\n\n\n")
    return result


def train_with_saliency(conf, x, y, architecture=simple):
    # parameter
    batch_size = conf["train_parameters"]["batch_size"]
    epochs = conf["train_parameters"]["epochs"]
    print(x.shape)
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
    sample = deepcopy(x_val[0])
    label = deepcopy(y_val[0])
    x_t, x_val = x_t[x_t.shape[0] % batch_size:], x_val[x_val.shape[0] % batch_size:]
    x_t, x_val = np_utils.to_categorical(x_t), np_utils.to_categorical(x_val)
    y_t, y_val = y_t[y_t.shape[0] % batch_size:], y_val[y_val.shape[0] % batch_size:]
    x_t, x_val = x_t.reshape(*x_t.shape, 1), x_val.reshape(*x_val.shape, 1)

    epoch_saliency = GifSaliency(conf, sample, label)
    paths = conf["paths"]
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    model_checkpoint = ModelCheckpoint(paths["model_path"], verbose=1, save_best_only=True)
    tensor_board = TensorBoard(log_dir=paths["log_dir_path"], histogram_freq=0)
    cb = [epoch_saliency, early_stopping, model_checkpoint, tensor_board]

    result = model.fit(x_t, y_t, epochs=epochs, batch_size=batch_size, callbacks=cb,
                       verbose=1, validation_data=(x_val, y_val))
    model.save(conf["paths"]["model_path"])
    print("=======" * 12, end="\n\n\n")

    with open(conf['paths']['log_dir_path'] + 'result.pkl', 'wb') as f:
        pickle.dump(result.history, f)
    print(result.history)


def test(x, y, conf):
    batch_size = conf["train_parameters"]["batch_size"]
    model = load_model(conf["paths"]["model_path"])
    x = x.reshape(x.shape[0], 1, x.shape[1])
    x, y = x[x.shape[0] % batch_size:], y[y.shape[0] % batch_size:]
    print("Test Score: ", model.evaluate(x, y, batch_size=batch_size))

