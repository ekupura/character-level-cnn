from keras.models import Model, load_model
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from architecture import simple
from callbacks import EpochSaliency, GifSaliency
from copy import deepcopy
import random
from keras.utils.training_utils import multi_gpu_model
from pprint import pprint
from plot import plot_train_loss_and_acc
from alt_model_checkpoint.keras import AltModelCheckpoint
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot, plot_model


def load_dataset(conf):
    with open(conf["paths"]["preprocessed_path"], "rb") as f:
        dataset = pickle.load(f)
    return dataset['x_train'], dataset['y_train'], dataset['x_test'], dataset['y_test']


def load_dataset_kfold(conf, k):
    with open(conf["paths"]["preprocessed_path"] + '.{}'.format(k), "rb") as f:
        dataset = pickle.load(f)
    return dataset['x_train'], dataset['y_train'], dataset['x_test'], dataset['y_test']


def train_model(conf, architecture=simple, verbose=1, multi_gpu=False, debug=False):
    # parameter
    batch_size = conf["train_parameters"]["batch_size"]
    epochs = conf["train_parameters"]["epochs"]

    # generate model
    x, y, x_test, y_test = load_dataset(conf)
    opt = Adam(lr=0.001)
    model_original = architecture(conf)
    for layer in model_original.layers:
        layer.trainable = True
    if multi_gpu:
        batch_size = batch_size * 4
        model = multi_gpu_model(model_original, gpus=4)
        model_original.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        model_original.summary()
    else:
        model = model_original

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()

    if debug:
        epochs = 1
        deb_idx = random.sample(range(len(x)), 10000)
        x, y = x[deb_idx], y[deb_idx]
        deb_idx = random.sample(range(len(x)), 100)
        x_test, y_test = x_test[deb_idx], y_test[deb_idx]

    x_t, x_val, y_t, y_val = train_test_split(x, y, test_size=0.1, random_state=0)

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
    tensor_board = TensorBoard(log_dir=paths["log_dir_path"], histogram_freq=0)
    if multi_gpu:
        model_checkpoint = AltModelCheckpoint(paths["model_path"], model_original, verbose=1, save_best_only=True)
    else:
        model_checkpoint = ModelCheckpoint(paths["model_path"], verbose=1, save_best_only=True)
    cb = [model_checkpoint, tensor_board]

    # train
    result = model.fit(x_t, y_t, epochs=epochs, batch_size=batch_size, callbacks=cb,
                       verbose=verbose, validation_data=(x_val, y_val))
    model_original.save(conf["paths"]["model_path"])

    with open(conf['paths']['log_dir_path'] + 'result.pkl', 'wb') as f:
        pickle.dump(result.history, f)

    print("\n" + "===============================================================================" * 2)
    pprint(result.history)
    plot_train_loss_and_acc(conf)
    plot_model(model_original, to_file='{}model.png'.format(conf["paths"]["log_dir_path"]))

    # test
    x_test = x_test.reshape(*x_test.shape, 1)
    score_train = model_original.evaluate(x=x_t, y=y_t, batch_size=1024, verbose=verbose)
    score_test = model_original.evaluate(x=x_test, y=y_test, batch_size=1024, verbose=verbose)
    print("train loss: {}, train acc: {}, test loss: {}, test acc: {}".format(*score_train, *score_test))
    print("\n" + "===============================================================================" * 2)


def train_model_kfold(conf, architecture=simple, verbose=1, multi_gpu=False, debug=False):
    # parameter
    batch_size = conf["train_parameters"]["batch_size"]
    epochs = conf["train_parameters"]["epochs"]
    k_result = []

    # generate model
    for k in range(5):
        x, y, x_test, y_test = load_dataset_kfold(conf, k)
        opt = Adam(lr=0.001)
        model_original = architecture(conf)
        for layer in model_original.layers:
            layer.trainable = True
        if multi_gpu:
            batch_size = batch_size * 4
            model = multi_gpu_model(model_original, gpus=4)
            model_original.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
            model_original.summary()
        else:
            model = model_original

        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        model.summary()

        if debug:
            epochs = 1
            deb_idx = random.sample(range(len(x)), 10000)
            x, y = x[deb_idx], y[deb_idx]
            deb_idx = random.sample(range(len(x)), 100)
            x_test, y_test = x_test[deb_idx], y_test[deb_idx]

        x_t, x_val, y_t, y_val = train_test_split(x, y, test_size=0.1, random_state=0)

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
        tensor_board = TensorBoard(log_dir=paths["log_dir_path"], histogram_freq=0)
        if multi_gpu:
            model_checkpoint = AltModelCheckpoint(paths["model_path"], model_original, verbose=1, save_best_only=True)
        else:
            model_checkpoint = ModelCheckpoint(paths["model_path"], verbose=1, save_best_only=True)
        cb = [model_checkpoint, tensor_board]

        # train
        result = model.fit(x_t, y_t, epochs=epochs, batch_size=batch_size, callbacks=cb,
                           verbose=verbose, validation_data=(x_val, y_val))
        model_original.save(conf["paths"]["model_path"] + '.{}'.format(k))

        with open(conf['paths']['log_dir_path'] + 'result.pkl' + '.{}'.format(k), 'wb') as f:
            pickle.dump(result.history, f)

        print("\n" + "===============================================================================" * 2)
        pprint(result.history)
        # test
        x_test = x_test.reshape(*x_test.shape, 1)
        score_train = model_original.evaluate(x=x_t, y=y_t, batch_size=1024, verbose=verbose)
        score_test = model_original.evaluate(x=x_test, y=y_test, batch_size=1024, verbose=verbose)
        print("train loss: {}, train acc: {}, test loss: {}, test acc: {}".format(*score_train, *score_test))
        print("\n" + "===============================================================================" * 2)
        k_result.append([*score_train, *score_test])
    result_mean = np.mean(k_result, axis=0)

    print("all results:")
    print("train loss: {}, train acc: {}, test loss: {}, test acc: {}".format(*result_mean))


def test(x, y, conf):
    batch_size = conf["train_parameters"]["batch_size"]
    model = load_model(conf["paths"]["model_path"])
    x = x.reshape(x.shape[0], 1, x.shape[1])
    x, y = x[x.shape[0] % batch_size:], y[y.shape[0] % batch_size:]
    print("Test Score: ", model.evaluate(x, y, batch_size=batch_size))

