from matplotlib import pyplot as plt
from keras.utils import np_utils
import pickle
import yaml
import numpy as np
from keras.models import load_model
from layer import CharacterEmbeddingLayer

def test_accuracy(confs):
    for conf_path in confs:
        with open(conf_path, 'r') as f:
            conf = yaml.load(f)
        pre_path = conf["paths"]["preprocessed_path"]
        model_path = conf["paths"]["model_path"]
        experiment_name = conf["experiment_name"]
        with open(pre_path, "rb") as f:
            dataset = pickle.load(f)
        layer_dict = {'CharacterEmbeddingLayer': CharacterEmbeddingLayer}
        model = load_model(model_path, custom_objects=layer_dict)

        print(experiment_name)
        x_test = np_utils.to_categorical(dataset['x_test'])
        x_test = x_test.reshape(*x_test.shape, 1)
        y_test = dataset['y_test']
        score = model.evaluate(x_test, y_test)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

def plot_loss_and_acc(confs):
    results = {}
    for conf_path in confs:
        with open(conf_path, 'r') as f:
            conf = yaml.load(f)
        log_path = conf["paths"]["log_dir_path"]
        experiment_name = conf["experiment_name"]
        with open(log_path + "result0.pkl", "rb") as f:
            results[experiment_name] = pickle.load(f)

    # val_loss
    plt.ylim(0.4, 0.55)
    val_loss_max = max([len(result["loss"]) for result in results.values()])
    x = np.arange(val_loss_max)
    plt.xlim(0, val_loss_max-1)
    plt.xlabel("epoch")
    for label, result in results.items():
        y = result["loss"] + [None] * (val_loss_max - len(result["loss"]))
        plt.plot(x, y, label=label)
    plt.legend(loc='upper right')
    plt.title("loss")
    plt.savefig("loss.png")

    # val_acc
    plt.figure()
    plt.legend(loc='upper right')
    plt.ylim(0.73, 0.82)
    val_acc_max = max([len(result["acc"]) for result in results.values()])
    x = np.arange(val_loss_max)
    plt.xlim(0, val_acc_max-1)
    plt.xlabel("epoch")
    for label, result in results.items():
        y = result["acc"] + [None] * (val_acc_max - len(result["acc"]))
        plt.plot(x, y, label=label)
    plt.legend(loc='lower right')
    plt.title("accuracy")
    plt.savefig("acc.png")



if __name__ == '__main__':
    simple1 = "./experiment/simpleconv_01/configuration.yml"
    simple2 = "./experiment/simpleconv_02/configuration.yml"
    simple3 = "./experiment/simpleconv_03/configuration.yml"
    two = "./experiment/twoconv_03/configuration.yml"
    confs = [simple1, simple2, simple3, two]
    plot_loss_and_acc(confs)
    test_accuracy(confs)