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
        #layer_dict = {'CharacterEmbeddingLayer': CharacterEmbeddingLayer}
        layer_dict = {}
        model = load_model(model_path, custom_objects=layer_dict)

        print(experiment_name)
        # x_test = np_utils.to_categorical(dataset['x_test'])
        x_test = (dataset['x_test'])
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
        with open(log_path + "result.pkl", "rb") as f:
            results[experiment_name] = pickle.load(f)

    # val_loss
    #plt.ylim(0.4, 0.55)
    val_loss_max = max([len(result["loss"]) for result in results.values()])
    x = np.arange(val_loss_max)
    plt.xlim(0, val_loss_max-1)
    plt.xlabel("epoch")
    name = ["1_conv", "2_conv", "3_conv", "4_conv"]
    for i, (label, result) in enumerate(results.items()):
        y = result["loss"] + [None] * (val_loss_max - len(result["loss"]))
        plt.plot(x, y, label=name[i])
    plt.legend(loc='upper right')
    plt.title("loss")
    plt.savefig("loss.png")

    # val_acc
    plt.figure()
    plt.legend(loc='upper right')
    #plt.ylim(0.73, 0.82)
    val_acc_max = max([len(result["acc"]) for result in results.values()])
    x = np.arange(val_loss_max)
    plt.xlim(0, val_acc_max-1)
    plt.xlabel("epoch")
    for label, result in results.items():
        y = result["val_loss"] + [None] * (val_acc_max - len(result["acc"]))
        plt.plot(x, y, label=label)
    plt.legend(loc='lower right')
    plt.title("validation_loss")
    plt.savefig("val_loss.png")



if __name__ == '__main__':
    model1 = "./experiment/origin2_1/configuration.yml"
    model2 = "./experiment/origin2_2/configuration.yml"
    model3 = "./experiment/origin2_3/configuration.yml"
    model4 = "./experiment/origin2_4/configuration.yml"
    confs = [model1, model2, model3, model4]
    plot_loss_and_acc(confs)
    #test_accuracy(confs)