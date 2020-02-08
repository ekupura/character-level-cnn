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


def plot_train_loss_and_acc(conf_path):
    with open(conf_path, 'r') as f:
        conf = yaml.load(f)
    log_path = conf["paths"]["log_dir_path"]
    with open(log_path + "result.pkl", "rb") as f:
        results = pickle.load(f)
    epochs = conf["train_parameters"]["epochs"]

    # loss
    f = plt.figure()
    x = np.arange(epochs)
    plt.xlim(0, epochs)
    plt.xlabel("epoch")
    y = results["loss"]
    plt.plot(x, y)
    plt.savefig("{}loss.png".format(log_path))
    plt.close(f)

    # accuracy
    f = plt.figure()
    x = np.arange(epochs)
    plt.xlim(0, epochs)
    plt.xlabel("epoch")
    y = results["acc"]
    plt.plot(x, y)
    plt.savefig("{}accuracy.png".format(log_path))
    plt.close(f)



if __name__ == '__main__':
    model1 = "./experiment/origin2_1/configuration.yml"
    model2 = "./experiment/origin2_2/configuration.yml"
    model3 = "./experiment/origin2_3/configuration.yml"
    model4 = "./experiment/origin2_4/configuration.yml"
    confs = [model1, model2, model3, model4]
    #test_accuracy(confs)