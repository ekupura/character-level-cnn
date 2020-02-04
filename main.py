from keras.models import load_model
import yaml
import pickle
from copy import deepcopy
import preprocessing
import training
import architecture
from saliency import calculate_saliency, generate_animation_heatmap, calculate_saliency_with_vis
from evaluate import Evaluation
from architecture import simple, two_convolution, numerous_convolution
import architecture
from layer import CharacterEmbeddingLayer
import time


class Main(object):
    def __init__(self):
        pass

    def preprocess(self, conf_path, aug=False, kfold=False):
        configuration = self._load_configuration(conf_path)
        data_type = configuration["preprocessing_parameters"]["architecture"]
        # select whether to do preprocessing
        if data_type == 'sentiment140':
            preprocessing.preprocess_sentiment140(configuration, dump=True, aug=aug, kfold=kfold)
        else:
            preprocessing.preprocess_imdb(configuration, dump=True, aug=aug)

    def train(self, conf_path, prep=False, aug=False, kfold=False, multi_gpu=False, debug=False, verbose=1):
        configuration = self._load_configuration(conf_path)
        model = configuration["model_parameters"]["architecture"]
        data_type = configuration["preprocessing_parameters"]["architecture"]
        # select whether to do preprocessing
        if prep:
            if data_type == 'sentiment140':
                preprocessing.preprocess_sentiment140(configuration, dump=True, aug=aug, kfold=kfold)
            else:
                preprocessing.preprocess_imdb(configuration, dump=True, aug=aug)

        # select model architecture
        if model == 'two':
            archi = two_convolution
        elif model == 'numerous':
            archi = numerous_convolution
        elif model == 'origin1':
            archi = architecture.character_level_cnn_origin
        elif model == 'bilstm':
            archi = architecture.character_level_cnn_bilstm
        elif model == 'parallel':
            archi = architecture.character_level_cnn_parallel
        elif model == 'serial':
            archi = architecture.character_level_cnn_serial
        elif model == 'serialparallel':
            archi = architecture.character_level_cnn_serial_and_parallel
        else:
            archi = simple
        # select whether to generate saliency map
        if kfold:
            training.train_model_kfold(configuration, archi, verbose, multi_gpu, debug)
        else:
            training.train_model(configuration, archi, verbose, multi_gpu, debug)

    def generate_saliency_gif(self, configuration_path):
        configuration = self._load_configuration(configuration_path)
        pkl_path = configuration["paths"]["saliency_pkl_path"]
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
            for i, j, k, l in zip(data[0], data[1], data[2], data[3]):
                generate_animation_heatmap(configuration, i, j, k, l)


    def keyword_evaluation(self, conf_path):
        configuration = self._load_configuration(conf_path)
        keywords = ["good", "congratulation", "people", "time"]
        evaluation = Evaluation()
        evaluation.keyword_evaluation(configuration, keywords, 20)

    def _load_configuration(self, configuration_path):
        print("Load configuration")
        with open(configuration_path, 'r') as f:
            configuration = yaml.load(f)
        return configuration

    def _load_trained_model(self, configuration):
        path = configuration["paths"]["model_path"]
        return load_model(path)

    def _load_preprocessed_dataset(self, configuration):
        path = configuration['paths']['preprocessed_path']
        with open(path, 'rb') as f:
            dataset = pickle.load(f)
        return dataset


if __name__ == '__main__':
    import fire
    fire.Fire(Main)
