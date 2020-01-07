from keras.models import load_model
import yaml
import pickle
from copy import deepcopy
import preprocess
import training
from saliency import calculate_saliency, generate_animation_heatmap, calculate_saliency_with_vis
from evaluation import Evaluation
from architecture import simple, two_convolution, numerous_convolution, autoencoder, autoencoder1d
from architecture import character_level_cnn_concatenate
import architecture
from layer import CharacterEmbeddingLayer


class Main(object):
    def __init__(self):
        pass

    def train(self, conf_path, prep=False, aug=False, auto=False, verbose=1):
        configuration = self._load_configuration(conf_path)
        model = configuration["model_parameters"]["architecture"]
        data_type = configuration["preprocessing_parameters"]["architecture"]
        # select whether to do preprocessing
        if prep:
            if data_type == 'sentiment140':
                preprocess.preprocess_sentiment140(configuration, dump=True, aug=aug)
            else:
                preprocess.preprocess_imdb(configuration, dump=True, aug=aug)

        # select model architecture
        if model == 'two':
            archi = two_convolution
        elif model == 'numerous':
            archi = numerous_convolution
        elif model == 'origin':
            archi = architecture.character_level_cnn_origin
        elif model == 'charcnn3':
            archi = character_level_cnn_concatenate
        elif model == 'auto':
            archi = autoencoder1d
        else:
            archi = simple
        # select whether to generate saliency map
        if model == 'auto':
            #training.train_autoencoder(configuration, archi, verbose)
            training.train_classify_cnn(configuration)
        else:
            training.train_with_saliency(configuration, archi, verbose, auto)

    def generate_saliency_gif(self, configuration_path):
        configuration = self._load_configuration(configuration_path)
        pkl_path = configuration["paths"]["saliency_pkl_path"]
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
            for i, j, k, l in zip(data[0], data[1], data[2], data[3]):
                generate_animation_heatmap(configuration, i, j, k, l)

    def test_evaluation(self, configuration_path):
        configuration = self._load_configuration(configuration_path)
        evaluation = Evaluation()
        evaluation.single_keyword_evaluation(configuration)

    def do_simple_evaluation(self, configuration_path):
        configuration = self._load_configuration(configuration_path)
        evaluation = Evaluation()
        evaluation.good_and_bad_evaluation(configuration)

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