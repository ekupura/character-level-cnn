from keras.models import load_model
import yaml
import pickle
from copy import deepcopy
import preprocess
import training
from saliency import calculate_saliency, generate_heatmap, calculate_saliency_with_vis
from evaluation import Evaluation
from architecture import simple, two_convolution, seven_convolution
from layer import CharacterEmbeddingLayer


class Main(object):
    def __init__(self):
        pass

    def train(self, conf_path, prep=False, sali=False, model='simple', aug=False, verbose=1):
        configuration = self._load_configuration(conf_path)
        model = configuration["model_parameters"]["architecture"]
        # select whether to do preprocessing
        if prep:
            preprocess.preprocess_wrap(configuration, dump=True, aug=aug)
        # select model architecture
        if model == 'two':
            archi = two_convolution
        elif model == 'seven':
            archi = seven_convolution
        else:
            archi = simple
        # select whether to generate saliency map
        training.train_with_saliency(configuration, archi, verbose)


    # testing saliency map
    def generate_saliency_heatmap(self, configuration_path):
        configuration = self._load_configuration(configuration_path)
        dataset = self._load_preprocessed_dataset(configuration)
        sample_text = dataset['x_train'][111]
        sample_label = dataset['y_train'][111]
        saliency_vector = calculate_saliency_with_vis(configuration, sample_text, sample_label)
        generate_heatmap(configuration, saliency_vector, sample_text, sample_label)

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