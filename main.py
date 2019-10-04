from keras.models import load_model
import yaml
import pickle
import preprocess
import train
from saliency import calculate_saliency, generate_heatmap, calculate_saliency_with_vis
from evaluation import Evaluation
from architecture import simple, two_convolution
from layer import CharacterEmbeddingLayer


class Main(object):
    def __init__(self):
        pass

    def preprocess_and_train(self, configuration_path):
        configuration = self._load_configuration(configuration_path)
        self._do_and_dump_preprocess(configuration)
        self._train_and_dump_modal(configuration, two_convolution)

    def preprocess_and_train_simple(self, configuration_path):
        configuration = self._load_configuration(configuration_path)
        self._do_and_dump_preprocess(configuration)
        self._train_and_dump_modal(configuration, simple)

    def generate_saliency_heatmap(self, configuration_path):
        configuration = self._load_configuration(configuration_path)
        dataset = self._load_preprocessed_dataset(configuration)
        sample_text = dataset['x_train'][111]
        sample_label = dataset['y_train'][111]
        """
        layer_dict = {'CharacterEmbeddingLayer': CharacterEmbeddingLayer}
        model =  load_model(configuration['paths']['model_path'], custom_objects=layer_dict)
        sample_predict = model.predict(sample_text)
        print(sample_predict)
        """
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

    # do preprocess
    def _do_and_dump_preprocess(self, configuration):
        paths, param = configuration["paths"], configuration["preprocessing_parameters"]
        x_train, x_test, y_train, y_test = preprocess.preprocess(paths["dataset_path"], param["limit_characters"])
        with open(paths["preprocessed_path"], "wb") as f:
            pickle.dump({'x_train': x_train, 'x_test': x_test, 'y_train': y_train, 'y_test': y_test}, f)

    # train
    def _train_and_dump_modal(self, configuration, architecture=simple):
        print("Start training")
        with open(configuration["paths"]["preprocessed_path"], "rb") as f:
            dataset = pickle.load(f)
        train.train(dataset['x_train'], dataset['y_train'], configuration, architecture)


if __name__ == '__main__':
    import fire
    fire.Fire(Main)