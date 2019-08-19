from keras.models import load_model
import yaml
import pickle
import preprocess
import simplecnn
from saliency import saliency, saliency_to_png, generate_heatmap


class Main(object):
    def __init__(self):
        pass

    def preprocess_and_train(self, configuration_path):
        configuration = self._load_configuration(configuration_path)
        self._do_and_dump_preprocess(configuration)
        self._train_and_dump_modal(configuration)

    def saliency_test(self, configuration_path):
        configuration = self._load_configuration(configuration_path)
        dataset = self._load_preprocessed_dataset(configuration)
        test_text = dataset['x_test'][10]
        test_label = dataset['y_test'][10]
        sali = saliency(configuration, test_text, test_label)
        saliency_to_png(sali)
        text = preprocess.id_list_to_characters(test_text)
        print(text)

    def generate_saliency_heatmap(self, configuration_path):
        configuration = self._load_configuration(configuration_path)
        dataset = self._load_preprocessed_dataset(configuration)
        sample_text = dataset['x_train'][100]
        sample_label = dataset['y_train'][100]
        text = preprocess.id_list_to_characters(sample_text)
        saliency_vector = saliency(configuration, sample_text, sample_label)
        generate_heatmap(configuration, saliency_vector, text)

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

    def _do_and_dump_preprocess(self, configuration):
        paths, param = configuration["paths"], configuration["preprocessing_parameters"]
        x_train, x_test, y_train, y_test = preprocess.preprocess(paths["dataset_path"], param["limit_characters"])
        with open(paths["preprocessed_path"], "wb") as f:
            pickle.dump({'x_train': x_train, 'x_test': x_test, 'y_train': y_train, 'y_test': y_test}, f)

    def _train_and_dump_modal(self, configuration):
        print("Start training")
        with open(configuration["paths"]["preprocessed_path"], "rb") as f:
            dataset = pickle.load(f)
        simplecnn.train(dataset['x_train'], dataset['y_train'], configuration)

    def _print_text_by_index(self, index, configuration):
        dataset_path = configuration["paths"]["dataset_path"]
        text = preprocess.search_text_by_index(dataset_path, index)
        print(text)


if __name__ == '__main__':
    import fire
    fire.Fire(Main)