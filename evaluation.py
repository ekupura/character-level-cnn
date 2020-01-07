import os
import nltk
from nltk.corpus import wordnet as wn
import pickle
import numpy as np
from keras.models import load_model
from collections import defaultdict
from tqdm import tqdm
from preprocess import id_list_to_characters
from saliency import calculate_saliency, generate_heatmap


class Evaluation:
    def __init__(self):
        nltk.download('wordnet')

    def collect_synonyms_for_sentiment(self):
        # collect synonyms for 'good'
        good_synonyms = []
        for ss in wn.synsets('good'):
            good_synonyms.extend(ss.lemma_names())
            good_synonyms.extend([sim.lemma_names()[0] for sim in ss.similar_tos()])
        good_synonyms = list(set(good_synonyms))

        # collect synonyms for 'bad'
        bad_synonyms = []
        for ss in wn.synsets('bad'):
            bad_synonyms.extend(ss.lemma_names())
            bad_synonyms.extend([sim.lemma_names()[0] for sim in ss.similar_tos()])
        bad_synonyms = list(set(bad_synonyms))

        return good_synonyms, bad_synonyms

    def search_keyword(self, dataset, keywords):
        #train_index = self._get_index(dataset['x_train'], keywords)
        test_index = self._get_index(dataset['x_test'], keywords)
        #return train_index, test_index
        return test_index

    def single_keyword_evaluation(self, configuration, good_bad=False):
        path = configuration['paths']['preprocessed_path']
        with open(path, 'rb') as f:
            dataset = pickle.load(f)

        if good_bad:
            keywords = ['good', 'comfortable', 'beautiful']
        else:
            good, bad = self.collect_synonyms_for_sentiment()
            keywords = good + bad

        test_index = self.search_keyword(dataset, keywords)

        def exclusion(index):
            return_index = defaultdict(list)
            for key in keywords:
                key_index_set = set(index[key])
                for other_key in keywords:
                    if key == other_key:
                        continue
                    other_index_set = set(index[other_key])
                    key_index_set = key_index_set - other_index_set
                return_index[key] = list(key_index_set)
            return return_index

        #single_train_index = exclusion(train_index)
        single_test_index = exclusion(test_index)

        def sample_text_to_saliency_heatmap(index, x, y, max_heatmap=10):
            dir_root_path = configuration['paths']['saliency_dir_path']
            for key in keywords:
                keyword_dir = dir_root_path + key + '/'
                if not os.path.exists(keyword_dir):
                    os.mkdir(keyword_dir)
                print("key = {}".format(key))
                for i, v in enumerate(index[key]):
                    if i == max_heatmap:
                        break
                    saliency = calculate_saliency(configuration, x[v], y[v])
                    generate_heatmap(configuration, saliency, x[v], y[v], path=keyword_dir + str(i) + '.png')

        #sample_text_to_saliency_heatmap(single_train_index, dataset['x_train'], dataset['y_train'])
        sample_text_to_saliency_heatmap(single_test_index, dataset['x_test'], dataset['y_test'])

    def _get_index(self, dataset, keywords):
        index = defaultdict(list)
        for i, v in tqdm(enumerate(dataset)):
            text = id_list_to_characters(v)
            for k in keywords:
                # if keyword is included in text
                if k in text:
                    # save text index
                    index[k].append(i)
        return index

    def good_and_bad_evaluation(self, configuration):
        self.single_keyword_evaluation(configuration, True)