import os
import nltk
from nltk.corpus import wordnet as wn
import pickle
import numpy as np
from keras.models import load_model
from collections import defaultdict
from tqdm import tqdm
from preprocessing import id_list_to_characters
from saliency import calculate_saliency, generate_heatmap
import time


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

    def get_index(self, x_test, keyword):
        for i, v in enumerate(x_test):
            text = id_list_to_characters(v)
            if ' {} '.format(keyword) in text:
                # yield text index
                yield i

    def keyword_evaluation(self, configuration, keywords, n_saliency = 10):
        path = configuration['paths']['preprocessed_path']
        dir_root_path = configuration['paths']['saliency_dir_path']
        with open(path, 'rb') as f:
            dataset = pickle.load(f)
        x_test, y_test = dataset['x_test'], dataset['y_test']
        # layer_dict = {'CharacterEmbeddingLayer': CharacterEmbeddingLayer}
        model = load_model(configuration['paths']['model_path'])

        for key in keywords:
            for c, idx in enumerate(self.get_index(x_test, key)):
                if c > n_saliency:
                    break
                keyword_dir = dir_root_path + key + '/'
                png_path = '{}{}.png'.format(keyword_dir, c)
                if not os.path.exists(keyword_dir):
                    os.mkdir(keyword_dir)
                saliency = calculate_saliency(configuration, x_test[idx], y_test[idx], model=model)
                generate_heatmap(configuration, saliency, x_test[idx], y_test[idx], path=png_path)
