from keras.utils import np_utils
import pandas as pd
import numpy as np
import codecs
import re
import pickle
from sklearn.model_selection import train_test_split
from numpy.random import geometric
import random
import nltk
from nltk.corpus import wordnet
from collections import defaultdict
import glob
from tqdm import tqdm
from scipy.special import comb


class Synonym:
    def __init__(self):
        nltk.download('wordnet')
        self.synonims = defaultdict(list)

    def get_synonym(self, word, pos):
        if not self.synonims[word]:
            for syn in wordnet.synsets(word, pos):
                for l in syn.lemmas():
                    self.synonims[word].append(l.name())
        if self.synonims[word]:
            return random.choice(self.synonims[word])
        else:
            return word


class PreprocessBase:
    def __init__(self):
        pass

    def lemmatize(self, dataset):
            labels, texts = [], []
            nltk.download('punkt')
            nltk.download('wordnet')
            lemma = nltk.stem.WordNetLemmatizer()
            for l, t in tqdm(zip(dataset["label"], dataset["text"])):
                words = nltk.word_tokenize(t)
                stem_text = ' '.join([lemma.lemmatize(word) for word in words])
                texts.append(stem_text)
                labels.append(l)
            texts = pd.Series(texts)
            labels = pd.Series(labels)
            return pd.DataFrame({'text': texts, 'label': labels})

    def character_restriction(self, dataset, restriction_rule=r'[^\w!?,\s]'):
        labels, texts = [], []
        for l, t in zip(dataset["label"], dataset["text"]):
            re_text = re.sub(r'\s', ' ', re.sub(restriction_rule, '', t)).strip()
            if not re_text:
                continue
            texts.append(re_text)
            labels.append(l)
        texts = pd.Series(texts)
        labels = pd.Series(labels)
        return pd.DataFrame({'text': texts, 'label': labels})

    # character to ID
    def conversion_rule_for_alphabet(self, sentence):
        id_list = []
        for c in sentence:
            if c.islower():
                number = ord(c) - 96
            elif c.isupper():
                number = ord(c) - 38
            elif c.isdecimal():
                number = int(c) + 53
            elif c == '?':
                number = 63
            elif c == '!':
                number = 64
            elif c == ',':
                number = 65
            else:
                number = 66
            if number >= 0 and number <= 65:
                id_list.append(number)
            else:
                id_list.append(66)
        return id_list

    def texts_to_characters_id_lists(self, texts, limit_characters, conversion_rule=conversion_rule_for_alphabet):
        characters_id_lists = []
        for sentence in texts:
            id_list = self.conversion_rule_for_alphabet(sentence)
            id_list = id_list + [0 for i in range(limit_characters - len(id_list))]
            characters_id_lists.append(id_list[:limit_characters])

        return np.array(characters_id_lists)

    def labels_to_onehot(self, labels):
        return np_utils.to_categorical(list(labels))


    def transform_pos(self, nltk_pos):
        # noun
        if nltk_pos[0] == 'N':
            return 'n'
        # verb
        elif nltk_pos[0] == 'V':
            return 'v'
        # adverb
        elif nltk_pos[0] == 'R':
            return 'r'
        # adjective
        elif nltk_pos[0] == 'J':
            return 'a'
        else:
            return ''

    def data_augmentation(self, dataset, sample_num=100000, sample_rate=None):
        nltk.download('averaged_perceptron_tagger')
        nltk.download('punkt')
        synonym = Synonym()
        texts = list(dataset['text'])
        labels = list(dataset['label'])
        dataset_size = len(texts)
        new_texts, new_labels = [], []
        if sample_rate is not None:
            sample_num = int(dataset_size * sample_rate)
        replace_numbers = geometric(p=0.5, size=sample_num)
        for r in tqdm(replace_numbers):
            i = random.choice(range(dataset_size))
            t, l = texts[i], labels[i]
            words = nltk.word_tokenize(t)
            words_and_pos = nltk.pos_tag(words)
            r = r if r < len(words) else len(words)
            replace_idx = random.sample(range(len(words)), r)
            # replace word
            for i in replace_idx:
                word, pos = words_and_pos[i]
                pos = self.transform_pos(pos)
                if pos != '':
                    syn = synonym.get_synonym(word, pos)
                    # replace
                    words[i] = syn
            new_text = ' '.join(words)
            if t == new_text:
                continue
            new_texts.append(new_text)
            new_labels.append(l)
        texts.extend(new_texts)
        labels.extend(new_labels)
        texts = pd.Series(texts)
        labels = pd.Series(labels)
        _max = np.max(labels)
        return pd.DataFrame({'text': texts, 'label': labels})

    def data_augmentation_emuneration(self, dataset, sample_num=100000, sample_rate=None):
        nltk.download('averaged_perceptron_tagger')
        nltk.download('punkt')
        synonym = Synonym()
        texts = list(dataset['text'])
        labels = list(dataset['label'])
        dataset_size = len(texts)
        new_texts, new_labels = [], []
        for i in tqdm(range(dataset_size)):
            t, l = texts[i], labels[i]
            words = nltk.word_tokenize(t)
            words_and_pos = nltk.pos_tag(words)

            replaceable_idx = []
            for i, (w, p) in enumerate(words_and_pos):
                p = self.transform_pos(p)
                if p != '':
                    replaceable_idx.append(i)

            # combination
            for i in range(len(words)):
                word, pos = words_and_pos[i]
                pos = self.transform_pos(pos)
                if pos != '':
                    syn = synonym.get_synonym(word, pos)
                    # replace
                    words[i] = syn
            new_text = ' '.join(words)
            if t == new_text:
                continue
            new_texts.append(new_text)
            new_labels.append(l)
        texts.extend(new_texts)
        labels.extend(new_labels)
        texts = pd.Series(texts)
        labels = pd.Series(labels)
        _max = np.max(labels)
        return pd.DataFrame({'text': texts, 'label': labels})


class Sentiment140(PreprocessBase):
    def __init__(self):
        super().__init__()

    def common_preprocess_sentiment140(self, input_name):
        with codecs.open(input_name, "r", "UTF-8", "ignore") as file:
            df = pd.read_csv(file, header=None)
            labels, texts = [], []
            for l, t in zip(df[0], df[5]):
                if l == 2:
                    continue
                re_text = re.sub(r'(https?://+\S*\s*|www.\S*\s*|#\S*\s*|@\S*\s*|&\S*\s*)', '', t).strip()
                if not re_text:
                    continue
                labels.append(0 if l == 0 else 1)
                texts.append(re_text)
            texts = pd.Series(texts)
            labels = pd.Series(labels)
            dataset = pd.DataFrame({'text': texts, 'label': labels})
            return dataset

    def preprocess(self, file_name, limit_characters, number_of_characters, aug):
        extracted_dataset = self.common_preprocess_sentiment140(file_name)
        lemma_dataset = self.lemmatize(extracted_dataset)

        # split
        x_train, x_test, y_train, y_test = train_test_split(lemma_dataset['text'], lemma_dataset['label'], test_size=0.2, random_state=183)
        train_set = pd.DataFrame({'text': x_train, 'label': y_train})
        test_set = pd.DataFrame({'text': x_test, 'label': y_test})

        # train
        if aug:
            print("Do augmentation")
            train_set = self.data_augmentation(train_set, sample_rate=4.0)
        restricted_train = self.character_restriction(train_set, restriction_rule=r'[^\w!?,\s]')
        re_x_train = self.texts_to_characters_id_lists(restricted_train['text'], limit_characters)
        re_y_train = self.labels_to_onehot(restricted_train['label'])

        # test
        restricted_test = self.character_restriction(test_set, restriction_rule=r'[^\w!?,\s]')
        re_x_test = self.texts_to_characters_id_lists(restricted_test['text'], limit_characters)
        re_y_test = self.labels_to_onehot(restricted_test['label'])

        return re_x_train, re_x_test, re_y_train, re_y_test


class IMDB(PreprocessBase):
    def __init__(self):
        super().__init__()

    def read_row_dataset(self, root_dir_name):
        # training_dataset_negative
        texts, labels = [], []
        parent_dir = [root_dir_name + 'train/neg/*', root_dir_name + 'train/pos/*']
        for dir_name in parent_dir:
            filenames = glob.glob(dir_name)
            for name in filenames:
                with codecs.open(name, "r", "UTF-8", "ignore") as file:
                    text = file.read()
                    texts.append(re.sub(r'(https?://+\S*\s*|www.\S*\s*|#\S*\s*|@\S*\s*|&\S*\s*)', '', text).strip())
                    labels.append(0 if 'neg' in dir_name else 1)
        texts = pd.Series(texts)
        labels = pd.Series(labels)
        train_dataset = pd.DataFrame({'text': texts, 'label': labels})

        # testing_dataset_negative
        texts, labels = [], []
        parent_dir = [root_dir_name + 'test/neg/*', root_dir_name + 'test/pos/*']
        for dir_name in parent_dir:
            filenames = glob.glob(dir_name)
            for name in filenames:
                with codecs.open(name, "r", "UTF-8", "ignore") as file:
                    text = file.read()
                    texts.append(re.sub(r'(https?://+\S*\s*|www.\S*\s*|#\S*\s*|@\S*\s*|&\S*\s*)', '', text).strip())
                    labels.append(0 if 'neg' in dir_name else 1)
        texts = pd.Series(texts)
        labels = pd.Series(labels)
        test_dataset = pd.DataFrame({'text': texts, 'label': labels})

        return train_dataset, test_dataset

    def preprocess(self, imdb_dir_path, limit_characters, number_of_characters, aug):
        train_dataset, test_dataset = self.read_row_dataset(imdb_dir_path)
        train_set, test_set = self.lemmatize(train_dataset), self.lemmatize(test_dataset)

        # train
        if aug:
            print("Do augmentation")
            train_set = self.data_augmentation(train_set, sample_rate=4.0)
        restricted_train = self.character_restriction(train_set, restriction_rule=r'[^\w!?,\s]')
        re_x_train = self.texts_to_characters_id_lists(restricted_train['text'], limit_characters)
        re_y_train = self.labels_to_onehot(restricted_train['label'])

        # test
        restricted_test = self.character_restriction(test_set, restriction_rule=r'[^\w!?,\s]')
        re_x_test = self.texts_to_characters_id_lists(restricted_test['text'], limit_characters)
        re_y_test = self.labels_to_onehot(restricted_test['label'])

        return re_x_train, re_x_test, re_y_train, re_y_test


# preprocess and  dump
def preprocess_sentiment140(conf, dump=True, aug=False):
    paths, param = conf["paths"], conf["preprocessing_parameters"]
    s140 = Sentiment140()
    x_train, x_test, y_train, y_test = s140.preprocess(paths["dataset_path"], param["limit_characters"], param['number_of_characters'], aug)
    if dump:
        with open(paths["preprocessed_path"], "wb") as f:
            pickle.dump({'x_train': x_train, 'x_test': x_test, 'y_train': y_train, 'y_test': y_test}, f, protocol=4)
    else:
        return x_train, x_test, y_train, y_test


def preprocess_imdb(conf, dump=True, aug=False):
    paths, param = conf["paths"], conf["preprocessing_parameters"]
    imdb = IMDB()
    x_train, x_test, y_train, y_test = imdb.preprocess(paths["imdb_dir_path"], param["limit_characters"], param['number_of_characters'], aug)
    if dump:
        with open(paths["preprocessed_path"], "wb") as f:
            pickle.dump({'x_train': x_train, 'x_test': x_test, 'y_train': y_train, 'y_test': y_test}, f, protocol=4)
    else:
        return x_train, x_test, y_train, y_test


# convert [1,2,3] -> ['abc']
def id_list_to_characters(id_list):
    sentence = ''
    for c in id_list:
        if c == 0:
            sentence += ' '
        elif c < 27:
            sentence += (chr(c + 96))
        elif c < 53:
            sentence += (chr(c + 65 - 27))
        elif c < 63:
            sentence += (chr(c + 48 - 53))
        elif c == 63:
            sentence += '?'
        elif c == 64:
            sentence += '!'
        elif c == 65:
            sentence += ','
        else:
            sentence += ' '
    return sentence
