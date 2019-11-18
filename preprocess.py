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
from tqdm import tqdm

def common_preprocess_sentiment140(input_name):
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


def lemmatize(dataset):
    labels, texts = [], []
    lemma = nltk.stem.WordNetLemmatizer()
    for l, t in tqdm(zip(dataset["label"], dataset["text"])):
        words = nltk.word_tokenize(t)
        stem_text = ' '.join([lemma.lemmatize(word) for word in words])
        texts.append(stem_text)
        labels.append(l)
    texts = pd.Series(texts)
    labels = pd.Series(labels)
    return pd.DataFrame({'text': texts, 'label': labels})


def character_restriction(dataset, restriction_rule=r'[^\w!?,\s]'):
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
def conversion_rule_for_alphabet(sentence):
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


def texts_to_characters_id_lists(texts, limit_characters, conversion_rule=conversion_rule_for_alphabet):
    characters_id_lists = []
    for sentence in texts:
        id_list = conversion_rule_for_alphabet(sentence)
        id_list = id_list + [0 for i in range(limit_characters - len(id_list))]
        characters_id_lists.append(id_list[:limit_characters])

    return np.array(characters_id_lists)


def labels_to_onehot(labels):
    return np_utils.to_categorical(list(labels))


def transform_pos(nltk_pos):
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


def data_augmentation(dataset, sample_num=100000, sample_rate=None):
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
            pos = transform_pos(pos)
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


# ----------------------------------------------------------------------------------------------------------------------

# main method
def preprocess(file_name, limit_characters, number_of_characters, aug=False):
    extracted_dataset = common_preprocess_sentiment140(file_name)
    #[0-9a-zA-Z_!?,space] 66 characters
    lemma_dataset = lemmatize(extracted_dataset)
    if aug:
        lemma_dataset = data_augmentation(lemma_dataset, sample_rate=4.0)
    restricted_dataset = character_restriction(lemma_dataset, restriction_rule=r'[^\w!?,\s]')
    characters_id_lists = texts_to_characters_id_lists(restricted_dataset['text'], limit_characters)
    labels = labels_to_onehot(restricted_dataset['label'])
    return train_test_split(characters_id_lists, labels, test_size=0.2, random_state=183)


# preprocess and  dump
def preprocess_wrap(conf, dump=True, aug=False):
    paths, param = conf["paths"], conf["preprocessing_parameters"]
    x_train, x_test, y_train, y_test = preprocess(paths["dataset_path"], param["limit_characters"], param['number_of_characters'], aug)
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
