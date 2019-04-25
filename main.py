from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
import pandas as pd
import numpy as np
import codecs
import re
from tqdm import tqdm


def common_preprocess_sentiment140(input_name):
    with codecs.open(input_name, "r", "UTF-8", "ignore") as file:
        df = pd.read_csv(file, header=None)
        labels, texts = [], []
        for l, t in zip(df[0], df[5]):
            re_text = re.sub(r'(https?://+\S*\s*|www.\S*\s*|#\S*\s*|@\S*\s*)', '', t).strip()
            if not re_text:
                continue
            labels.append(0 if l == 0 else 1)
            texts.append(re_text)
        texts = pd.Series(texts)
        labels = pd.Series(labels)
        dataset = pd.DataFrame({'text': texts, 'label': labels})
        return dataset


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


def conversion_rule_for_alphabet(sentence):
    id_list = []
    for c in sentence:
        if c.islower():
            id_list.append(ord(c) - 97)
        elif c.isupper():
            id_list.append(ord(c) - 39)
        elif c.isdecimal():
            id_list.append(int(c) + 52)
        elif c == '?':
            id_list.append(62)
        elif c == '!':
            id_list.append(63)
        elif c == ',':
            id_list.append(64)
        else:
            id_list.append(65)
    return id_list


def character_to_onehot(sentences, conversion_rule=conversion_rule_for_alphabet):
    characters_id_list = [[conversion_rule(c) for c in t] for t in tqdm(sentences)]
    sentence_onehot_matrix = np_utils.to_categorical(characters_id_list)
    print(sentence_onehot_matrix.shape)
    return sentence_onehot_matrix


def label_to_onehot(labels):
    pass

def train():
    model = Sequential()


def main():
    extracted_dataset = common_preprocess_sentiment140('dataset/sentiment140.csv')
    #[0-9a-zA-Z_!?,space] 66 characters
    restricted_dataset = character_restriction(extracted_dataset, restriction_rule=r'[^\w!?,\s]')
    text_matrices = character_to_onehot(restricted_dataset['text'], conversion_rule_for_alphabet)


if __name__ == '__main__':
    main()



