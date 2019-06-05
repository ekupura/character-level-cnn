from keras.utils import np_utils
import pandas as pd
import numpy as np
import codecs
import re
import simplecnn
import exposecnn
import sentimentcnn
from sklearn.model_selection import train_test_split
import yaml

# global parameter
limit_characters = 150
number_of_characters = 67


def common_preprocess_sentiment140(input_name):
    with codecs.open(input_name, "r", "UTF-8", "ignore") as file:
        df = pd.read_csv(file, header=None)
        labels, texts = [], []
        for l, t in zip(df[0], df[5]):
            re_text = re.sub(r'(https?://+\S*\s*|www.\S*\s*|#\S*\s*|@\S*\s*|&\S*\s*)', '', t).strip()
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
            id_list.append(65)
    return id_list


def texts_to_characters_id_lists(texts, conversion_rule=conversion_rule_for_alphabet):
    characters_id_lists = []
    for sentence in texts:
        id_list = conversion_rule_for_alphabet(sentence)
        id_list = id_list + [0 for i in range(limit_characters - len(id_list))]
        characters_id_lists.append(id_list)

    return np.array(characters_id_lists)


def labels_to_onehot(labels):
    return np_utils.to_categorical(labels)


def main():
    extracted_dataset = common_preprocess_sentiment140('dataset/sentiment140.csv')
    #[0-9a-zA-Z_!?,space] 66 characters
    restricted_dataset = character_restriction(extracted_dataset, restriction_rule=r'[^\w!?,\s]')
    characters_id_lists = texts_to_characters_id_lists(restricted_dataset['text'], conversion_rule_for_alphabet)
    labels = labels_to_onehot(restricted_dataset['label'])
    x_train, x_test, y_train, y_test = train_test_split(characters_id_lists, labels,
                                                        test_size=0.2, random_state=183)
    print("Load configuration")
    f = open("./experiment/simple_9/configuration.yml", 'r')
    configuration = yaml.load(f)
    f.close()

    print("Start training")
    simplecnn.train(x_train, y_train, limit_characters, number_of_characters, configuration)
    simplecnn.test(x_test, y_test, configuration)


if __name__ == '__main__':
    main()
