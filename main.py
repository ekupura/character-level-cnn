from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Dropout, MaxPooling2D, Lambda, Embedding
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import keras.backend as K
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import codecs
import re
from tqdm import tqdm

# parameter
limit_characters = 150
number_of_characters = 67
batch_size = 128
embedding_dimension = 300
filter_size = 1024


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


def charcnn():
    inputs = Input(shape=(limit_characters, ))
    x1 = Embedding(input_dim=number_of_characters, output_dim=embedding_dimension,
                   embeddings_initializer='uniform', mask_zero=False)(inputs)
    x2 = Lambda(lambda a: K.reshape(a, (batch_size, limit_characters, embedding_dimension, 1)))(x1)
    x3 = Conv2D(filters=filter_size, kernel_size=(3, embedding_dimension), padding='valid',
                activation='relu', input_shape=(batch_size, limit_characters, embedding_dimension, 1),
                data_format='channels_last')(x2)
    x4 = MaxPooling2D((limit_characters-2, 1), padding='valid')(x3)
    x5 = Dropout(0.5)(x4)
    x6 = Lambda(lambda b: K.reshape(b, (batch_size, filter_size, )))(x5)
    x7 = Dense(512, activation='relu')(x6)
    x8 = Dense(512, activation='relu')(x7)
    prediction = Dense(2, activation='softmax')(x8)
    return Model(input=inputs, output=prediction)


def train(x, y):
    # split dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    # x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    # x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    # generate model structure
    model = charcnn()
    for layer in model.layers:
        layer.trainable = True
    model.compile(loss='binary_crossentropy',
                  optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])
    model.summary()

    # configure callback function
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    model_checkpoint = ModelCheckpoint('./model/fast_charcnn.h5', verbose=1, save_best_only=True)

    n_folds = 3
    epochs = 10
    model_history = []

    for i in range(n_folds):
        print("Training on Fold: ", i + 1)
        x_t, x_val, y_t, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=0)
        x_t, x_val = x_t[x_t.shape[0] % batch_size:], x_val[x_val.shape[0] % batch_size:]
        y_t, y_val = y_t[y_t.shape[0] % batch_size:], y_val[y_val.shape[0] % batch_size:]
        results = model.fit(x_t, y_t, epochs=epochs, batch_size=batch_size, callbacks=[early_stopping, model_checkpoint]
                            , verbose=1, validation_data=(x_val, y_val))
        print("Val Score: ", model.evaluate(x_val, y_val, batch_size=batch_size))
        print("=======" * 12, end="\n\n\n")
        model_history.append(results)


def main():
    extracted_dataset = common_preprocess_sentiment140('dataset/sentiment140.csv')
    #[0-9a-zA-Z_!?,space] 66 characters
    restricted_dataset = character_restriction(extracted_dataset, restriction_rule=r'[^\w!?,\s]')
    characters_id_lists = texts_to_characters_id_lists(restricted_dataset['text'], conversion_rule_for_alphabet)
    labels = labels_to_onehot(restricted_dataset['label'])
    print("Start training")
    train(characters_id_lists, labels)


if __name__ == '__main__':
    main()



