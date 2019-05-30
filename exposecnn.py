from keras.models import Model, load_model
from keras.layers import Input, Dense, Conv2D, Dropout, MaxPooling2D, Lambda, Embedding, Reshape
from keras.layers import BatchNormalization, Concatenate
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import keras.backend as K
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
import numpy as np

n_folds = 1
epochs = 20


def configure(limit_characters, number_of_characters):
    # parameter
    embedding_dimension = 32
    filter_size = 256
    # input layer
    inputs = Input(shape=(limit_characters, ))
    # embedding layer
    x1 = Embedding(input_dim=number_of_characters, output_dim=embedding_dimension,
                   embeddings_initializer='uniform', mask_zero=False)(inputs)
    # convolution layer (parallel)
    x2 = Reshape(target_shape=(limit_characters, embedding_dimension, 1))(x1)
    conv = []
    for i in range(4):
        c = Conv2D(filters=filter_size, kernel_size=(i + 2, embedding_dimension), padding='valid',
                   activation='relu', data_format='channels_last')(x2)
        n = BatchNormalization()(c)
        p = MaxPooling2D((limit_characters-i-1, 1), padding='valid')(n)
        d = Dropout(0.5)(p)
        r = Reshape(target_shape=(filter_size, ))(d)
        conv.append(r)
    # concatenation layer
    x3 = Concatenate(axis=1)(conv)
    n3 = BatchNormalization()(x3)
    # fully-connected layer (1st)
    x4 = Dense(1024, activation='relu')(n3)
    n4 = BatchNormalization()(x4)
    d4 = Dropout(0.5)(n4)
    # fully-connected layer (2nd)
    x5 = Dense(1024, activation='relu')(d4)
    n5 = BatchNormalization()(x5)
    d5 = Dropout(0.5)(n5)
    # fully-connected layer (3rd)
    x6 = Dense(1024, activation='relu')(d5)
    n6 = BatchNormalization()(x6)
    d6 = Dropout(0.5)(n6)
    # output layer
    prediction = Dense(2, activation='softmax')(d6)
    return Model(input=inputs, output=prediction)


def callbacks():
    # configure callback function
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    model_checkpoint = ModelCheckpoint('./model/fast_exposecnn.h5', verbose=1, save_best_only=True)
    tensor_board = TensorBoard(log_dir='./log/expose3/', histogram_freq=0)
    return [early_stopping, model_checkpoint, tensor_board]


def train(x, y, limit_characters, number_of_characters):
    session = tf.Session('')
    ktf.set_session(session)
    for i in range(n_folds):
        print("Training on Fold: ", i + 1)
        result = fit_and_evaluate(x, y, limit_characters, number_of_characters)


def fit_and_evaluate(x, y, limit_characters, number_of_characters):
    # parameter
    batch_size = 128
    # generate model
    model = configure(limit_characters, number_of_characters)
    for layer in model.layers:
        layer.trainable = True
    model.compile(loss='binary_crossentropy',
                  optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])
    model.summary()

    x_t, x_val, y_t, y_val = train_test_split(x, y, test_size=0.1, random_state=0)
    x_t, x_val = x_t[x_t.shape[0] % batch_size:], x_val[x_val.shape[0] % batch_size:]
    y_t, y_val = y_t[y_t.shape[0] % batch_size:], y_val[y_val.shape[0] % batch_size:]

    result = model.fit(x_t, y_t, epochs=epochs, batch_size=batch_size, callbacks=callbacks(),
                       verbose=1, validation_data=(x_val, y_val))
    print("Val Score: ", model.evaluate(x_val, y_val, batch_size=batch_size))
    print("=======" * 12, end="\n\n\n")
    return result


def test(x, y):
    batch_size = 128
    model = load_model('./model/fast_exposecnn.h5')
    x, y = x[x.shape[0] % batch_size:], y[y.shape[0] % batch_size:]
    print("Test Score: ", model.evaluate(x, y, batch_size=batch_size))
