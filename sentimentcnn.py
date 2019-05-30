from keras.models import Model, load_model
from keras.layers import Input, Dense, Conv2D, Dropout, MaxPooling2D, Lambda, Embedding, Reshape
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
    embedding_dimension = 128
    filter_size = 256
    # layer
    inputs = Input(shape=(limit_characters, ))
    # embedding
    e1 = Embedding(input_dim=number_of_characters, output_dim=embedding_dimension,
                   embeddings_initializer='uniform', mask_zero=False)(inputs)
    e2 = Reshape(target_shape=(limit_characters, embedding_dimension, 1))(e1)
    # conv layer 1
    c1 = Conv2D(filters=filter_size, kernel_size=(7, embedding_dimension), padding='valid',
                activation='relu',
                data_format='channels_last')(e2)
    l1 = int(c1.shape[1])
    r1 = Reshape(target_shape=(l1, filter_size, 1))(c1)
    p1 = MaxPooling2D((2, 1), padding='valid')(r1)
    d1 = int(p1.shape[2])
    """
    # conv layer 2
    c2 = Conv2D(filters=filter_size, kernel_size=(7, d1), padding='valid',
                activation='relu',
                data_format='channels_last')(p1)
    l2 = int(c2.shape[1])
    r2 = Reshape(target_shape=(l2, filter_size, 1))(c2)
    p2 = MaxPooling2D((2, 1), padding='valid')(r2)
    d2 = int(p2.shape[2])
    # conv layer 3
    c3 = Conv2D(filters=filter_size, kernel_size=(3, d2), padding='valid',
                activation='relu',
                data_format='channels_last')(p2)
    l3 = int(c3.shape[1])
    r3 = Reshape(target_shape=(l3, filter_size, 1))(c3)
    # conv layer 4
    c4 = Conv2D(filters=filter_size, kernel_size=(3, filter_size), padding='valid',
                activation='relu',
                data_format='channels_last')(r3)
    l4 = int(c4.shape[1])
    r4 = Reshape(target_shape=(l4, filter_size, 1))(c4)
    # conv layer 5
    c5 = Conv2D(filters=filter_size, kernel_size=(3, filter_size), padding='valid',
                activation='relu',
                data_format='channels_last')(r4)
    l5 = int(c5.shape[1])
    r5 = Reshape(target_shape=(l5, filter_size, 1))(c5)
    """
    # conv layer 6
    c6 = Conv2D(filters=filter_size, kernel_size=(7, filter_size), padding='valid',
                activation='relu',
                data_format='channels_last')(p1)
    l6 = int(c6.shape[1])
    r6 = Reshape(target_shape=(l6, filter_size, 1))(c6)
    p6 = MaxPooling2D((l6, 1), padding='valid')(r6)
    d6 = int(p6.shape[2])
    # fully-connected
    f1 = Reshape(target_shape=(filter_size, ))(p6)
    f2 = Dense(1024, activation='relu')(f1)
    fd2 = Dropout(0.5)(f2)
    f3 = Dense(1024, activation='relu')(fd2)
    fd3 = Dropout(0.5)(f3)
    prediction = Dense(2, activation='softmax')(fd3)
    return Model(input=inputs, output=prediction)


def callbacks():
    # configure callback function
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    model_checkpoint = ModelCheckpoint('./model/fast_charcnn.h5', verbose=1, save_best_only=True)
    tensor_board = TensorBoard(log_dir='./log/sentiment', histogram_freq=0)
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
    batch_size = 4096
    model = load_model('./model/fast_charcnn.h5')
    x, y = x[x.shape[0] % batch_size:], y[y.shape[0] % batch_size:]
    print("Test Score: ", model.evaluate(x, y, batch_size=batch_size))
