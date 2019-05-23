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
epochs = 1


def configure(limit_characters, number_of_characters):
    # parameter
    embedding_dimension = 50
    filter_size = 256
    # layer
    inputs = Input(shape=(limit_characters, ))
    x1 = Embedding(input_dim=number_of_characters, output_dim=embedding_dimension,
                   embeddings_initializer='uniform', mask_zero=False)(inputs)
    x2 = Reshape(target_shape=(limit_characters, embedding_dimension, 1))(x1)
    x3 = Conv2D(filters=filter_size, kernel_size=(3, embedding_dimension), padding='valid',
                activation='relu',
                data_format='channels_last')(x2)
    x4 = MaxPooling2D((limit_characters-2, 1), padding='valid')(x3)
    x5 = Dropout(0.5)(x4)
    x6 = Reshape(target_shape=(filter_size, ))(x5)
    x7 = Dense(512, activation='relu')(x6)
    x8 = Dense(512, activation='relu')(x7)
    prediction = Dense(2, activation='softmax')(x8)
    return Model(input=inputs, output=prediction)


def callbacks():
    # configure callback function
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    model_checkpoint = ModelCheckpoint('./model/fast_charcnn.h5', verbose=1, save_best_only=True)
    tensor_board = TensorBoard(log_dir='./log/', histogram_freq=0)
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
    model = load_model('./model/fast_charcnn.h5')
    x, y = x[x.shape[0] % batch_size:], y[y.shape[0] % batch_size:]
    print("Test Score: ", model.evaluate(x, y, batch_size=batch_size))
