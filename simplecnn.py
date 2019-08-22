from keras.models import Model, load_model
from keras.layers import Input, Dense, Conv2D, Dropout, MaxPooling2D, Lambda, Embedding, Reshape, Activation, Flatten
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.utils import np_utils
import keras.backend as K
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
import numpy as np
from layer import CharacterEmbeddingLayer


def configure(limit_characters, number_of_characters, param):
    # parameter
    embedding_dimension = param["embedding_dimension"]
    filter_size = param["filter_size"]
    convolution_width = param["convolution_width"]
    dense_size = param["dense_size"]

    # layer
    inputs = Input(shape=(1, limit_characters, number_of_characters))
    # embedding
    #x1 = Embedding(input_dim=number_of_characters, output_dim=embedding_dimension,
    #              embeddings_initializer='uniform', mask_zero=False, name='embedding')(inputs)
    x1 = CharacterEmbeddingLayer(embedding_dimension, name='embedding')(inputs)
    #x2 = Reshape(target_shape=(limit_characters, embedding_dimension, 1))(x1)
    # conv
    x3 = Conv2D(filters=filter_size, kernel_size=(convolution_width, embedding_dimension), padding='valid',
                activation='relu', data_format='channels_first', name='conv')(x1)
    x4 = MaxPooling2D((limit_characters - convolution_width + 1, 1), padding='valid', name='pooling',
                      data_format='channels_first')(x3)
    x5 = Dropout(0.5)(x4)
    # fully-connected
    f1 = Flatten()(x5)
    f2 = Dense(dense_size, activation='relu', name='dense_1')(f1)
    fd2 = Dropout(0.5)(f2)
    f3 = Dense(dense_size, activation='relu', name='dense_2')(fd2)
    fd3 = Dropout(0.5)(f3)
    f4 = Dense(2, activation=None, name='dense_3')(fd3)
    prediction = Activation('softmax', name='preds')(f4)
    return Model(input=inputs, output=prediction)


def callbacks(paths):
    # configure callback function
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    model_checkpoint = ModelCheckpoint(paths["model_path"], verbose=1, save_best_only=True)
    tensor_board = TensorBoard(log_dir=paths["log_dir_path"], histogram_freq=0)
    return [early_stopping, model_checkpoint, tensor_board]


def train(x, y, conf):
    old_session = ktf.get_session()
    session = tf.Session('')
    ktf.set_session(session)
    ktf.set_learning_phase(1)
    for i in range(conf["train_parameters"]["n_folds"]):
        print("Training on Fold: ", i + 1)
        result = fit_and_evaluate(x, y, conf)
    ktf.set_session(old_session)


def fit_and_evaluate(x, y, conf):
    # parameter
    batch_size = conf["train_parameters"]["batch_size"]
    epochs = conf["train_parameters"]["epochs"]
    pre_param = conf["preprocessing_parameters"]
    limit_characters, number_of_characters = pre_param["limit_characters"], pre_param["number_of_characters"]
    print(x.shape)
    # generate model
    model = configure(limit_characters, number_of_characters, conf["model_parameters"])
    for layer in model.layers:
        layer.trainable = True
    model.compile(loss='binary_crossentropy',
                  optimizer=SGD(lr=0.01, decay=1e-4, momentum=0.9),
                  metrics=['accuracy'])
    model.summary()

    x = x.reshape(x.shape[0], 1, x.shape[1])
    x_t, x_val, y_t, y_val = train_test_split(x, y, test_size=0.1, random_state=0)
    x_t, x_val = x_t[x_t.shape[0] % batch_size:], x_val[x_val.shape[0] % batch_size:]
    x_t, x_val = np_utils.to_categorical(x_t), np_utils.to_categorical(x_val)
    y_t, y_val = y_t[y_t.shape[0] % batch_size:], y_val[y_val.shape[0] % batch_size:]

    result = model.fit(x_t, y_t, epochs=epochs, batch_size=batch_size, callbacks=callbacks(conf["paths"]),
                       verbose=1, validation_data=(x_val, y_val))
    # print("Val Score: ", model.evaluate(x_val, y_val, batch_size=batch_size))
    print(K.gradients(model.output, model.input))
    model.save(conf["paths"]["model_path"])
    print("=======" * 12, end="\n\n\n")
    return result


def test(x, y, conf):
    batch_size = conf["train_parameters"]["batch_size"]
    model = load_model(conf["paths"]["model_path"])
    x = x.reshape(x.shape[0], 1, x.shape[1])
    x, y = x[x.shape[0] % batch_size:], y[y.shape[0] % batch_size:]
    print("Test Score: ", model.evaluate(x, y, batch_size=batch_size))
