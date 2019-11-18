from keras.models import Model, load_model
from keras.layers import Input, Dense, Conv2D, Dropout, MaxPooling2D, AveragePooling2D
from keras.layers import Lambda, Embedding, Reshape, Activation, Flatten
from keras.layers import BatchNormalization
from keras.utils import np_utils
from layer import CharacterEmbeddingLayer
import keras.backend as K
import numpy as np


def simple(conf):
    # parameter
    model_param, pre_param = conf["model_parameters"], conf["preprocessing_parameters"]
    limit_characters = pre_param["limit_characters"]
    number_of_characters = pre_param["number_of_characters"]
    embedding_dimension = model_param["embedding_dimension"]
    filter_size = model_param["1st_layer"]["filter_size"]
    convolution_width = model_param["1st_layer"]["convolution_width"]
    dense_size = model_param["dense_size"]

    # layer
    inputs = Input(shape=(limit_characters, 1), dtype='int32')
    l1 = Lambda(lambda x: K.one_hot(x, num_classes=number_of_characters))(inputs)
    r1 = Reshape(target_shape=(1, limit_characters, number_of_characters))(l1)
    x1 = CharacterEmbeddingLayer(embedding_dimension, name='embedding')(r1)
    x2 = Reshape(target_shape=(limit_characters, embedding_dimension, 1))(x1)
    # conv
    x3 = Conv2D(filters=filter_size, kernel_size=(convolution_width, embedding_dimension), padding='valid',
                activation='relu', data_format='channels_last', name='conv')(x2)
    x4 = MaxPooling2D((limit_characters - convolution_width + 1, 1), padding='valid', name='pooling',
                      data_format='channels_last')(x3)
    x5 = Dropout(0.5)(x4)
    # fully-connected
    f1 = Flatten()(x5)
    f2 = Dense(dense_size, activation='relu', name='dense_1')(f1)
    fd2 = Dropout(0.5)(f2)
    f3 = Dense(dense_size, activation='relu', name='dense_2')(fd2)
    fd3 = Dropout(0.5)(f3)
    f4 = Dense(2, activation='linear', name='final')(fd3)
    prediction = Activation('softmax', name='preds')(f4)
    return Model(input=inputs, output=prediction)


def two_convolution(conf):
    # parameter
    model_param, pre_param = conf["model_parameters"], conf["preprocessing_parameters"]
    limit_characters = pre_param["limit_characters"]
    number_of_characters = pre_param["number_of_characters"]
    embedding_dimension = model_param["embedding_dimension"]
    dense_size = model_param["dense_size"]
    filter_size_1st = model_param["1st_layer"]["filter_size"]
    convolution_width_1st = model_param["1st_layer"]["convolution_width"]
    pooling_size_1st = model_param["1st_layer"]["pooling_size"]
    filter_size_2nd = model_param["2nd_layer"]["filter_size"]
    convolution_width_2nd = model_param["2nd_layer"]["convolution_width"]

    # layer
    inputs = Input(shape=(limit_characters, 1), dtype='int32')
    l1 = Lambda(lambda x: K.one_hot(x, num_classes=number_of_characters))(inputs)
    r1 = Reshape(target_shape=(1, limit_characters, number_of_characters))(l1)

    # embedding
    x1 = CharacterEmbeddingLayer(embedding_dimension, name='embedding')(r1)

    # 1st-conv
    c1 = Conv2D(filters=filter_size_1st, kernel_size=(convolution_width_1st, embedding_dimension), padding='valid',
                activation='relu', data_format='channels_first', name='conv')(x1)
    b1 = Dropout(0.5)(c1)
    r1 = Reshape(target_shape=(1, limit_characters - convolution_width_1st + 1, filter_size_1st))(b1)
    p1 = MaxPooling2D(pool_size=(1, pooling_size_1st), data_format='channels_first')(r1)

    # 2nd-conv (final conv)
    c2 = Conv2D(filters=filter_size_2nd, kernel_size=(convolution_width_2nd, filter_size_1st // pooling_size_1st),
                padding='valid', activation='relu', data_format='channels_first', name='conv2')(p1)
    b2 = Dropout(0.5)(c2)
    p2 = MaxPooling2D((limit_characters - convolution_width_1st - convolution_width_2nd + 2, 1),
                      padding='valid', name='pooling', data_format='channels_first')(b2)

    # fully-connected
    f1 = Flatten()(p2)
    f2 = Dense(dense_size, activation='relu', name='dense_1')(f1)
    fd2 = Dropout(0.5)(f2)
    f3 = Dense(dense_size, activation='relu', name='dense_2')(fd2)
    fd3 = Dropout(0.5)(f3)
    f4 = Dense(2, activation='linear', name='final')(fd3)

    # predict
    prediction = Activation('softmax', name='preds')(f4)
    return Model(input=inputs, output=prediction)
