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


def seven_convolution(conf):
    # parameter
    model_param, pre_param = conf["model_parameters"], conf["preprocessing_parameters"]
    limit_characters = pre_param["limit_characters"]
    number_of_characters = pre_param["number_of_characters"]
    embedding_dimension = model_param["embedding_dimension"]
    dense_size = model_param["dense_size"]

    # layer
    inputs = Input(shape=(limit_characters, 1), dtype='int32')
    l1 = Lambda(lambda x: K.one_hot(x, num_classes=number_of_characters))(inputs)
    r1 = Reshape(target_shape=(1, limit_characters, number_of_characters))(l1)

    # embedding
    x1 = CharacterEmbeddingLayer(embedding_dimension, name='embedding')(r1)

    # 1st-conv
    filter_size_1st = model_param["1st_layer"]["filter_size"]
    convolution_width_1st = model_param["1st_layer"]["convolution_width"]
    pooling_size_1st = model_param["1st_layer"]["pooling_size"]
    after_width = limit_characters - convolution_width_1st + 1

    c1 = Conv2D(filters=filter_size_1st, kernel_size=(convolution_width_1st, embedding_dimension), padding='valid',
                activation='relu', data_format='channels_first', name='conv')(x1)
    b1 = Dropout(0.5)(c1)
    r1 = Reshape(target_shape=(1, after_width, filter_size_1st))(b1)
    p1 = MaxPooling2D(pool_size=(pooling_size_1st, pooling_size_1st), data_format='channels_first')(r1)
    next_conv_dim = filter_size_1st // pooling_size_1st
    after_width = after_width // pooling_size_1st

    # 2nd-conv
    filter_size_2nd = model_param["2nd_layer"]["filter_size"]
    convolution_width_2nd = model_param["2nd_layer"]["convolution_width"]
    pooling_size_2nd = model_param["2nd_layer"]["pooling_size"]
    after_width = after_width - convolution_width_2nd + 1

    c2 = Conv2D(filters=filter_size_2nd, kernel_size=(convolution_width_2nd, next_conv_dim),
                padding='valid', activation='relu', data_format='channels_first', name='conv2')(p1)
    b2 = Dropout(0.5)(c2)
    r2 = Reshape(target_shape=(1, after_width, filter_size_2nd))(b2)
    p2 = MaxPooling2D(pool_size=(pooling_size_2nd, pooling_size_2nd), data_format='channels_first')(r2)
    next_conv_dim = filter_size_2nd // pooling_size_2nd
    after_width = after_width // pooling_size_2nd

    # 3rd-conv
    filter_size_3rd = model_param["3rd_layer"]["filter_size"]
    convolution_width_3rd = model_param["3rd_layer"]["convolution_width"]
    pooling_size_3rd = model_param["3rd_layer"]["pooling_size"]
    after_width = after_width - convolution_width_3rd + 1

    c3 = Conv2D(filters=filter_size_3rd, kernel_size=(convolution_width_3rd, next_conv_dim),
                padding='valid', activation='relu', data_format='channels_first', name='conv3')(p2)
    b3 = Dropout(0.5)(c3)
    r3 = Reshape(target_shape=(1, after_width, filter_size_3rd))(b3)
    p3 = MaxPooling2D(pool_size=(pooling_size_3rd, pooling_size_3rd), data_format='channels_first')(r3)
    next_conv_dim = filter_size_3rd // pooling_size_3rd
    after_width = after_width // pooling_size_3rd

    # 4th-conv
    filter_size_4th = model_param["4th_layer"]["filter_size"]
    convolution_width_4th = model_param["4th_layer"]["convolution_width"]
    pooling_size_4th = model_param["4th_layer"]["pooling_size"]
    after_width = after_width - convolution_width_4th + 1

    c4 = Conv2D(filters=filter_size_4th, kernel_size=(convolution_width_4th, next_conv_dim),
                padding='valid', activation='relu', data_format='channels_first', name='conv4')(p3)
    b4 = Dropout(0.5)(c4)
    r4 = Reshape(target_shape=(1, after_width, filter_size_4th))(b4)
    p4 = MaxPooling2D(pool_size=(pooling_size_4th, pooling_size_4th), data_format='channels_first')(r4)
    next_conv_dim = filter_size_4th // pooling_size_4th
    after_width = after_width // pooling_size_4th

    # 5th-conv
    filter_size_5th = model_param["5th_layer"]["filter_size"]
    convolution_width_5th = model_param["5th_layer"]["convolution_width"]
    pooling_size_5th = model_param["5th_layer"]["pooling_size"]
    after_width = after_width - convolution_width_5th + 1

    c5 = Conv2D(filters=filter_size_5th, kernel_size=(convolution_width_5th, next_conv_dim),
                padding='valid', activation='relu', data_format='channels_first', name='conv5')(p4)
    b5 = Dropout(0.5)(c5)
    r5 = Reshape(target_shape=(1, after_width, filter_size_5th))(b5)
    p5 = MaxPooling2D(pool_size=(pooling_size_5th, pooling_size_5th), data_format='channels_first')(r5)
    next_conv_dim = filter_size_5th // pooling_size_5th
    after_width = after_width // pooling_size_5th

    # 6th-conv
    filter_size_6th = model_param["6th_layer"]["filter_size"]
    convolution_width_6th = model_param["6th_layer"]["convolution_width"]
    pooling_size_6th = model_param["6th_layer"]["pooling_size"]
    after_width = after_width - convolution_width_6th + 1

    c6 = Conv2D(filters=filter_size_6th, kernel_size=(convolution_width_6th, next_conv_dim),
                padding='valid', activation='relu', data_format='channels_first', name='conv6')(p5)
    b6 = Dropout(0.5)(c6)
    r6 = Reshape(target_shape=(1, after_width, filter_size_6th))(b6)
    p6 = MaxPooling2D(pool_size=(pooling_size_6th, pooling_size_6th), data_format='channels_first')(r6)
    next_conv_dim = filter_size_6th // pooling_size_6th
    after_width = after_width // pooling_size_6th

    # 7th-conv (final)
    filter_size_7th = model_param["7th_layer"]["filter_size"]
    convolution_width_7th = model_param["7th_layer"]["convolution_width"]
    pooling_size_7th = model_param["7th_layer"]["pooling_size"]
    after_width = after_width - convolution_width_7th + 1

    c7 = Conv2D(filters=filter_size_7th, kernel_size=(convolution_width_7th, next_conv_dim),
                padding='valid', activation='relu', data_format='channels_first', name='conv7')(p6)
    b7 = Dropout(0.5)(c7)

    # fully-connected
    all = MaxPooling2D((after_width, 1), padding='valid', name='pooling', data_format='channels_first')(b7)
    f1 = Flatten()(all)
    f2 = Dense(dense_size, activation='relu', name='dense_1')(f1)
    fd2 = Dropout(0.5)(f2)
    f3 = Dense(dense_size, activation='relu', name='dense_2')(fd2)
    fd3 = Dropout(0.5)(f3)
    f4 = Dense(2, activation='linear', name='final')(fd3)

    # predict
    prediction = Activation('softmax', name='preds')(f4)
    return Model(input=inputs, output=prediction)
