from keras.models import Model, load_model
from keras.layers import Input, Dense, Conv2D, Dropout, MaxPooling2D, Conv2DTranspose, MaxPooling3D, UpSampling2D
from keras.layers import Lambda, Embedding, Reshape, Activation, Flatten, Conv1D, MaxPooling1D, LSTM
from keras.layers import BatchNormalization, Concatenate, Bidirectional
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


def numerous_convolution(conf):
    # parameter
    model_param, pre_param = conf["model_parameters"], conf["preprocessing_parameters"]
    limit_characters = pre_param["limit_characters"]
    number_of_characters = pre_param["number_of_characters"]
    embedding_dimension = model_param["embedding_dimension"]
    dense_size = model_param["dense_size"]
    dropout_late = 0.50
    print(dropout_late)

    # layer
    inputs = Input(shape=(limit_characters, 1), dtype='int32')
    l1 = Lambda(lambda x: K.one_hot(x, num_classes=number_of_characters))(inputs)
    r1 = Reshape(target_shape=(1, limit_characters, number_of_characters))(l1)

    # embedding
    last = CharacterEmbeddingLayer(embedding_dimension, name='embedding')(r1)

    # 1st-conv
    filter_size_1st = model_param["1st_layer"]["filter_size"]
    convolution_width_1st = model_param["1st_layer"]["convolution_width"]
    pooling_size_1st = model_param["1st_layer"]["pooling_size"]
    after_width = limit_characters - convolution_width_1st + 1

    x1 = Conv2D(filters=filter_size_1st, kernel_size=(convolution_width_1st, embedding_dimension), padding='valid',
                activation='relu', data_format='channels_first', name='conv')(last)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(dropout_late)(x1)
    x1 = Reshape(target_shape=(1, after_width, filter_size_1st))(x1)
    last = MaxPooling2D(pool_size=(pooling_size_1st, pooling_size_1st), data_format='channels_first')(x1)
    next_conv_dim = filter_size_1st // pooling_size_1st
    after_width = after_width // pooling_size_1st

    """
    # 2nd-conv
    if "2nd_layer" in model_param:
        filter_size_2nd = model_param["2nd_layer"]["filter_size"]
        convolution_width_2nd = model_param["2nd_layer"]["convolution_width"]
        pooling_size_2nd = model_param["2nd_layer"]["pooling_size"]
        after_width = after_width - convolution_width_2nd + 1

        x2 = Conv2D(filters=filter_size_2nd, kernel_size=(convolution_width_2nd, next_conv_dim),
                    padding='valid', activation='relu', data_format='channels_first', name='conv2')(last)
        x2 = BatchNormalization()(x2)
        x2 = Dropout(dropout_late)(x2)
        x2 = Reshape(target_shape=(1, after_width, filter_size_2nd))(x2)
        last = MaxPooling2D(pool_size=(pooling_size_2nd, pooling_size_2nd), data_format='channels_first')(x2)
        next_conv_dim = filter_size_2nd // pooling_size_2nd
        after_width = after_width // pooling_size_2nd

    # 3rd-conv
    if "3rd_layer" in model_param:
        filter_size_3rd = model_param["3rd_layer"]["filter_size"]
        convolution_width_3rd = model_param["3rd_layer"]["convolution_width"]
        pooling_size_3rd = model_param["3rd_layer"]["pooling_size"]
        after_width = after_width - convolution_width_3rd + 1

        x3 = Conv2D(filters=filter_size_3rd, kernel_size=(convolution_width_3rd, next_conv_dim),
                    padding='valid', activation='relu', data_format='channels_first', name='conv3')(last)
        x3 = BatchNormalization()(x3)
        x3 = Dropout(dropout_late)(x3)
        x3 = Reshape(target_shape=(1, after_width, filter_size_3rd))(x3)
        last = MaxPooling2D(pool_size=(pooling_size_3rd, 2), data_format='channels_first')(x3)
        #next_conv_dim = filter_size_3rd // pooling_size_3rd
        next_conv_dim = filter_size_3rd // 2
        after_width = after_width // pooling_size_3rd

    # 4th-conv
    if "4th_layer" in model_param:
        filter_size_4th = model_param["4th_layer"]["filter_size"]
        convolution_width_4th = model_param["4th_layer"]["convolution_width"]
        pooling_size_4th = model_param["4th_layer"]["pooling_size"]
        after_width = after_width - convolution_width_4th + 1

        x4 = Conv2D(filters=filter_size_4th, kernel_size=(convolution_width_4th, next_conv_dim),
                    padding='valid', activation='relu', data_format='channels_first', name='conv4')(last)
        x4 = BatchNormalization()(x4)
        x4 = Dropout(dropout_late)(x4)
        x4 = Reshape(target_shape=(1, after_width, filter_size_4th))(x4)
        last = MaxPooling2D(pool_size=(pooling_size_4th, pooling_size_4th), data_format='channels_first')(x4)
        next_conv_dim = filter_size_4th // pooling_size_4th
        after_width = after_width // pooling_size_4th

    # 5th-conv
    if "5th_layer" in model_param:
        filter_size_5th = model_param["5th_layer"]["filter_size"]
        convolution_width_5th = model_param["5th_layer"]["convolution_width"]
        pooling_size_5th = model_param["5th_layer"]["pooling_size"]
        after_width = after_width - convolution_width_5th + 1

        x5 = Conv2D(filters=filter_size_5th, kernel_size=(convolution_width_5th, next_conv_dim),
                    padding='valid', activation='relu', data_format='channels_first', name='conv5')(last)
        x5 = BatchNormalization()(x5)
        x5 = Dropout(dropout_late)(x5)
        x5 = Reshape(target_shape=(1, after_width, filter_size_5th))(x5)
        last = MaxPooling2D(pool_size=(pooling_size_5th, pooling_size_5th), data_format='channels_first')(x5)
        next_conv_dim = filter_size_5th // pooling_size_5th
        after_width = after_width // pooling_size_5th

    # 6th-conv
    if "6th_layer" in model_param:
        filter_size_6th = model_param["6th_layer"]["filter_size"]
        convolution_width_6th = model_param["6th_layer"]["convolution_width"]
        pooling_size_6th = model_param["6th_layer"]["pooling_size"]
        after_width = after_width - convolution_width_6th + 1

        x6 = Conv2D(filters=filter_size_6th, kernel_size=(convolution_width_6th, next_conv_dim),
                    padding='valid', activation='relu', data_format='channels_first', name='conv6')(last)
        x6 = BatchNormalization()(x6)
        x6 = Dropout(dropout_late)(x6)
        x6 = Reshape(target_shape=(1, after_width, filter_size_6th))(x6)
        last = MaxPooling2D(pool_size=(pooling_size_6th, pooling_size_6th), data_format='channels_first')(x6)
        next_conv_dim = filter_size_6th // pooling_size_6th
        after_width = after_width // pooling_size_6th

    """
    # 7th-conv (final)
    filter_size_final = model_param["final_layer"]["filter_size"]
    convolution_width_final = model_param["final_layer"]["convolution_width"]
    pooling_size_final = model_param["final_layer"]["pooling_size"]
    after_width = after_width - convolution_width_final + 1

    x7 = Conv2D(filters=filter_size_final, kernel_size=(convolution_width_final, next_conv_dim),
                padding='valid', activation='relu', data_format='channels_first', name='conv7')(last)
    x7 = BatchNormalization()(x7)
    x7 = Dropout(dropout_late)(x7)
    all = MaxPooling2D((after_width, 1), padding='valid', name='pooling', data_format='channels_first')(x7)

    # fully-connected
    f1 = Flatten()(all)
    f2 = Dense(dense_size, activation='relu', name='dense_1')(f1)
    fd2 = Dropout(0.5)(f2)
    f3 = Dense(dense_size, activation='relu', name='dense_2')(fd2)
    fd3 = Dropout(0.5)(f3)
    f4 = Dense(2, activation='linear', name='final')(fd3)

    # predict
    prediction = Activation('softmax', name='preds')(f4)
    return Model(input=inputs, output=prediction)


def autoencoder(conf):
    model_param, pre_param = conf["model_parameters"], conf["preprocessing_parameters"]
    limit_characters = pre_param["limit_characters"]
    number_of_characters = pre_param["number_of_characters"]
    dropout_rate = 0.5

    # input layer
    inputs = Input(shape=(limit_characters, 1), dtype='int32')
    l1 = Lambda(lambda x: K.one_hot(x, num_classes=number_of_characters))(inputs)
    emb = Reshape(target_shape=(number_of_characters, limit_characters))(l1)

    # 1st-conv
    x = Conv1D(filters=160, kernel_size=5, padding='same',
                activation='relu', data_format='channels_first', name='conv1d')(emb)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    chara = Reshape(target_shape=(1, 160, 160))(x)

    # 2nd-conv
    x = Conv2D(filters=64, kernel_size=(2, 2),
                padding='same', activation='relu', data_format='channels_first')(chara)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = MaxPooling2D(pool_size=(2, 2), data_format='channels_first', padding='same')(x)

    # 3nd-conv
    x = Conv2D(filters=32, kernel_size=(2, 2),
                padding='same', activation='relu', data_format='channels_first')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = MaxPooling2D(pool_size=(2, 2), data_format='channels_first', padding='same')(x)

    # concentlate
    x = Conv2D(filters=16, kernel_size=(2, 2),
                padding='same', activation='relu', data_format='channels_first', name='encode')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Reshape(target_shape=(1, 16, 40, 40))(x)
    x = MaxPooling3D(pool_size=(16, 1, 1), data_format='channels_first', padding='same')(x)
    encoded = Reshape(target_shape=(1, 40, 40))(x)

    # 5th-conv (decoder)
    x = Conv2D(filters=32, kernel_size=(2, 2),
                padding='same', activation='relu', data_format='channels_first')(encoded)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = UpSampling2D((2, 2), data_format='channels_first')(x)

    # 5th-conv (decoder)
    x = Conv2D(filters=64, kernel_size=(2, 2),
               padding='same', activation='relu', data_format='channels_first')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = UpSampling2D((2, 2), data_format='channels_first')(x)

    # final-conv
    x = Conv2D(filters=1, kernel_size=(2, 2),
                padding='same', activation='sigmoid', data_format='channels_first', name='decode')(x)
    x = BatchNormalization()(x)
    decoded = Dropout(dropout_rate)(x)

    autoencoder = Model(input=inputs, output=decoded)
    encoder = Model(input=inputs, output=encoded)

    final = Lambda(lambda x: K.square(x[0] - x[1]))([chara, decoded])
    training = Model(input=inputs, output=final)

    return training, autoencoder, encoder


def autoencoder1d(conf):
    model_param, pre_param = conf["model_parameters"], conf["preprocessing_parameters"]
    limit_characters = pre_param["limit_characters"]
    number_of_characters = pre_param["number_of_characters"]
    dropout_rate = 0.25

    # input layer
    inputs = Input(shape=(limit_characters, 1), dtype='int32')
    l1 = Lambda(lambda x: K.one_hot(x, num_classes=number_of_characters))(inputs)
    emb = Reshape(target_shape=(number_of_characters, limit_characters))(l1)

    # 1st-conv
    x = Conv1D(filters=160, kernel_size=5, padding='same',
               activation='relu', data_format='channels_first', name='conv1d')(emb)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    chara = Reshape(target_shape=(160, 160))(x)

    # 2nd-conv
    x = Conv1D(filters=160, kernel_size=3,
               padding='same', activation='relu', data_format='channels_first')(chara)
    x = Reshape(target_shape=(1, 160, 160))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = MaxPooling2D(pool_size=(2, 2), data_format='channels_first', padding='same')(x)
    x = Reshape(target_shape=(80, 80))(x)

    # concentlate
    x = Conv1D(filters=80, kernel_size=3,
               padding='same', activation='relu', data_format='channels_first', name='encode')(x)
    x = Reshape(target_shape=(80, 80))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    encoded = Reshape(target_shape=(1, 80, 80))(x)

    # 5th-conv (decoder)
    x = Conv1D(filters=160, kernel_size=3,
               padding='same', activation='relu', data_format='channels_first')(x)
    x = Reshape(target_shape=(1, 160, 80))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = UpSampling2D((1, 2), data_format='channels_first')(x)
    x = Reshape(target_shape=(160, 160))(x)

    # final-conv
    x = Conv1D(filters=160, kernel_size=3,
               padding='same', activation='sigmoid', data_format='channels_first', name='decode')(x)
    x = Reshape(target_shape=(1, 160, 160))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    decoded = Reshape(target_shape=(160, 160))(x)

    autoencoder = Model(input=inputs, output=decoded)
    encoder = Model(input=inputs, output=encoded)

    final = Lambda(lambda x: K.square(x[0] - x[1]))([chara, decoded])
    training = Model(input=inputs, output=final)

    return training, autoencoder, encoder

def classification_cnn(conf):
    dropout_rate = 0.5

    # input layer
    inputs = Input(shape=(1, 80, 80), dtype='float32')

    # block-1
    x = Conv2D(filters=16, kernel_size=(2, 2),
               padding='same', activation='relu', data_format='channels_first')(inputs)
    x = Conv2D(filters=16, kernel_size=(2, 2),
               padding='same', activation='relu', data_format='channels_first')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(4, 4), data_format='channels_first', padding='same')(x)
    x = Dropout(dropout_rate)(x)

    # block-2
    x = Conv2D(filters=32, kernel_size=(2, 2),
               padding='same', activation='relu', data_format='channels_first')(x)
    x = Conv2D(filters=32, kernel_size=(2, 2),
               padding='same', activation='relu', data_format='channels_first')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(4, 4), data_format='channels_first', padding='same')(x)
    x = Dropout(dropout_rate)(x)

    """
    # block-3
    x = Conv2D(filters=256, kernel_size=(2, 2),
               padding='same', activation='relu', data_format='channels_first')(x)
    x = Conv2D(filters=256, kernel_size=(2, 2),
               padding='same', activation='relu', data_format='channels_first')(x)
    x = Conv2D(filters=256, kernel_size=(2, 2),
               padding='same', activation='relu', data_format='channels_first')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), data_format='channels_first', padding='same')(x)
    x = Dropout(dropout_rate)(x)

    # block-4
    x = Conv2D(filters=512, kernel_size=(2, 2),
               padding='same', activation='relu', data_format='channels_first')(x)
    x = Conv2D(filters=512, kernel_size=(2, 2),
               padding='same', activation='relu', data_format='channels_first')(x)
    x = Conv2D(filters=512, kernel_size=(2, 2),
               padding='same', activation='relu', data_format='channels_first')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), data_format='channels_first', padding='same')(x)
    x = Dropout(dropout_rate)(x)

    # block-5
    x = Conv2D(filters=512, kernel_size=(2, 2),
               padding='same', activation='relu', data_format='channels_first')(x)
    x = Conv2D(filters=512, kernel_size=(2, 2),
               padding='same', activation='relu', data_format='channels_first')(x)
    x = Conv2D(filters=512, kernel_size=(2, 2),
               padding='same', activation='relu', data_format='channels_first')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2), data_format='channels_first', padding='same')(x)
    x = Dropout(dropout_rate)(x)
    """

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    prediction = Dense(1, activation='sigmoid')(x)

    return Model(input=inputs, output=prediction)

def classification_dense(conf):
    dropout_rate = 0.1

    # input layer
    inputs = Input(shape=(1, 40, 40), dtype='float32')
    x = Flatten()(inputs)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    prediction = Dense(2, activation='softmax')(x)

    return Model(input=inputs, output=prediction)

def character_level_cnn(conf):

    model_param, pre_param = conf["model_parameters"], conf["preprocessing_parameters"]
    limit_characters = pre_param["limit_characters"]
    number_of_characters = pre_param["number_of_characters"]
    dropout_rate = 0.25

    # input layer
    inputs = Input(shape=(limit_characters, 1), dtype='int32')
    l1 = Lambda(lambda x: K.one_hot(x, num_classes=number_of_characters))(inputs)
    emb = Reshape(target_shape=(number_of_characters, limit_characters))(l1)

    # 1st-conv
    x = Conv1D(filters=32, kernel_size=5, padding='same',
               activation='relu', data_format='channels_first', name='embedding')(emb)
    x = Reshape(target_shape=(1, 32, 160))(x)
    x = MaxPooling2D(pool_size=2, data_format='channels_first', padding='valid')(x)
    x = Reshape(target_shape=(16, 80))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    # 2nd-conv
    x = Conv1D(filters=64, kernel_size=3,
               padding='same', activation='relu', data_format='channels_first')(x)
    x = Reshape(target_shape=(1, 64, 80))(x)
    x = MaxPooling2D(pool_size=2, data_format='channels_first', padding='valid')(x)
    x = Reshape(target_shape=(32, 40))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    """
    # 2nd-conv
    x = Conv1D(filters=128, kernel_size=3,
               padding='same', activation='relu', data_format='channels_first')(x)
    x = Reshape(target_shape=(1, 128, 40))(x)
    x = MaxPooling2D(pool_size=2, data_format='channels_first', padding='valid')(x)
    x = Reshape(target_shape=(64, 20))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    # 2nd-conv
    x = Conv1D(filters=256, kernel_size=3,
               padding='same', activation='relu', data_format='channels_first')(x)
    x = Reshape(target_shape=(1, 256, 20))(x)
    x = MaxPooling2D(pool_size=2, data_format='channels_first', padding='valid')(x)
    x = Reshape(target_shape=(128, 10))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    """

    # concentlate
    x = Reshape(target_shape=(40, 32))(x)
    x = LSTM(128)(x)
    x = Dropout(dropout_rate)(x)
    prediction = Dense(1, activation='sigmoid', name='final')(x)

    return Model(input=inputs, output=prediction)


def character_level_cnn2(conf):

    model_param, pre_param = conf["model_parameters"], conf["preprocessing_parameters"]
    limit_characters = pre_param["limit_characters"]
    number_of_characters = pre_param["number_of_characters"]
    dropout_rate = 0.25

    # input layer
    inputs = Input(shape=(limit_characters, 1), dtype='int32')
    l1 = Lambda(lambda x: K.one_hot(x, num_classes=number_of_characters))(inputs)
    emb = Reshape(target_shape=(number_of_characters, limit_characters))(l1)

    # 1st-conv
    x = Conv1D(filters=256, kernel_size=3, padding='same',
               activation='relu', data_format='channels_first', name='embedding')(emb)
    x = Reshape(target_shape=(160, 256))(x)
    x = MaxPooling1D(pool_size=2, data_format='channels_first', padding='same')(x)
    #x = Conv1D(filters=256, kernel_size=3, padding='same',
    #           activation='relu', data_format='channels_first')(x)
    #x = Conv1D(filters=256, kernel_size=3, padding='same',
    #           activation='relu', data_format='channels_first')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    # concentlate
    x = Reshape(target_shape=(160, 128))(x)
    x = LSTM(512)(x)
    x = Dropout(dropout_rate)(x)
    prediction = Dense(1, activation='sigmoid', name='final')(x)

    return Model(input=inputs, output=prediction)


def character_level_cnn_concatenate(conf):
    model_param, pre_param = conf["model_parameters"], conf["preprocessing_parameters"]
    limit_characters = pre_param["limit_characters"]
    number_of_characters = pre_param["number_of_characters"]
    dropout_rate = 0.25

    # input layer
    inputs = Input(shape=(limit_characters, 1), dtype='int32')
    l1 = Lambda(lambda x: K.one_hot(x, num_classes=number_of_characters), name='input')(inputs)
    emb = Reshape(target_shape=(number_of_characters, limit_characters))(l1)

    # 1st-conv
    x = Conv1D(filters=256, kernel_size=3, padding='same',
               activation='relu', data_format='channels_first', name='embedding')(emb)
    x = Reshape(target_shape=(160, 256))(x)
    x = MaxPooling1D(pool_size=2, data_format='channels_first', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)

    # 2nd
    x2 = Reshape(target_shape=(128, 160))(x)
    x2 = Conv1D(filters=128, kernel_size=3, padding='same',
                activation='relu', data_format='channels_first')(x2)
    x2 = Reshape(target_shape=(160, 128))(x2)
    x2 = MaxPooling1D(pool_size=2, data_format='channels_first', padding='same')(x2)
    x2 = BatchNormalization()(x2)
    x2 = Dropout(dropout_rate)(x2)

    # 3rd
    x3 = Reshape(target_shape=(64, 160))(x2)
    x3 = Conv1D(filters=64, kernel_size=3, padding='same',
                activation='relu', data_format='channels_first')(x3)
    x3 = Reshape(target_shape=(160, 64))(x3)
    x3 = MaxPooling1D(pool_size=2, data_format='channels_first', padding='same')(x3)
    x3 = BatchNormalization()(x3)
    x3 = Dropout(dropout_rate)(x3)

    # concentlate
    x = Concatenate(axis=-1)([x, x2, x3])

    x = Bidirectional(LSTM(200))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    prediction = Dense(2, activation='softmax', name='final')(x)

    return Model(input=inputs, output=prediction)
