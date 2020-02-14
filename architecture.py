from keras.models import Model, load_model
from keras.layers import Input, Dense, Conv2D, Dropout, MaxPooling2D, Conv2DTranspose, MaxPooling3D, UpSampling2D
from keras.layers import Lambda, Embedding, Reshape, Activation, Flatten, Conv1D, MaxPooling1D, LSTM
from keras.layers import BatchNormalization, Concatenate, Bidirectional
from keras.regularizers import l2
from keras.utils import np_utils
from layer import CharacterEmbeddingLayer
import tensorflow as tf
import keras.backend as K
import numpy as np
from pprint import pprint


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


def character_level_cnn_origin(conf):
    model_param, pre_param = conf["model_parameters"], conf["preprocessing_parameters"]
    limit_characters = pre_param["limit_characters"]
    number_of_characters = pre_param["number_of_characters"]
    dropout_rate = 0.5

    # input layer
    inputs = Input(shape=(limit_characters, 1), dtype='int64')
    x = Reshape(target_shape=(limit_characters,))(inputs)
    #l1 = Embedding(input_dim=1, output_dim=number_of_characters, embeddings_initializer='random_normal',
    #               embeddings_regularizer=l2(0.01))(x)
    l1 = Lambda(lambda x: K.one_hot(K.cast(x, "int64"), num_classes=number_of_characters))(inputs)
    emb = Reshape(target_shape=(limit_characters, number_of_characters), name='start')(l1)
    loop_num = 0

    # first_7_conv
    x = Conv1D(filters=256, kernel_size=7, padding='same', activation='relu')(emb)
    x = MaxPooling1D(pool_size=2, padding='valid')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    loop_num = loop_num + 1

    # 7-conv
    for i in range(model_param["conv_7_loop"] - 1):
        x = Conv1D(filters=256, kernel_size=7, padding='same', activation='relu')(x)
        x = MaxPooling1D(pool_size=2, padding='valid')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        loop_num = loop_num + 1

    # 3-conv
    for i in range(model_param["conv_3_loop"]):
        x = Conv1D(filters=256, kernel_size=3, padding='same', activation='relu')(x)
        x = MaxPooling1D(pool_size=2, padding='valid')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
        loop_num = loop_num + 1

    # concentlate
    x = MaxPooling1D(pool_size=limit_characters // (2 ** loop_num), padding='valid')(x)
    x = Flatten()(x)

    for i in range(3):
        x = Dense(1024, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)
    prediction = Dense(2, activation='softmax', name='final')(x)

    return Model(input=inputs, output=prediction)


def character_level_cnn_bilstm(conf):
    model_param, pre_param = conf["model_parameters"], conf["preprocessing_parameters"]
    limit_characters = pre_param["limit_characters"]
    number_of_characters = pre_param["number_of_characters"]
    convolution_widths = model_param["convolution_widths"]
    filter_sizes = model_param["filter_sizes"]
    pooling_sizes = model_param["pooling_sizes"]
    use_bn = [True for i in range(len(convolution_widths))] if "use_bn" not in model_param else model_param["use_bn"]
    cnn_regularizer = l2(1e-7) if "cnn_regularizer" not in model_param else l2(model_param["cnn_regularizer"])
    lstm_regularizer = l2(1e-6) if "lstm_regularizer" not in model_param else l2(model_param["lstm_regularizer"])
    cnn_dropout_rate = 0.00 if "cnn_dropout_rate" not in model_param else model_param["cnn_dropout_rate"]
    lstm_dropout_rate = 0.00 if "lstm_dropout_rate" not in model_param else model_param["lstm_dropout_rate"]
    params = {'conv_w': convolution_widths, 'fil_s': filter_sizes, 'pool_s': pooling_sizes, "use_bn": use_bn,
              "cnn_reg": cnn_regularizer.l2, "lstm_reg": lstm_regularizer.l2}
    pprint(params)

    # input layer
    inputs = Input(shape=(limit_characters, 1), dtype='int32')
    l = Lambda(lambda x: K.one_hot(K.cast(x, "int32"), num_classes=number_of_characters))(inputs)
    x = Reshape(target_shape=(limit_characters, number_of_characters), name='start')(l)


    # convolution
    for i in range(len(convolution_widths)):
        x = Conv1D(filters=filter_sizes[i], kernel_size=convolution_widths[i],
                   padding='same', activation='linear',
                   kernel_regularizer=cnn_regularizer, bias_regularizer=cnn_regularizer)(x)
        if use_bn[i]:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if pooling_sizes[i] > 1:
            x = MaxPooling1D(pool_size=pooling_sizes[i], padding='valid')(x)
        if cnn_dropout_rate > 0.00:
            x = Dropout(cnn_dropout_rate)(x)

    # bilstm
    lstm_size = 128 if "lstm_size" not in model_param else model_param["lstm_size"]
    # x = Bidirectional(LSTM(lstm_size, kernel_regularizer=lstm_regularizer, bias_regularizer=lstm_regularizer))(x)
    x = LSTM(lstm_size, kernel_regularizer=lstm_regularizer, bias_regularizer=lstm_regularizer,
             dropout=lstm_dropout_rate, return_sequences=True, go_backwards=True)(x)
    x = BatchNormalization()(x)
    x = LSTM(lstm_size, kernel_regularizer=lstm_regularizer, bias_regularizer=lstm_regularizer,
             dropout=lstm_dropout_rate, return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = LSTM(lstm_size, kernel_regularizer=lstm_regularizer, bias_regularizer=lstm_regularizer,
             dropout=lstm_dropout_rate)(x)
    x = BatchNormalization()(x)
    prediction = Dense(2, activation='softmax', name='final')(x)

    return Model(input=inputs, output=prediction)


def character_level_cnn_parallel(conf):
    model_param, pre_param = conf["model_parameters"], conf["preprocessing_parameters"]
    limit_characters = pre_param["limit_characters"]
    number_of_characters = pre_param["number_of_characters"]

    # input layer
    inputs = Input(shape=(limit_characters, 1), dtype='int32')
    l1 = Lambda(lambda x: K.one_hot(K.cast(x, "int32"), num_classes=number_of_characters))(inputs)
    r = Reshape(target_shape=(limit_characters, number_of_characters), name='start')(l1)

    convolution_widths = model_param["convolution_widths"]
    filter_sizes = model_param["filter_sizes"]
    pooling_sizes = model_param["pooling_sizes"]
    dense_size = model_param["dense_size"]
    use_bn = [True for i in range(len(convolution_widths))] if "use_bn" not in model_param else model_param["use_bn"]
    cnn_regularizer = l2(1e-7) if "cnn_regularizer" not in model_param else l2(model_param["cnn_regularizer"])
    params = {'conv_w': convolution_widths, 'fil_s': filter_sizes, 'pool_s': pooling_sizes, "use_bn": use_bn,
              "cnn_reg": cnn_regularizer.l2}
    pprint(params)

    c = []
    # convolution
    for i in range(len(convolution_widths)):
        x = Conv1D(filters=filter_sizes[i], kernel_size=convolution_widths[i],
                   padding='same', activation='relu',
                   kernel_regularizer=cnn_regularizer, bias_regularizer=cnn_regularizer)(r)
        if pooling_sizes[i] > 1:
            x = MaxPooling1D(pool_size=pooling_sizes[i], padding='valid')(x)
        if use_bn[i]:
            x = BatchNormalization()(x)
        c.append(x)

    # concatenate
    x = Concatenate()(c)
    x = Flatten()(x)
    for i in range(3):
        x = Dense(dense_size, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)
    prediction = Dense(2, activation='softmax', name='final')(x)

    return Model(input=inputs, output=prediction)


def character_level_cnn_serial(conf):
    model_param, pre_param = conf["model_parameters"], conf["preprocessing_parameters"]
    limit_characters = pre_param["limit_characters"]
    number_of_characters = pre_param["number_of_characters"]

    # input layer
    inputs = Input(shape=(limit_characters, 1), dtype='int32')
    l1 = Lambda(lambda x: K.one_hot(K.cast(x, "int32"), num_classes=number_of_characters))(inputs)
    x = Reshape(target_shape=(limit_characters, number_of_characters), name='start')(l1)

    convolution_widths = model_param["convolution_widths"]
    filter_sizes = model_param["filter_sizes"]
    pooling_sizes = model_param["pooling_sizes"]
    dense_size = model_param["dense_size"]
    use_bn = [True for i in range(len(convolution_widths))] if "use_bn" not in model_param else model_param["use_bn"]
    cnn_regularizer = l2(1e-7) if "cnn_regularizer" not in model_param else l2(model_param["cnn_regularizer"])
    cnn_dropout_rate = 0.00 if "cnn_dropout_rate" not in model_param else model_param["cnn_dropout_rate"]
    dense_dropout_rate = 0.00 if "dense_dropout_rate" not in model_param else model_param["dense_dropout_rate"]
    params = {'conv_w': convolution_widths, 'fil_s': filter_sizes, 'pool_s': pooling_sizes, "use_bn": use_bn,
              "cnn_reg": cnn_regularizer.l2}
    pprint(params)

    # convolution
    for i in range(len(convolution_widths)):
        x = Conv1D(filters=filter_sizes[i], kernel_size=convolution_widths[i],
                   padding='same', activation='linear',
                   kernel_regularizer=cnn_regularizer, bias_regularizer=cnn_regularizer)(x)
        if use_bn[i]:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        if pooling_sizes[i] > 1:
            x = MaxPooling1D(pool_size=pooling_sizes[i], padding='valid')(x)
        if cnn_dropout_rate > 0.00:
            x = Dropout(cnn_dropout_rate)(x)

    # dense
    x = Flatten()(x)
    for i in range(3):
        x = Dense(dense_size, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)
        if dense_dropout_rate > 0.00:
            x = Dropout(dense_dropout_rate)(x)
    prediction = Dense(2, activation='softmax', name='final')(x)

    return Model(input=inputs, output=prediction)


def character_level_cnn_serial_and_parallel(conf):
    model_param, pre_param = conf["model_parameters"], conf["preprocessing_parameters"]
    limit_characters = pre_param["limit_characters"]
    number_of_characters = pre_param["number_of_characters"]

    # input layer
    inputs = Input(shape=(limit_characters, 1), dtype='int32')
    l1 = Lambda(lambda x: K.one_hot(K.cast(x, "int32"), num_classes=number_of_characters))(inputs)
    r = Reshape(target_shape=(limit_characters, number_of_characters), name='start')(l1)

    convolution_widths = model_param["convolution_widths"]
    filter_sizes = model_param["filter_sizes"]
    pooling_sizes = model_param["pooling_sizes"]
    dense_size = model_param["dense_size"]
    use_bn = [True for i in range(len(convolution_widths))] if "use_bn" not in model_param else model_param["use_bn"]
    cnn_regularizer = l2(1e-7) if "cnn_regularizer" not in model_param else l2(model_param["cnn_regularizer"])
    params = {'conv_w': convolution_widths, 'fil_s': filter_sizes, 'pool_s': pooling_sizes, "use_bn": use_bn,
              "cnn_reg": cnn_regularizer.l2}
    pprint(params)

    c = []
    # convolution
    seq = convolution_widths[0]
    for i in range(len(convolution_widths[1])):
        x = r
        for j in range(seq):
            x = Conv1D(filters=filter_sizes[i], kernel_size=convolution_widths[i],
                       padding='same', activation='relu',
                       kernel_regularizer=cnn_regularizer, bias_regularizer=cnn_regularizer)(x)
            if pooling_sizes[i] > 1:
                x = MaxPooling1D(pool_size=pooling_sizes[i], padding='valid')(x)
            if use_bn[i]:
                x = BatchNormalization()(x)
        c.append(x)

    # concatenate
    x = Concatenate()(c)
    x = Flatten()(x)
    for i in range(3):
        x = Dense(dense_size, activation='relu', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)
    prediction = Dense(2, activation='softmax', name='final')(x)

    return Model(input=inputs, output=prediction)
