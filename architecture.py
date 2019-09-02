from keras.models import Model, load_model
from keras.layers import Input, Dense, Conv2D, Dropout, MaxPooling2D
from keras.layers import Lambda, Embedding, Reshape, Activation, Flatten
from layer import CharacterEmbeddingLayer


def simple(conf):
    # parameter
    model_param, pre_param = conf["model_parameters"], conf["preprocessing_parameters"]
    limit_characters = pre_param["limit_characters"]
    number_of_characters = pre_param["number_of_characters"]
    embedding_dimension = model_param["embedding_dimension"]
    filter_size = model_param["filter_size"]
    convolution_width = model_param["convolution_width"]
    dense_size = model_param["dense_size"]

    # layer
    inputs = Input(shape=(1, limit_characters, number_of_characters))
    # embedding
    x1 = CharacterEmbeddingLayer(embedding_dimension, name='embedding')(inputs)
    x2 = Reshape(target_shape=(1, limit_characters, embedding_dimension))(x1)
    # conv
    x3 = Conv2D(filters=filter_size, kernel_size=(convolution_width, embedding_dimension), padding='valid',
                activation='relu', data_format='channels_first', name='conv')(x2)
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


def two_convolution(conf):
    # parameter
    model_param, pre_param = conf["model_parameters"], conf["preprocessing_parameters"]
    limit_characters = pre_param["limit_characters"]
    number_of_characters = pre_param["number_of_characters"]
    embedding_dimension = model_param["embedding_dimension"]
    filter_size = model_param["filter_size"]
    convolution_width = model_param["convolution_width"]
    dense_size = model_param["dense_size"]

    # layer
    inputs = Input(shape=(1, limit_characters, number_of_characters))

    # embedding
    x1 = CharacterEmbeddingLayer(embedding_dimension, name='embedding')(inputs)

    # two-conv
    c1 = Conv2D(filters=32, kernel_size=(convolution_width, embedding_dimension), padding='valid',
                activation='relu', data_format='channels_first', name='conv')(x1)
    r1 = Reshape(target_shape=(1, limit_characters - convolution_width + 1, 32))(c1)
    c2 = Conv2D(filters=filter_size, kernel_size=(convolution_width, 32), padding='valid',
                activation='relu', data_format='channels_first', name='conv2')(r1)

    #pooling
    p = MaxPooling2D((limit_characters - 2 * convolution_width + 2, 1), padding='valid', name='pooling',
                      data_format='channels_first')(c2)

    # fully-connected
    f1 = Flatten()(p)
    f2 = Dense(dense_size, activation='relu', name='dense_1')(f1)
    fd2 = Dropout(0.5)(f2)
    f3 = Dense(dense_size, activation='relu', name='dense_2')(fd2)
    fd3 = Dropout(0.5)(f3)
    f4 = Dense(2, activation=None, name='dense_3')(fd3)

    # predict
    prediction = Activation('softmax', name='preds')(f4)
    return Model(input=inputs, output=prediction)
