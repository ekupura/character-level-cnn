from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Dropout, MaxPooling2D, Lambda, Embedding
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import keras.backend as K
from sklearn.model_selection import train_test_split

limit_characters = 150
number_of_characters = 67
batch_size = 128
embedding_dimension = 50
filter_size = 256
n_folds = 1
epochs = 5


def configure():
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
    model = configure()
    for layer in model.layers:
        layer.trainable = True
    model.compile(loss='binary_crossentropy',
                  optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),
                  metrics=['accuracy'])
    model.summary()

    # configure callback function
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    model_checkpoint = ModelCheckpoint('./model/fast_charcnn.h5', verbose=1, save_best_only=True)

    model_history = []

    for i in range(n_folds):
        print("Training on Fold: ", i + 1)
        x_t, x_val, y_t, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=0)
        x_t, x_val = x_t[x_t.shape[0] % batch_size:], x_val[x_val.shape[0] % batch_size:]
        y_t, y_val = y_t[y_t.shape[0] % batch_size:], y_val[y_val.shape[0] % batch_size:]
        results = model.fit(x_t, y_t, epochs=epochs, batch_size=batch_size,
                            callbacks=[early_stopping, model_checkpoint], verbose=1, validation_data=(x_val, y_val))
        print("Val Score: ", model.evaluate(x_val, y_val, batch_size=batch_size))
        print("=======" * 12, end="\n\n\n")
        model_history.append(results)
