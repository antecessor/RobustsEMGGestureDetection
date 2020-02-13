import numpy as np
from keras import metrics
from keras.layers import Dense, LSTM, Conv1D, Bidirectional, MaxPool1D
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from tensorflow import keras

from python.utils.DLUtils import convertWindowsToBeUsedByDeepLearning
from python.utils.preProcessing import extractSoundFeatures


def lstmTrain(X_train, Y_train, X_test, Y_test):
    num_classes = len(np.unique(Y_train))
    X_train, Y_train = convertWindowsToBeUsedByDeepLearning(X_train, Y_train)
    X_test, Y_test = convertWindowsToBeUsedByDeepLearning(X_test, Y_test)
    X_train = extractSoundFeatures(X_train)
    X_test = extractSoundFeatures(X_test)
    # X_train = calculateICA(X_train, component=4)
    # X_test = calculateICA(X_test, component=4)

    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)

    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)

    signal_size_row = X_train[0].shape[0]
    signal_size_col = X_train[0].shape[1]
    input_shape = (signal_size_row, signal_size_col)
    lstm_out = 8
    model = Sequential()
    model.add(Conv1D(10, 16, padding='same', activation="relu", input_shape=input_shape))
    model.add(MaxPool1D(strides=4, padding='same'))
    model.add(Bidirectional(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
    model.add(LSTM(lstm_out, recurrent_dropout=0.2, activation="relu"))
    model.add(Dense(30, activation="relu"))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='nadam',
                  metrics=[metrics.Recall()])
    print(model.summary())

    batch_size = 64
    epochs = 300

    model.fit(X_train, Y_train,
              epochs=epochs,
              batch_size=batch_size, shuffle=True)

    out = np.round(model.predict(X_test, batch_size=batch_size))
    conf = confusion_matrix(np.argmax(Y_test, axis=1), np.argmax(out, axis=1))
    print(conf)
    return model, conf
