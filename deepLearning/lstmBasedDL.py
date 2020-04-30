import numpy as np
from keras import metrics
from keras.layers import Dense, LSTM, Conv1D, Bidirectional
from keras.models import Sequential
from keras.utils import to_categorical
from pycm import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tensorflow import keras

from python.utils.DLUtils import convertWindowsToBeUsedByDeepLearning
from python.utils.preProcessing import extractSoundFeatures, extractHudginsFeatures


def ldaTrain(X_train, Y_train, X_test, Y_test):
    X_train, Y_train = convertWindowsToBeUsedByDeepLearning(X_train, Y_train)
    X_test, Y_test = convertWindowsToBeUsedByDeepLearning(X_test, Y_test)
    X_train = extractHudginsFeatures(X_train)
    X_test = extractHudginsFeatures(X_test)
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)

    clf = LinearDiscriminantAnalysis()
    clf.fit(X_train, Y_train)
    out = clf.predict(X_test)
    cm = ConfusionMatrix(actual_vector=Y_test, predict_vector=out)
    print(cm)
    return clf, cm


def lstmTrain(X_train, Y_train, X_test, Y_test):
    num_classes = len(np.unique(Y_train))
    X_train, Y_train = convertWindowsToBeUsedByDeepLearning(X_train, Y_train)
    X_test, Y_test = convertWindowsToBeUsedByDeepLearning(X_test, Y_test)
    # X_train=calculateNormalizedOnCorr(X_train)
    # X_test = calculateNormalizedOnCorr(X_test)
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
    lstm_out = 32
    model = Sequential()
    model.add(Conv1D(10, 32, padding='same', activation="relu", input_shape=input_shape))
    # model.add(MaxPool1D(strides=4, padding='same'))
    model.add(Bidirectional(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2, return_sequences=True, input_shape=input_shape)))
    model.add(LSTM(lstm_out, recurrent_dropout=0.2, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='nadam',
                  metrics=[metrics.Recall()])
    print(model.summary())

    batch_size = 128
    epochs = 200

    model.fit(X_train, Y_train,
              epochs=epochs,
              batch_size=batch_size, shuffle=True)

    out = np.round(model.predict(X_test, batch_size=batch_size))
    cm = ConfusionMatrix(actual_vector=np.argmax(Y_test, axis=1), predict_vector=np.argmax(out, axis=1))
    print(cm)
    return model, cm
