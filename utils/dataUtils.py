import os

import numpy as np
import scipy.io as sio
from sklearn.model_selection import StratifiedKFold

from python.utils.preProcessing import windowingSignalWithOverLap, butter_bandpass_filter

windowLen = 500
overlap = 150


def getSignal(subject, dayFrom, sessionOfDay, filePath="../Data/"):
    if os.path.exists(filePath):
        folders = os.listdir(filePath)
        mainDirectory = filePath + folders[subject] + "/Data/DAQ/"
        foldersSignal = os.listdir(mainDirectory)
        selectedFolderMain = foldersSignal[3 * (dayFrom) + sessionOfDay]
        mainDirectory = mainDirectory + selectedFolderMain
        fileSignals = os.listdir(mainDirectory)
        dataLoaded = []
        if subject < 12:
            subjecType = "healthy"
        else:
            if sessionOfDay > 1:
                print("For amputee only two session exist")
            subjecType = "amputees"
        for i in range(120):
            mainDirectoryFile = mainDirectory + "/" + fileSignals[i]
            if fileSignals[i].__contains__("C001"):
                label = 0
            elif fileSignals[i].__contains__("C010"):
                label = 1
            elif fileSignals[i].__contains__("C011"):
                label = 2
            elif fileSignals[i].__contains__("C012"):
                label = 3
            elif fileSignals[i].__contains__("C013"):
                label = 4
            elif fileSignals[i].__contains__("C016"):
                label = 5
            elif fileSignals[i].__contains__("C017"):
                label = 6
            elif fileSignals[i].__contains__("C020"):
                label = 7
            else:
                label = 8
            mat = sio.loadmat(mainDirectoryFile)
            mvc = 0
            if str(mat['movement']).__contains__("30"):
                mvc = 30
            elif str(mat['movement']).__contains__("60"):
                mvc = 60
            elif str(mat['movement']).__contains__("90"):
                mvc = 90
            mat["outdataOriginal"] = butter_bandpass_filter(mat["outdataOriginal"], 20, 450, 1000)
            dataLoaded.append({'data': mat["outdataOriginal"][:, 1000:4000], 'label': label, 'movement': mvc, "subjectType": subjecType})

        return dataLoaded


def getSignalForADay(subject, dayFrom, filePath="../Data/"):
    data = []
    if subject < 12:
        for session in range(3):
            data.append(getSignal(subject, dayFrom, session, filePath))
    else:
        for session in range(2):
            data.append(getSignal(subject, dayFrom, session, filePath))
    return data


def getTrainTestKFoldBasedOnDayToDay(subject, mvc, filePath):
    data = []
    for day in range(5):
        data.extend(getSignalForADay(subject, day, filePath))
    allWindows = []
    allLabels = []
    kFold = 5
    for day in range(5):
        for j in range(120):
            if data[day][j]["movement"] == 0 or data[day][j]["movement"] == mvc:
                windows = windowingSignalWithOverLap(data[day][j]["data"], windowLen, overlap)
                allWindows.extend(windows)
                allLabels.append(data[day][j]["label"])
    allWindows = np.asarray(allWindows)
    allLabels = np.asarray(allLabels)
    skf = StratifiedKFold(n_splits=kFold)
    for train_index, test_index in skf.split(allWindows, allLabels):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_test, X_train = allWindows[train_index], allWindows[test_index]
        y_test, y_train = allLabels[train_index], allLabels[test_index]
        yield X_train, y_train, X_test, y_test


def getTrainTestKFoldBasedOnADay(subject, day, mvc, filePath):
    data = getSignalForADay(subject, day, filePath)
    allWindows = []
    allLabels = []
    kFold = 3
    if subject > 12:
        kFold = 2
    for session in range(kFold):
        for j in range(120):
            if data[session][j]["movement"] == 0 or data[session][j]["movement"] == mvc:
                windows = windowingSignalWithOverLap(data[session][j]["data"], windowLen, overlap)
                allWindows.extend(windows)
                allLabels.append(data[session][j]["label"])
    allWindows = np.asarray(allWindows)
    allLabels = np.asarray(allLabels)
    skf = StratifiedKFold(n_splits=kFold)
    for train_index, test_index in skf.split(allWindows, allLabels):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_test, X_train = allWindows[train_index], allWindows[test_index]
        y_test, y_train = allLabels[train_index], allLabels[test_index]
        yield X_train, y_train, X_test, y_test


def getTrainTestKFoldBasedOnSession(subject, day, session, mvc, filePath, kFold):
    data = getSignal(subject, day, session, filePath)
    allWindows = []
    allLabels = []
    for j in range(120):
        if data[j]["movement"] == 0 or data[j]["movement"] == mvc:
            windows = windowingSignalWithOverLap(data[j]["data"], windowLen, overlap)
            allWindows.extend(windows)
            allLabels.append(data[j]["label"])
    allWindows = np.asarray(allWindows)
    allLabels = np.asarray(allLabels)
    skf = StratifiedKFold(n_splits=kFold, shuffle=True)
    for train_index, test_index in skf.split(allWindows, allLabels):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = allWindows[train_index], allWindows[test_index]
        y_train, y_test = allLabels[train_index], allLabels[test_index]
        yield X_train, y_train, X_test, y_test


def convertLabel2OneDimentional(labels):
    return np.array([np.sum(label) for label in labels])
