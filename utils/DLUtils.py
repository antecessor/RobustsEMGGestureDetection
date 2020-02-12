def convertWindowsToBeUsedByDeepLearning(X, Y):
    n, channel, windowNumber, windowLength = X.shape
    newX = []
    newY = []
    for i in range(n):
        for w in range(windowNumber):
            newX.append(X[i, :, w, :])
            newY.append(Y[i])
    return newX, newY
