import numpy as np
from numpy.dual import pinv
from numpy.lib.stride_tricks import as_strided
from scipy import signal
from scipy.signal import butter, lfilter
from sklearn.decomposition import FastICA


# calculating the SD signal
def calculateSD(sig):
    nRow, nCol = sig.shape
    singleDifferentialSignal = []
    for row in range(nRow - 1):
        for col in range(nCol):
            singleDifferentialSignal.extend(sig[row + 1, col] - sig[row, col])
    singleDifferentialSignal = np.array(singleDifferentialSignal)
    return singleDifferentialSignal


def calculateICA(sig, component=7):
    ica = FastICA(n_components=component)
    icaRes = []

    for index, sig in enumerate(sig):
        try:
            icaRes.append(np.array(ica.fit_transform(sig.transpose())).transpose())
        except:
            pass
    return np.array(icaRes)


def calculateFFTOnWindows(sdSigWindows):
    return np.array([np.array(np.abs(np.fft.fft(sdSig.transpose()))).transpose() for sdSig in sdSigWindows])


def prepareFiringSignal(firings, sizeInputSignal=None, numSignals=None):
    maxIndex = np.max([np.max(firing[0]) for firing in firings])
    numbers = len(firings)
    preparedFirings = np.zeros([numbers, maxIndex + 1], dtype=float)
    for idx, firing in enumerate(firings):
        preparedFirings[idx, firing[0]] = 1
    if sizeInputSignal:
        preparedFirings = preparedFirings[:, 0:sizeInputSignal]
    if numSignals:
        return preparedFirings[0:numSignals, :]
    return preparedFirings[1:30, :]


def windowingSignalWithOverLap(sig, windowSize, overlap):
    windows = []
    arr = np.asarray(sig)
    window = np.kaiser(windowSize, 5)
    window_step = windowSize - overlap
    new_shape = arr.shape[:-1] + ((arr.shape[-1] - overlap) // window_step, windowSize)
    new_strides = (arr.strides[:-1] + (window_step * arr.strides[-1],) + arr.strides[-1:])
    windows.append(as_strided(arr, shape=new_shape, strides=new_strides) * window)
    return windows


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data, axis=1)
    return y


def calculateWhiten(sdSig):
    return np.array(whiten(sdSig.transpose())).transpose()


def autocorr(x):
    corr = signal.correlate2d(x, x, boundary='symm', mode='same')
    return corr


def calculateNormalizedOnCorr(signalWindow, label):
    Rx = [autocorr(sigwin) for sigwin in signalWindow]
    mainSignalWindow = []
    mainLalbe = []
    for index, rx in enumerate(Rx):
        signalWindow[index] = pinv(rx) @ signalWindow[index]
        if signalWindow[index].shape[1] == signalWindow[0].shape[1]:
            mainSignalWindow.append(signalWindow[index])
            mainLalbe.append(label[index])

    return np.array(mainSignalWindow), np.array(mainLalbe)


def whiten(X, method='zca'):
    """
    Whitens the input matrix X using specified whitening method.
    Inputs:
        X:      Input data matrix with data examples along the first dimension
        method: Whitening method. Must be one of 'zca', 'zca_cor', 'pca',
                'pca_cor', or 'cholesky'.
    """
    X = X.reshape((-1, np.prod(X.shape[1:])))
    X_centered = X - np.mean(X, axis=0)
    Sigma = np.dot(X_centered.T, X_centered) / X_centered.shape[0]
    W = None

    if method in ['zca', 'pca', 'cholesky']:
        U, Lambda, _ = np.linalg.svd(Sigma)
        if method == 'zca':
            W = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)), U.T))
        elif method == 'pca':
            W = np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)), U.T)
        elif method == 'cholesky':
            W = np.linalg.cholesky(np.dot(U, np.dot(np.diag(1.0 / (Lambda + 1e-5)), U.T))).T
    elif method in ['zca_cor', 'pca_cor']:
        V_sqrt = np.diag(np.std(X, axis=0))
        P = np.dot(np.dot(np.linalg.inv(V_sqrt), Sigma), np.linalg.inv(V_sqrt))
        G, Theta, _ = np.linalg.svd(P)
        if method == 'zca_cor':
            W = np.dot(np.dot(G, np.dot(np.diag(1.0 / np.sqrt(Theta + 1e-5)), G.T)), np.linalg.inv(V_sqrt))
        elif method == 'pca_cor':
            W = np.dot(np.dot(np.diag(1.0 / np.sqrt(Theta + 1e-5)), G.T), np.linalg.inv(V_sqrt))
    else:
        raise Exception('Whitening method not found.')

    return np.dot(X_centered, W.T)
