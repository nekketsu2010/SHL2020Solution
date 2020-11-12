import numpy as np
import scipy

import sys

def spectram(x, nfft, overlap):
    nfft = nfft
    overlap = nfft - overlap
    
    x = scipy.signal.spectrogram(x, fs=100, nfft=nfft, noverlap=overlap, nperseg=nfft)[2]
    x = (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)
    return x

def load_npy(sensor, file_name):
    x = np.load("../Data/numpy/" + file_name + "/" + sensor + ".npy")
    return x

def xy(x):
    x = np.sqrt(np.square(x[:, :, 0]) + np.square(x[:, :, 1]))
    return x.reshape([-1, 500])
def z(x):
    return x[:, :, 2].reshape([-1, 500])


if __name__ == "__main__":
    sensor = sys.argv[1]
    kind = sys.argv[2]
    nfft = int(sys.argv[3])
    overlap = int(sys.argv[4])

    X = load_npy(sensor, kind)
    if len(sys.argv) > 5:
        X = xy(X)
    else:
        X = z(X)

    X = np.apply_along_axis(spectram, 1, X, nfft=nfft, overlap=overlap)
    if len(sys.argv) > 5:
        np.save("../Output/" + kind + "/Spectrum_" + sensor + "_xy_" + str(nfft) + "_" + str(overlap), X)
    else:
        np.save("../Output/" + kind + "/Spectrum_" + sensor + "_z_" + str(nfft) + "_" + str(overlap), X)
