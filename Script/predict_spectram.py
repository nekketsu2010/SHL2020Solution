import numpy as np
import os

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.initializers import he_normal
from tensorflow.python.keras.utils.vis_utils import plot_model

import sys

file_path = "../Output/"
def load(kind, sensor, axis, nfft, overlap):
    x = np.load(file_path + kind + "/Spectrum_" + sensor + "_" + axis + "_" + str(nfft) + "_" + str(overlap) + ".npy")
    return x.reshape([-1, 1, x.shape[1], x.shape[2], 1])

if __name__ == "__main__":
    kind = sys.argv[1]
    nfft = sys.argv[2]
    overlap = sys.argv[3]

    X = np.concatenate([load(kind, "NED_Acc", "xy", nfft, overlap), load(kind, "NED_Acc", "z", nfft, overlap),\
        load(kind, "NED_Gyr", "xy", nfft, overlap), load(kind, "NED_Gyr", "z", nfft, overlap),\
        load(kind, "NED_Mag", "xy", nfft, overlap), load(kind, "NED_Mag", "z", nfft, overlap)], axis=1)

    # round5する
    X = np.round(X, 5)

    save_folder = "/ModelCheckPoint_" + str(nfft) + "_" + str(overlap) + "/" #保存ディレクトリを指定(後ろにスラッシュ入れてね)
    model = tf.keras.models.load_model(save_folder + "model_" + nfft + "_" + overlap + "_2.hdf5")
    predict = model.predict([X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5]])

    np.save("../" + kind + "_predict_time_frequency_spectrum", predict)