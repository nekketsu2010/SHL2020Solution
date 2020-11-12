import numpy as np

from tqdm import tqdm

import os
import sys

args = sys.argv

def txtToNumpy(kind, hold_position, sensor):
    label = np.loadtxt("../Data/Raw/" + kind + "/" + hold_position + "/Label.txt")
    NGindex = []
    for i in tqdm(range(label.shape[0])):
        if np.unique(label[i]).size > 1:
            NGindex.append(i)
    print("NGindex", len(NGindex))

    # save numpy
    file_path = "../Data/Raw/" + kind + "/" + hold_position + "/" + sensor
    x = np.loadtxt(file_path + "_x.txt")
    y = np.loadtxt(file_path + "_y.txt")
    z = np.loadtxt(file_path + "_z.txt")
    x = np.delete(x, NGindex, 0).reshape([-1, 500, 1])
    y = np.delete(y, NGindex, 0).reshape([-1, 500, 1])
    z = np.delete(z, NGindex, 0).reshape([-1, 500, 1])
    result = np.concatenate([x, y, z], axis=2)
    if not os.path.isdir("../Data/numpy/" + kind + "/" + hold_position + "/"):
        os.makedirs("../Data/numpy/" + kind + "/" + hold_position + "/")
    np.save("../Data/numpy/" + kind + "/" + hold_position + "/" + sensor + ".npy", result)
    if not os.path.exists("../Data/numpy/" + kind + "/" + hold_position + "/Label.npy"):
        np.save("../Data/numpy/" + kind + "/" + hold_position + "/Label.npy", np.delete(label, NGindex, 0).reshape([-1, 500]))

def test_txtToNumpy(kind, sensor):
    file_path = "../Data/Raw/" + kind + "/" + sensor
    x = np.loadtxt(file_path + "_x.txt")
    y = np.loadtxt(file_path + "_y.txt")
    z = np.loadtxt(file_path + "_z.txt")
    result = np.concatenate([x, y, z], axis=2)
    if not os.path.isdir("../Data/numpy/" + kind + "/"):
        os.makedirs("../Data/numpy/" + kind + "/")
    np.save("../Data/numpy/" + kind + "/" + sensor + ".npy", result)

def load_numpy(kind, hold_position, sensor):
    x = np.load("../Data/numpy/" + kind + "/" + hold_position + "/" + sensor + ".npy")[:, :, 2]
    return x

if __name__ == "__main__":
    kind = args[1]
    hold_position = args[2]

    sensors = ['Acc', 'Gyr', 'Mag', 'LAcc']
    for sensor in sensors:
        print(kind, hold_position, "NED_" + sensor)
        if kind == "test":
            test_txtToNumpy(kind, sensor)
        else:
            txtToNumpy(kind, hold_position, "NED_" + sensor)