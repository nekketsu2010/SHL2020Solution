import numpy as np
from scipy import stats

import os
import sys

def load_numpy(file_name, sensor):
    x = np.load("../Data/numpy/" + file_name + "/NED_" + sensor + ".npy")
    return x

def traditional_features(x, file_name):
    x_xy = np.sqrt(np.square(x[:, :, 0]) + np.square(x[:, :, 1]))
    x_xy_mean = np.mean(x_xy, axis=1)
    x_xy_var = np.var(x_xy, axis=1)
    x_z_mean = np.mean(x[:, :, 2], axis=1)
    x_z_abs_mean = np.mean(np.abs(x[:, :, 2]), axis=1)
    x_z_var = np.var(x[:, :, 2], axis=1)
    x_z_kurtosis = stats.kurtosis(x[:, :, 2], axis=1)
    x_z_skew = stats.skew(x[:, :, 2], axis=1)
    x = np.concatenate([x_xy_mean.reshape(-1, 1), x_xy_var.reshape([-1, 1]), x_z_mean.reshape([-1, 1]), x_z_abs_mean.reshape([-1, 1]), x_z_var.reshape([-1, 1]), x_z_skew.reshape([-1, 1]), x_z_kurtosis.reshape([-1, 1])], axis=1)
    del x_xy, x_xy_mean, x_xy_var, x_z_mean, x_z_abs_mean, x_z_var

    file_path = "../Output/" + file_name + "/"
    if not os.path.isdir(file_path):
        os.makedirs(file_path)
    np.save(file_path + "NED_LAcc_XY_mean", x[:, 0])
    np.save(file_path + "NED_LAcc_XY_var", x[:, 1])
    np.save(file_path + "NED_LAcc_Z_mean", x[:, 2])
    np.save(file_path + "NED_LAcc_Z_abs_mean", x[:, 3])
    np.save(file_path + "NED_LAcc_Z_var", x[:, 4])
    np.save(file_path + "NED_LAcc_Z_skew", x[:, 5])
    np.save(file_path + "NED_LAcc_Z_kurtosis", x[:, 6])

# 0~7Hz
# 7~12Hz ... 42Hz~47Hz
def fft_max(x, file_name, sensor):
    for i in range(x.shape[0]):
        x[i] = x[i] - np.mean(x[i])
    x_tmp = np.abs(np.fft.fft(x, axis=1))
    
    max_amplitude = np.amax(x_tmp[:, 0:35], axis=1)
    max_index = np.argmax(x_tmp[:, 0:35], axis=1)
    max_frequency = max_index * 0.2
    result = np.concatenate([max_amplitude.reshape([-1, 1]), max_frequency.reshape([-1, 1])], axis=1)
    
    frequency_range = [i for i in range(35, 275, 25)] # 0~50Hzを5Hz刻みでやる
    for frequency in frequency_range:
        max_amplitude = np.amax(x_tmp[:, frequency:frequency+25], axis=1)
        max_index = np.argmax(x_tmp[:, frequency:frequency+25], axis=1)
        max_frequency = (max_index + frequency) * 0.2
        result_tmp = np.concatenate([max_amplitude.reshape([-1, 1]), max_frequency.reshape([-1, 1])], axis=1)
        try:
            result = np.concatenate([result, result_tmp], axis=1)
        except:
            result = result_tmp.copy()
    
    file_path = "../Output/" + file_name + "/"
    if not os.path.isdir(file_path):
        os.makedirs(file_path)
    np.save(file_path + "NED_" + sensor + "_amplitude_frequency_range5Hz", result)


# 0~7Hz
# 7~12Hz ... 42Hz~47Hz
def fft_sum(x, file_name, sensor):
    for i in range(x.shape[0]):
        x[i] = x[i] - np.mean(x[i])
    x_tmp = np.abs(np.fft.fft(x, axis=1))
    
    sum_amplitude = np.sum(x_tmp[:, 0:35], axis=1)
    sum_index = np.argmax(x_tmp[:, 0:35], axis=1)
    sum_frequency = sum_index * 0.2
    result = np.concatenate([sum_amplitude.reshape([-1, 1]), sum_frequency.reshape([-1, 1])], axis=1)
    
    frequency_range = [i for i in range(35, 275, 25)] # 0~50Hzを5Hz刻みでやる
    for frequency in frequency_range:
        sum_amplitude = np.sum(x_tmp[:, frequency:frequency+25], axis=1)
        sum_index = np.argmax(x_tmp[:, frequency:frequency+25], axis=1)
        sum_frequency = (sum_index + frequency) * 0.2
        result_tmp = np.concatenate([sum_amplitude.reshape([-1, 1]), sum_frequency.reshape([-1, 1])], axis=1)
        try:
            result = np.concatenate([result, result_tmp], axis=1)
        except:
            result = result_tmp.copy()
    
    file_path = "../Output/" + file_name + "/"
    if not os.path.isdir(file_path):
        os.makedirs(file_path)
    np.save(file_path + "NED_" + sensor + "_sum_frequency_range5Hz", result)

if __name__ == "__main__":
    args = sys.argv
    kind = args[1]

    sensors = ['Gyr', 'Mag', 'LAcc']
    hold_positions = ['Bag', 'Hips', 'Torso', 'Hand']
    
    if kind == "test":
        file_name = kind
        x = load_numpy(file_name, "LAcc")
        traditional_features(x, file_name)

        # FFT
        # LAcc_Z, Gyr_Z, Mag_norm
        x = load_numpy(file_name, "LAcc")[:, :, 2]
        fft_max(x, file_name, 'LAcc_Z')
        fft_sum(x, file_name, 'LAcc_Z')

        x = load_numpy(file_name, 'Gyr')[:, :, 2]
        fft_max(x, file_name, 'Gyr_Z')
        fft_sum(x, file_name, 'Gyr_Z')

        x = load_numpy(file_name, 'Mag')
        x = np.sqrt(np.square(x[:, :, 0]) + np.square(x[:, :, 1]) + np.square(x[:, :, 2]))
        fft_max(x, file_name, 'Mag_norm')
        fft_sum(x, file_name, 'Mag_norm')
        
        exit()
    
    for hold_position in hold_positions:
        # Traditional feature values
        file_name = kind + "/" + hold_position
        x = load_numpy(file_name, "LAcc")
        traditional_features(x, file_name)

        # FFT
        # LAcc_Z, Gyr_Z, Mag_norm
        x = load_numpy(file_name, "LAcc")[:, :, 2]
        fft_max(x, file_name, 'LAcc_Z')
        fft_sum(x, file_name, 'LAcc_Z')

        x = load_numpy(file_name, 'Gyr')[:, :, 2]
        fft_max(x, file_name, 'Gyr_Z')
        fft_sum(x, file_name, 'Gyr_Z')

        x = load_numpy(file_name, 'Mag')
        x = np.sqrt(np.square(x[:, :, 0]) + np.square(x[:, :, 1]) + np.square(x[:, :, 2]))
        fft_max(x, file_name, 'Mag_norm')
        fft_sum(x, file_name, 'Mag_norm')
    