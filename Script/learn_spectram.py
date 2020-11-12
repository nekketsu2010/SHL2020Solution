import numpy as np
import os

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.initializers import he_normal

import pickle

import sys

kind = sys.argv[1]
nfft = sys.argv[2]
overlap = sys.argv[3]
step = str(sys.argv[4])

file_path = "../Output/"

def load(kind, sensor, axis, nfft, overlap):
    x = np.load(file_path + kind + "/Spectrum_" + sensor + "_" + axis + "_" + str(nfft) + "_" + str(overlap) + ".npy")
    return x.reshape([-1, 1, x.shape[1], x.shape[2], 1])

X_train = np.concatenate([load("train/Hips", "NED_Acc", "xy", nfft, overlap), load("train/Hips", "NED_Acc", "z", nfft, overlap),\
    load("train/Hips", "NED_Gyr", "xy", nfft, overlap), load("train/Hips", "NED_Gyr", "z", nfft, overlap),\
        load("train/Hips", "NED_Mag", "xy", nfft, overlap), load("train/Hips", "NED_Mag", "z", nfft, overlap)], axis=1)

X_val = np.concatenate([load("validation/Hips", "NED_Acc", "xy", nfft, overlap), load("validation/Hips", "Acc", "z", nfft, overlap),\
    load("validation/Hips", "NED_Gyr", "xy", nfft, overlap), load("validation/Hips", "NED_Gyr", "z", nfft, overlap),\
        load("validation/Hips", "NED_Mag", "xy", nfft, overlap), load("validation/Hips", "NED_Mag", "z", nfft, overlap)], axis=1)

# Load Label
Y_train = np.load("../Data/numpy/train/Hips/Label.npy")[:, 0] - 1
val_label = np.load("../Data/numpy/validation/Hips/Label.npy")[:, 0] - 1

# trainデータはNaN消す
X_train = np.delete(X_train, 120845, 0)
Y_train = np.delete(Y_train, 120845, 0)

# round5する
X_train = np.round(X_train, 5)
X_val = np.round(X_val, 5)

# パターン分ける
pattern_file = np.load("../pattern2.npy").reshape([-1])
X_val_pattern0 = X_val[pattern_file == 0]
X_val_pattern1 = X_val[pattern_file == 1]
del X_val

if step == "1":
    x_train = X_train
    x_val = X_val_pattern1
    y_train = Y_train
    y_val = val_label[pattern_file == 1]
else:
    x_train = X_val_pattern1
    x_val = X_val_pattern0
    y_train = val_label[pattern_file == 1]
    y_val = val_label[pattern_file == 0]

del X_train, X_val_pattern0, X_val_pattern1, Y_train, val_label

def BuildModel(input_shape):
    input1 = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, (3, 3), strides=(1,1), padding='valid', activation='relu', kernel_initializer=he_normal())(input1)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    try:
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.2)(x)
    except:
        pass
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(16, activation='relu')(x)
    x = models.Model(inputs=input1, outputs=x)    
    return x

save_folder = "/ModelCheckPoint_" + str(nfft) + "_" + str(overlap) + "/"

def getModel(step="1"):
    if step == "1":
        x1 = BuildModel(x_train[0, 0].shape)
        x2 = BuildModel(x_train[0, 1].shape)
        x3 = BuildModel(x_train[0, 2].shape)
        x4 = BuildModel(x_train[0, 3].shape)
        x5 = BuildModel(x_train[0, 4].shape)
        x6 = BuildModel(x_train[0, 5].shape)

        combined = layers.concatenate([x1.output, x2.output, x3.output, x4.output, x5.output, x6.output])

        z = layers.Dense(64, activation='relu')(combined)
        z = layers.Dense(16, activation='relu')(z)
        z = layers.Dense(8, activation='softmax')(z)

        model = models.Model(inputs=[x1.input, x2.input, x3.input, x4.input, x5.input, x6.input], outputs=z)
        model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
    else:
        model = tf.keras.models.load_model(save_folder + "model_" + nfft + "_" + overlap + "_1.hdf5")
    return model

model = getModel(step)

import os
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

cp_cb = tf.keras.callbacks.ModelCheckpoint(filepath=save_folder + "model_" + nfft + "_" + overlap + "_" + step + ".hdf5", 
                                           monitor='val_loss', verbose=0, save_best_only=True, mode='auto')
es_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=0, mode='auto')
history = model.fit([x_train[:, 0], x_train[:, 1], x_train[:, 2], x_train[:, 3], x_train[:, 4], x_train[:, 5]], y_train, epochs=512, batch_size=256, \
    validation_data=([x_val[:, 0], x_val[:, 1], x_val[:, 2], x_val[:, 3], x_val[:, 4], x_val[:, 5]], y_val), callbacks=[cp_cb, es_cb])
