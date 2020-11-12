import numpy as np
import os
import sys

def calcNEDAcc(x, y, z, ori_w, ori_x, ori_y, ori_z):
    qwqx = np.multiply(ori_w, ori_x)
    qxqx = np.multiply(ori_x, ori_x)
    qzqx = np.multiply(ori_z, ori_x)
    qxqy = np.multiply(ori_x, ori_y)
    del ori_x
    qyqz = np.multiply(ori_y, ori_z)
    qyqy = np.multiply(ori_y, ori_y)
    qwqy = np.multiply(ori_w, ori_y)
    del ori_y
    qwqz = np.multiply(ori_w, ori_z)
    del ori_w
    qzqz = np.multiply(ori_z, ori_z)
    del ori_z

    qxqyqwqz_sub = 2 * np.subtract(qxqy, qwqz)
    qxqyqwqz_add = 2 * np.add(qxqy, qwqz)
    del qxqy,qwqz
    qxqzqwqy_sub = 2 * np.subtract(qzqx, qwqy)
    qxqzqwqy_add = 2 * np.add(qzqx, qwqy)
    del qzqx,qwqy
    qyqzqwqx_sub = 2 * np.subtract(qyqz, qwqx)
    qyqzqwqx_add = 2 * np.add(qyqz, qwqx)
    del qyqz,qwqx

    qxqxqyqy = 1 - 2 * np.add(qxqx, qyqy)
    qzqzqyqy = 1 - 2 * np.add(qzqz, qyqy)
    qxqxqzqz = 1 - 2 * np.add(qxqx, qzqz)
    del qxqx,qyqy,qzqz

    #SENSOR COORDINATE CHANGE
    x = np.add(np.multiply(qzqzqyqy, x), np.multiply(qxqyqwqz_sub, y))
    x = np.add(x, np.multiply(qxqzqwqy_add, z))
    del qxqyqwqz_sub,qxqzqwqy_add
    y = np.add(np.multiply(qxqyqwqz_add, x), np.multiply(qxqxqzqz, y))
    y =np.add(y, np.multiply(qyqzqwqx_sub, z))
    del qxqxqzqz,qyqzqwqx_sub
    z = np.add(np.multiply(qxqzqwqy_sub, x), np.multiply(qyqzqwqx_add, y))
    z = np.add(z, np.multiply(qxqxqyqy, z))
    
    return np.concatenate([x.reshape([-1, 500, 1]), y.reshape([-1, 500, 1]), z.reshape([-1, 500, 1])], axis=2)

argv = sys.argv[1]
path = '../Data/Raw/' + argv  + "/"
sensor_name = sys.argv[2]

X = np.loadtxt(path + sensor_name + "_x.txt", delimiter=' ')
Y = np.loadtxt(path + sensor_name + "_y.txt", delimiter=' ')
Z = np.loadtxt(path + sensor_name + "_z.txt", delimiter=' ')

Ori_W = np.loadtxt(path + "Ori_w.txt", delimiter=' ')
Ori_X = np.loadtxt(path + "Ori_x.txt", delimiter=' ')
Ori_Y = np.loadtxt(path + "Ori_y.txt", delimiter=' ')
Ori_Z = np.loadtxt(path + "Ori_z.txt", delimiter=' ')

NED_X = []
NED_Y = []
NED_Z = []

NED = calcNEDAcc(X, Y, Z, Ori_W, Ori_X, Ori_Y, Ori_Z)

np.savetxt(path + "NED_" + sensor_name + "_x.txt", NED[:, :, 0], delimiter=' ')
np.savetxt(path + "NED_" + sensor_name + "_y.txt", NED[:, :, 1], delimiter=' ')
np.savetxt(path + "NED_" + sensor_name + "_z.txt", NED[:, :, 2], delimiter=' ')