import xgboost as xgb
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

def load_numpy(file_name):
    features = [
        'NED_LAcc_XY_mean',
        'NED_LAcc_XY_var',
        'NED_LAcc_Z_mean',
        'NED_LAcc_Z_var',
        'NED_LAcc_Z_skew',
        'NED_LAcc_Z_kurtosis',
        "NED_LAcc_Z_sum_frequency_range5Hz",
        "NED_Gyr_Z_sum_frequency_range5Hz",
        "NED_Mag_norm_sum_frequency_range5Hz",
        "NED_LAcc_Z_amplitude_frequency_range5Hz",
        "NED_Gyr_Z_amplitude_frequency_range5Hz",
        "NED_Mag_norm_amplitude_frequency_range5Hz",
    ]
    
    file_path = "../Output/" + file_name + "/"
    x = np.load(file_path + features[0]).reshape([-1, 1])
    for i in range(1, len(features)):
        feature = features[i]
        if i >= 6 and i <= 8:
            x = np.concatenate([x, np.load(file_path + feature)[:, 0:-1:2].reshape([-1, 1])], axis=1)
        else:
            x = np.concatenate([x, np.load(file_path + feature).reshape([-1, 1])], axis=1)
    
    if not file_name == 'test':
        return x

    y = np.load("../Data/numpy/" + file_name + "/Label.npy")
    return x, y

if __name__ == "__main__":
    # train
    X_train, Y_train = np.delete(load_numpy("train/Hips"), 120845, 0)

    # validation
    X_val, Y_val = load_numpy("validation/Hips")

    # test
    X_test = load_numpy("test")

    # round
    X_train = np.round(X_train, 5)
    X_val = np.round(X_val, 5)
    X_test = np.round(X_test, 5)

    # Standardization
    user1_std = StandardScaler()
    X_train[:, :38] = user1_std.transform(X_train[:, :38])

    user23_std = StandardScaler()
    user23_std.fit(np.concatenate([X_val, X_test], axis=0)[:, :38])
    X_val[:, :38] = user23_std.transform(X_val[:, :38])
    X_test[:, :38] = user23_std.transform(X_test[:, :38])

    # train_validation split
    pattern_data = np.load("../pattern2.npy").reshape([-1])
    X_train = np.concatenate([X_train, X_val[pattern_data == 1]], axis=0)
    X_val = X_val[pattern_data == 0 or pattern_data == 2]
    Y_train = np.concatenate([Y_train, Y_val[pattern_data == 1]], axis=0)
    Y_val = Y_val[pattern_data == 0 or pattern_data == 2]


    model = xgb.XGBClassifier(max_depth=18, min_child_weight=7, learning_rate=0.01, gamma=0.005, sub_sample=0.9, colsample_bytree=0.8, 
                            n_estimators=10000, n_jobs=-1, tree_method='gpu_hist', gpu_id=0)
    model.fit(X_train, Y_train, early_stopping_rounds=30, eval_set=[(X_train, Y_train), (X_val, Y_val)], eval_metric='merror', verbose=False)

    test_predict = model.predict_proba(X_test)
    np.save("test_predict_xgboost", test_predict)
