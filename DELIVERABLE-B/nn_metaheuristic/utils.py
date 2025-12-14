import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_data():
    iris = load_iris()
    X = iris.data
    y = iris.target

    encoder = OneHotEncoder(sparse_output=False)
    y_onehot = encoder.fit_transform(y.reshape(-1, 1))

    X_train, X_temp, y_train, y_temp, yoh_train, yoh_temp = train_test_split(
        X, y, y_onehot, test_size=0.4, random_state=42
    )

    X_val, X_test, y_val, y_test, yoh_val, yoh_test = train_test_split(
        X_temp, y_temp, yoh_temp, test_size=0.5, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test, yoh_train, yoh_val, yoh_test

def cross_entropy(y_true, y_pred):
    eps = 1e-12
    return -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1))
