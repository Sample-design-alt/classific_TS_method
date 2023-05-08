import os
import numpy as np
from sklearn.preprocessing import LabelEncoder,MinMaxScaler

datasets=[
    'Gunpoint',
]


def load_data(base_path,dataset):
    dataset_path = os.path.join(base_path,dataset)
    train_data = np.loadtxt(dataset_path + '\\' + dataset + '_TRAIN.tsv')
    test_data = np.loadtxt(dataset_path + '\\' + dataset + '_TEST.tsv')
    y_train = train_data[:, 0]
    X_train = train_data[:, 1:]
    y_test = test_data[:, 0]
    X_test = test_data[:, 1:]

    # preprocessing
    le = LabelEncoder()
    le.fit(y_train)
    le.transform(y_train)
    le.transform(y_test)
    mms = MinMaxScaler()
    mms.fit(X_train)
    mms.transform(X_train)
    mms.transform(X_test)
    return X_train, X_test, y_train, y_test

