from sklearn import metrics
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sktime.classification.feature_based import Catch22Classifier
from load_UCR import load_data
from sktime.datasets import load_basic_motions, load_italy_power_demand
from sktime.transformations.panel.catch22 import Catch22
import pandas as pd






# def load_data(base_path,dataset):
#     # dataset_path = os.path.join(base_path,dataset)
#     # train_data = pd.read_csv(dataset_path + '\\' + dataset + '_TRAIN.tsv',sep='\t')
#     # test_data = pd.read_csv(dataset_path + '\\' + dataset + '_TEST.tsv')
#     # y_train =train_data.iloc[:,0]
#     # X_train =train_data.iloc[:,1:]
#     # return X_train,y_train
#
#     dataset_path = os.path.join(base_path, dataset)
#     train_data = np.loadtxt(dataset_path + '\\' + dataset + '_TRAIN.tsv')
#     test_data = np.loadtxt(dataset_path + '\\' + dataset + '_TEST.tsv')
#     y_train = train_data[:, 0]
#     X_train = train_data[:, 1:]
#     y_test = test_data[:, 0]
#     X_test = test_data[:, 1:]
#
#     # preprocessing
#     le = LabelEncoder()
#     le.fit(y_train)
#     le.transform(y_train)
#     le.transform(y_test)
#     mms = MinMaxScaler()
#     mms.fit(X_train)
#     mms.transform(X_train)
#     mms.transform(X_test)
#     return X_train, X_test, y_train, y_test

datasets = [
    'Gunpoint',
]
base_path = r'E:\数据集\TSC'
for dataset in datasets:
    X_train, X_test, y_train, y_test = load_data(base_path, dataset)

    c22f = Catch22Classifier(random_state=0)
    c22f.fit(X_train, y_train)
    c22f_preds = c22f.predict(X_test)
    print("C22F Accuracy: " + str(metrics.accuracy_score(y_test, c22f_preds)))