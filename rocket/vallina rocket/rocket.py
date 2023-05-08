import numpy as np
from sklearn.linear_model import RidgeClassifierCV

from pyts.transformation import ROCKET
from load_UCR import load_data
from sktime.transformations.panel.rocket import Rocket
datasets = [
    'Gunpoint',
]

base_path = r'E:\数据集\TSC'
for dataset in datasets:
    X_train, X_test, y_train, y_test = load_data(base_path, dataset)
    rocket = ROCKET()  # by default, ROCKET uses 10,000 kernels
    rocket.fit(X_train)  # 50 150
    X_train_transform = rocket.transform(X_train)
    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    classifier.fit(X_train_transform, y_train)

    X_test_transform = rocket.transform(X_test)
    print(classifier.score(X_test_transform, y_test))


