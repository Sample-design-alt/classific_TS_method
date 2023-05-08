from sktime.classification.deep_learning.tapnet import TapNetClassifier
from load_UCR import load_data
datasets=[
    'Gunpoint',
]


base_path = r'E:\数据集\TSC'
for dataset in datasets:
    X_train, X_test, y_train, y_test = load_data(base_path,dataset)

    tapnet = TapNetClassifier(n_epochs=20,batch_size=4)  # doctest: +SKIP
    tapnet.fit(X_train, y_train)
    tapnet.score(X_test,y_test)