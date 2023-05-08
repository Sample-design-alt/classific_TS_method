from pyts.classification import TimeSeriesForest
from load_UCR import load_data

datasets = [
    'Gunpoint',
]

base_path = r'E:\数据集\TSC'
for dataset in datasets:
    X_train, X_test, y_train, y_test = load_data(base_path, dataset)
    clf = TimeSeriesForest(random_state=43, n_windows=3)  # n_windows: the time series is divided to n_window*3
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)
 