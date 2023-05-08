from pyts.classification import LearningShapelets
from load_UCR import load_data

datasets = [
    'Gunpoint',
]

base_path = r'E:\数据集\TSC'
for dataset in datasets:
    X_train, X_test, y_train, y_test = load_data(base_path, dataset)

    # estimate
    clf = LearningShapelets(random_state=42, tol=0.01)
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))
