from sktime.classification.distance_based import ElasticEnsemble
from load_UCR import load_data
datasets=[
    'Gunpoint',
]


base_path = r'E:\数据集\TSC'
for dataset in datasets:
    X_train, X_test, y_train, y_test = load_data(base_path,dataset)

    #estimate
    clf = ElasticEnsemble(proportion_of_param_options=0.1, proportion_train_for_test=0.1, distance_measures=["dtw", "ddtw"],
                          majority_vote=True)
    clf.fit(X_train, y_train)
    print(clf.score(X_test,y_test))
