from sktime.classification.interval_based import CanonicalIntervalForest
from sktime.datasets import load_from_ucr_tsv_to_dataframe
import os
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
datasets = [
    'Gunpoint',
]



base_path = r'E:\数据集\TSC'
for dataset in datasets:
    X_train, y_train = load_from_ucr_tsv_to_dataframe(os.path.join(base_path, f"{dataset}/{dataset}_TRAIN.tsv"))
    X_test, y_test = load_from_ucr_tsv_to_dataframe(os.path.join(base_path, f"{dataset}/{dataset}_TEST.tsv"))
    CIF_pipeline = make_pipeline(
        # StandardScaler(),
        CanonicalIntervalForest(),
    )
    CIF_pipeline.fit(X_train, y_train)
    print(CIF_pipeline.score(X_test, y_test))