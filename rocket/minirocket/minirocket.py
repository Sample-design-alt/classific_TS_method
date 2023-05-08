import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sktime.transformations.panel.rocket import (
    MiniRocket,
    MiniRocketMultivariate,
    MiniRocketMultivariateVariable,
)
from sktime.datasets import load_from_ucr_tsv_to_dataframe
import os

datasets = [
    'Gunpoint',
]



base_path = r'E:\数据集\TSC'
for dataset in datasets:
    X_train, y_train = load_from_ucr_tsv_to_dataframe(os.path.join(base_path, f"{dataset}/{dataset}_TRAIN.tsv"))
    X_test, y_test = load_from_ucr_tsv_to_dataframe(os.path.join(base_path, f"{dataset}/{dataset}_TEST.tsv"))
    minirocket_pipeline = make_pipeline(
        MiniRocket(),
        StandardScaler(with_mean=False),
        RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
    )
    minirocket_pipeline.fit(X_train, y_train)
    print(minirocket_pipeline.score(X_test, y_test))
