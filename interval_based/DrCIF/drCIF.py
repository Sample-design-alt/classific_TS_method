from sktime.classification.interval_based import DrCIF
from sktime.classification.dictionary_based import TemporalDictionaryEnsemble
from sktime.classification.dictionary_based import BOSSEnsemble
from sktime.classification.dictionary_based import TemporalDictionaryEnsemble
from sktime.datasets import load_unit_test
X_train, y_train = load_unit_test(split="train", return_X_y=True)
X_test, y_test = load_unit_test(split="test", return_X_y=True)
# clf = TemporalDictionaryEnsemble(n_parameter_samples=10,max_ensemble_size=3,randomly_selected_params=5)
clf = BOSSEnsemble()
clf.fit(X_train, y_train)
