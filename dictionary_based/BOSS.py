from sktime.transformations.panel.dictionary_based import SFA
from sktime.datasets import load_unit_test
import numpy as np
X_test, y_test = load_unit_test(split="test", return_X_y=True)


#sfa虽然
sfa=SFA(window_size=2,alphabet_size=2)
s=sfa.fit_transform(X_test)
print(s)