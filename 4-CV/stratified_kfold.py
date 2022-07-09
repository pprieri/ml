# Model selection
from sklearn.model_selection import StratifiedKFold

# Setting a 5-fold stratified cross-validation (note: shuffle=True)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
