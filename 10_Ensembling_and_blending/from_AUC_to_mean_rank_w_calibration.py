import numpy as np
from sklearn.model_selection import KFold

from sklearn.preprocessing import MinMaxScaler
proba = np.stack(
    [model_1.predict_proba(X_train)[:, 1],
    model_2.predict_proba(X_train)[:, 1],
    model_3.predict_proba(X_train)[:, 1]]).T
arithmetic = MinMaxScaler().fit_transform(proba).mean(axis=1)
ras = roc_auc_score(y_true=y_test, y_score=arithmetic)
print(f"Mean averaging ROC-AUC is: {ras:0.5f}"
