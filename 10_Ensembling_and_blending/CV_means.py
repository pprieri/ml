import numpy as np
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=0)
scores = list()

for k, (train_index, test_index) in enumerate(kf.split(X_train)):
    model_1.fit(X_train[train_index, :], y_train[train_index])
    model_2.fit(X_train[train_index, :], y_train[train_index])
    model_3.fit(X_train[train_index, :], y_train[train_index])

    proba = np.stack([model_1.predict_proba(X_train[test_index, :])[:, 1],
                      model_2.predict_proba(X_train[test_index, :])[:, 1],
                      model_3.predict_proba(X_train[test_index, :])[:, 1]]).T

    arithmetic = proba.mean(axis=1)
    ras = roc_auc_score(y_true=y_train[test_index], y_score=arithmetic)
    scores.append(ras)
    print(f"FOLD {k} Mean averaging ROC-AUC is: {ras:0.5f}")

print(f"CV Mean averaging ROC-AUC is: {np.mean(scores):0.5f}")
