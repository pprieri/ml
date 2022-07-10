import numpy as np

proba = np.stack([model_1.predict_proba(X_test)[:, 1],
                  model_2.predict_proba(X_test)[:, 1],
                  model_3.predict_proba(X_test)[:, 1]]).T
n = 3
arithmetic = proba.mean(axis=1)
geometric = proba.prod(axis=1)**(1/n)
harmonic = 1 / np.mean(1. / (proba + 0.00001), axis=1)
mean_of_powers = np.mean(proba**n, axis=1)**(1/n)
logarithmic = np.expm1(np.mean(np.log1p(proba), axis=1))


cormat = np.corrcoef(proba.T)
np.fill_diagonal(cormat, 0.0)
W = 1 / np.mean(cormat, axis=1)
W = W / sum(W) # normalizing to sum==1.0
weighted = proba.dot(W)
ras = roc_auc_score(y_true=y_test, y_score=weighted)
print(f"Weighted averaging ROC-AUC is: {ras:0.5f}")
