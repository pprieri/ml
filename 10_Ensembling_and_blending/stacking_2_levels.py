import numpy as np
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=0)
scores = list()

first_lvl_oof = np.zeros((len(X_train), 3))
fist_lvl_preds = np.zeros((len(X_test), 3))

for k, (train_index, val_index) in enumerate(kf.split(X_train)):
    model_1.fit(X_train[train_index, :], y_train[train_index])
    first_lvl_oof[val_index, 0] = model_1.predict_proba(X_train[val_index, :])[:, 1]

    model_2.fit(X_train[train_index, :], y_train[train_index])
    first_lvl_oof[val_index, 1] = model_2.predict_proba(X_train[val_index, :])[:, 1]

    model_3.fit(X_train[train_index, :], y_train[train_index])
    first_lvl_oof[val_index, 2] = model_3.predict_proba(X_train[val_index, :])[:, 1]

model_1.fit(X_train, y_train)
fist_lvl_preds[:, 0] = model_1.predict_proba(X_test)[:, 1]

model_2.fit(X_train, y_train)
fist_lvl_preds[:, 1] = model_2.predict_proba(X_test)[:, 1]

model_3.fit(X_train, y_train)
fist_lvl_preds[:, 2] = model_3.predict_proba(X_test)[:, 1]

second_lvl_oof = np.zeros((len(X_train), 3))
second_lvl_preds = np.zeros((len(X_test), 3))

for k, (train_index, val_index) in enumerate(kf.split(X_train)):
    skip_X_train = np.hstack([X_train, first_lvl_oof])
    model_1.fit(skip_X_train[train_index, :], y_train[train_index])
    second_lvl_oof[val_index, 0] = model_1.predict_proba(skip_X_train[val_index, :])[:, 1]

    model_2.fit(skip_X_train[train_index, :], y_train[train_index])
    second_lvl_oof[val_index, 1] = model_2.predict_proba(skip_X_train[val_index, :])[:, 1]

    model_3.fit(skip_X_train[train_index, :], y_train[train_index])
    second_lvl_oof[val_index, 2] = model_3.predict_proba(skip_X_train[val_index, :])[:, 1]

skip_X_test = np.hstack([X_test, fist_lvl_preds])

model_1.fit(skip_X_train, y_train)
second_lvl_preds[:, 0] = model_1.predict_proba(skip_X_test)[:, 1]

model_2.fit(skip_X_train, y_train)
second_lvl_preds[:, 1] = model_2.predict_proba(skip_X_test)[:, 1]

model_3.fit(skip_X_train, y_train)
second_lvl_preds[:, 2] = model_3.predict_proba(skip_X_test)[:, 1]

arithmetic = second_lvl_preds.mean(axis=1)
ras = roc_auc_score(y_true=y_test, y_score=arithmetic)
scores.append(ras)
print(f"Stacking ROC-AUC is: {ras:0.5f}")
