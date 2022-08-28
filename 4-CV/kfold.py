import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=1)

X = df_train.drop('failure',axis=1)
y = df_train['failure'].copy()
predictions = y.copy()
rf = RandomForestClassifier(random_state=51,n_jobs=-1)

for f, ( train_i, valid_i ) in enumerate( kf.split(X) ):

    print("# fold {}".format( f + 1))

    X_train = X.iloc[train_i]
    X_valid = X.iloc[valid_i]
    y_train = y.iloc[train_i]
    y_valid = y.iloc[valid_i]

    rf.fit( X_train, y_train )

    p = rf.predict_proba( X_valid )[:,1]

    auc = roc_auc_score( y_valid, p )
    print("# AUC: {:.2%}\n".format( auc ))

    predictions[ valid_i ] = p
