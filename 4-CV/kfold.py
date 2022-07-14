import numpy as np
from sklearn.model_selection import KFold

kf = KFold(n_splits=2, shuffle=True, random_state=1)

for f, ( train_i, test_i ) in enumerate( cv ):

    print "# fold {}, {}".format( f + 1, ctime())

    x_train = x.iloc[train_i]
    x_test = x.iloc[test_i]
    y_train = y.iloc[train_i]
    y_test = y.iloc[test_i]

    clf.fit( x_train, y_train )

    p = clf.predict_proba( x_test )[:,1]

    auc = AUC( y_test, p )
    print "# AUC: {:.2%}\n".format( auc )

    predictions[ test_i ] = p
