import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict,StratifiedKFold
from sklearn.metrics import roc_auc_score
from time import ctime

train = pd.read_csv( 'data/train.csv' )
test = pd.read_csv( 'data/test.csv' )

train = train.fillna(-1).drop(["id", "target"], axis=1)
test = test.fillna(-1).drop(["id", axis=1])
X = train.append(test)
y = [0] * len(train) + [1] * len(test)

model = RandomForestClassifier()
cv_preds = cross_val_predict(model, X, y, cv=5, n_jobs=-1)
print(roc_auc_score(y_true=y, y_score=cv_preds[:,1]))

##for feat importance: uncomment below
model.fit(X, y)
##




test.drop( 't_id', axis = 1, inplace = True )
test['target'] = 0		# dummy for preserving column order when concatenating

train['is_test'] = 0
test['is_test'] = 1

orig_train = train.copy()
assert( np.all( orig_train.columns == test.columns ))

train = pd.concat([orig_train, test],axis=0)
train.reset_index( inplace = True, drop = True )

x = train.drop( [ 'is_test', 'target' ], axis = 1 )
y = train.is_test

#

n_estimators = 100
clf = RandomForestClassifier( n_estimators = n_estimators, n_jobs = -1 )

predictions = np.zeros(y.shape)

cv = StratifiedKFold( y, n_folds = 5, shuffle = True, random_state = 5678 )

for f, ( train_i, test_i ) in enumerate( cv ):

	print(f"# fold {f+1}, {ctime()}")

	x_train = x.iloc[train_i]
	x_test = x.iloc[test_i]
	y_train = y.iloc[train_i]
	y_test = y.iloc[test_i]

	clf.fit( x_train, y_train )

	p = clf.predict_proba( x_test )[:,1]

	auc = AUC( y_test, p )
	print(f"# AUC: {auc:.2%}\n")

	predictions[ test_i ] = p

#code here: https://github.com/zygmuntz/adversarial-validation/blob/master/numerai/sort_train.py

#CASE A: Ideal case
# Training and testing examples coming from the same distribution,
# so that the validation error should give good estimation of the test error
# and classifier should generalize well to unseen test examples.
# if we attempted to train a classifier to distinguish training examples from test examples,
# it would perform no better than random. This would correspond to ROC AUC of 0.5.
# example: http://fastml.com/adversarial-validation-part-one/

#CASE B: Train and test differ
#To be more precise, we’ll choose a number of misclassified examples that the model
#was most certain about. It means that they look like test examples but in reality are training examples.
#First, let’s try training a classifier to tell train from test,
#just like we did with the Santander data. Mechanics are the same, but instead of 0.5, we get 0.87 AUC, meaning that the model is able to classify the examples pretty well (at least in terms of AUC, which measures ordering/ranking).
