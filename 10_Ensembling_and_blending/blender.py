import numpy as np

X_blend, X_holdout, y_blend, y_holdout = train_test_split(X_train, y_train, test_size=0.25, random_state=0)

model_1.fit(X_blend, y_blend)
model_2.fit(X_blend, y_blend)
model_3.fit(X_blend, y_blend)

proba = np.stack([model_1.predict_proba(X_holdout)[:, 1],
                  model_2.predict_proba(X_holdout)[:, 1],
                  model_3.predict_proba(X_holdout)[:, 1]]).T

# By looking at the coefficients, we can figure out which model contributes more
# to the meta-ensemble. However, remember that coefficients also rescale
#  probabilities when they are not well calibrated, so a larger coefficient
#  for a model may not imply that it is the most important one.
#  If you want to figure out the role of each model in the blend by
#  looking at coefficients, you first have to rescale them by standardization (in our code example, this has been done
# using Scikit-learn’s StandardScaler)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
proba = scaler.fit_transform(proba)

from sklearn.linear_model import LogisticRegression
blender = LogisticRegression(solver='liblinear')
blender.fit(proba, y_holdout)

print(blender.coef_)

test_proba = np.stack([model_1.predict_proba(X_test)[:, 1],
                       model_2.predict_proba(X_test)[:, 1],
                       model_3.predict_proba(X_test)[:, 1]]).T

blending = blender.predict_proba(test_proba)[:, 1]
ras = roc_auc_score(y_true=y_test, y_score=blending)
print(f"ROC-AUC for linear blending {model} is: {ras:0.5f}")
