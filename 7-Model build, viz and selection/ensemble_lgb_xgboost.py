from sklearn.base import BaseEstimator, ClassifierMixin
import lightgbm as lgb
import xgboost as xgb

class ensemble_lgb_xgboost(BaseEstimator, ClassifierMixin):  

    def __init__(self, seed=0, nest_lgb=1.0, nest_xgb=1.0, cbt=0.5, ss=0.5, alpha=0.5):

        print('LGB + XGB')
        self.models = [lgb.LGBMClassifier(num_leaves=2, learning_rate=0.07, n_estimators=int(1400*nest_lgb),
                                          colsample_bytree=cbt, subsample=ss,
                                          nthread=-1, random_state=0+seed),
                       lgb.LGBMClassifier(num_leaves=3, learning_rate=0.07, n_estimators=int(800*nest_lgb),
                                          colsample_bytree=cbt, subsample=ss,
                                          nthread=-1, random_state=1+seed),
                       lgb.LGBMClassifier(num_leaves=4, learning_rate=0.07, n_estimators=int(800*nest_lgb),
                                          colsample_bytree=cbt, subsample=ss,
                                          nthread=-1, random_state=2+seed),
                       lgb.LGBMClassifier(num_leaves=5, learning_rate=0.07, n_estimators=int(600*nest_lgb),
                                          colsample_bytree=cbt, subsample=ss,
                                          nthread=-1, random_state=3+seed,),
                       xgb.XGBClassifier(max_depth=1,
                                         learning_rate=0.1,
                                         n_estimators=int(800*nest_xgb),
                                         subsample=ss,
                                         colsample_bytree=cbt,
                                         nthread=-1,
                                         seed=0+seed),
                       xgb.XGBClassifier(max_depth=2,
                                         learning_rate=0.1,
                                         n_estimators=int(400*nest_xgb),
                                         subsample=ss,
                                         colsample_bytree=cbt,
                                         nthread=-1,
                                         seed=1+seed),
                       xgb.XGBClassifier(max_depth=3,
                                         learning_rate=0.1,
                                         n_estimators=int(200*nest_xgb),
                                         subsample=ss,
                                         colsample_bytree=cbt,
                                         nthread=-1,
                                         seed=2+seed),
                       xgb.XGBClassifier(max_depth=4,
                                         learning_rate=0.1,
                                         n_estimators=int(200*nest_xgb),
                                         subsample=ss,
                                         colsample_bytree=cbt,
                                         nthread=-1,
                                         seed=3+seed)
                      ]
        self.weights = [(1-alpha)*1, (1-alpha)*1, (1-alpha)*1, (1-alpha)*0.5, alpha*0.5, alpha*1, alpha*1.5, alpha*0.5]


    def fit(self, X, y=None):

        for t, clf in enumerate(self.models):
            # print ('train', t)
            clf.fit(X, y)
        return self

    def predict(self, X):

        suma = 0.0
        for t, clf in enumerate(self.models):
            a = clf.predict_proba(X)[:, 1]
            suma += (self.weights[t] * a)
        return (suma / sum(self.weights))
            
    def predict_proba(self, X):
     
        return (self.predict(X))