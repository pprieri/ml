##Optuna has become a gold standard among Kagglers
##because of its speedier convergence to optimal hyperparameters
## in the time allowed in a Kaggle Notebook

##Another notable aspect s that we can optimize for either minimization or maximization,
##depending on the problem (scikit-optimize works only for minimization)

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor
import optuna

from optuna.integration import XGBoostPruningCallback

## note the tree_method and predictor below
## if you don't want to use GPU, just remove the tree_method and predictor parameters

def objective(trial):

    params = {
            'learning_rate': trial.suggest_float("learning_rate", 0.01, 1.0, log=True),
            'reg_lambda': trial.suggest_loguniform("reg_lambda", 1e-9, 100.0),
            'reg_alpha': trial.suggest_loguniform("reg_alpha", 1e-9, 100.0),
            'subsample': trial.suggest_float("subsample", 0.1, 1.0),
            'colsample_bytree': trial.suggest_float("colsample_bytree", 0.1, 1.0),
            'max_depth': trial.suggest_int("max_depth", 1, 7),
            'min_child_weight': trial.suggest_int("min_child_weight", 1, 7),
            'gamma': trial.suggest_float("gamma", 0.1, 1.0, step=0.1)
    }

    model = XGBRegressor(
        random_state=0,
        tree_method="gpu_hist",
        predictor="gpu_predictor",
        n_estimators=10_000,
        **params,
    )

    model.fit(x, y, early_stopping_rounds=300, eval_set=[(x_val, y_val)], verbose=1000,
              callbacks=[XGBoostPruningCallback(trial, 'validation_0-rmse')])
    preds = model.predict(x_test)
    rmse = mean_squared_error(y_test, preds, squared=False)

    return rmse

x, x_val, y, y_val = train_test_split(X_train, y_train, random_state=0, test_size=0.2)
x, x_test, y, y_test = train_test_split(x, y, random_state=0, test_size=0.25)
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)


print(study.best_value)
print(study.best_params)
