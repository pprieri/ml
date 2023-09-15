import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
from path import Path
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold

class Config:
    input_path = Path('../input/porto-seguro-safe-driver-prediction')
    optuna_lgb = False
    n_estimators = 1500
    early_stopping_round = 150
    cv_folds = 5
    random_state = 0
    params = {'objective': 'binary',
              'boosting_type': 'gbdt',
              'learning_rate': 0.01,
              'max_bin': 25,
              'num_leaves': 31,
              'min_child_samples': 1500,
              'colsample_bytree': 0.7,
              'subsample_freq': 1,
              'subsample': 0.7,
              'reg_alpha': 1.0,
              'reg_lambda': 1.0,
              'verbosity': 0,
              'random_state': 0}
    
config = Config()

train = pd.read_csv(config.input_path / 'train.csv', index_col='id')
test = pd.read_csv(config.input_path / 'test.csv', index_col='id')
submission = pd.read_csv(config.input_path / 'sample_submission.csv', index_col='id')

calc_features = [feat for feat in train.columns if "_calc" in feat]
cat_features = [feat for feat in train.columns if "_cat" in feat]

# Extracting target
target = train["target"]

if "target" in train.columns:
    train = train.drop("target", axis="columns")

# Removing calc features
train = train.drop(calc_features, axis="columns")
test = test.drop(calc_features, axis="columns")

# Adding one-hot encoding of cat features
train = pd.get_dummies(train, columns=cat_features)
test = pd.get_dummies(test, columns=cat_features)

assert((train.columns==test.columns).all())

#Optional if new metrics
# from numba import jit

# @jit
# def eval_gini(y_true, y_pred):
#     y_true = np.asarray(y_true)
#     y_true = y_true[np.argsort(y_pred)]
#     ntrue = 0
#     gini = 0
#     delta = 0
#     n = len(y_true)
#     for i in range(n-1, -1, -1):
#         y_i = y_true[i]
#         ntrue += y_i
#         gini += y_i * delta
#         delta += 1 - y_i
#     gini = 1 - 2 * gini / (ntrue * (n - ntrue))
#     return gini

# def gini_lgb(y_true, y_pred):
#     eval_name = 'normalized_gini_coef'
#     eval_result = eval_gini(y_true, y_pred)
#     is_higher_better = True
#     return eval_name, eval_result, is_higher_better


 if config.optuna_lgb:
        
    def objective(trial):
        params = {
                'learning_rate': trial.suggest_float("learning_rate", 0.01, 1.0),
                'num_leaves': trial.suggest_int("num_leaves", 3, 255),
                'min_child_samples': trial.suggest_int("min_child_samples", 3, 3000),
                'colsample_bytree': trial.suggest_float("colsample_bytree", 0.1, 1.0),
                'subsample_freq': trial.suggest_int("subsample_freq", 0, 10),
                'subsample': trial.suggest_float("subsample", 0.1, 1.0),
                'reg_alpha': trial.suggest_loguniform("reg_alpha", 1e-9, 10.0),
                'reg_lambda': trial.suggest_loguniform("reg_lambda", 1e-9, 10.0),
        }
        
        score = list()
        skf = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)

        for train_idx, valid_idx in skf.split(train, target):
            X_train, y_train = train.iloc[train_idx], target.iloc[train_idx]
            X_valid, y_valid = train.iloc[valid_idx], target.iloc[valid_idx]

            model = lgb.LGBMClassifier(**params,
                                    n_estimators=1500,
                                    early_stopping_round=150,
                                    force_row_wise=True)

            callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False)]
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric=gini_lgb, callbacks=callbacks)
            score.append(model.best_score_['valid_0']['normalized_gini_coef'])

        return np.mean(score)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=300)

    print("Best Gini Normalized Score", study.best_value)
    print("Best parameters", study.best_params)
    
    params = {'objective': 'binary',
            'boosting_type': 'gbdt',
            'verbosity': 0,
            'random_state': 0}
    
    params.update(study.best_params)
    
else:
    params = config.params

preds = np.zeros(len(test))
oof = np.zeros(len(train))
metric_evaluations = list()


if not config.bagging:
    
    skf = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.random_state)

else:
    
    skf = RepeatedStratifiedKFold(n_splits=config.cv_folds, n_repeats=config.n_repeats,random_state=config.random_state)

print(skf.get_n_splits(train,target))

for idx, (train_idx, valid_idx) in enumerate(skf.split(train, target)):
    print(f"CV fold {idx}")
    X_train, y_train = train.iloc[train_idx], target.iloc[train_idx]
    X_valid, y_valid = train.iloc[valid_idx], target.iloc[valid_idx]
    
    model = lgb.LGBMClassifier(**params,
                               n_estimators=config.n_estimators,
                               early_stopping_round=config.early_stopping_round,
                               force_row_wise=True)
    
    callbacks=[lgb.early_stopping(stopping_rounds=150), 
               lgb.log_evaluation(period=100, show_stdv=False)]
                                                                                           
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric=gini_lgb, callbacks=callbacks)
    metric_evaluations.append(model.best_score_['valid_0']['normalized_gini_coef'])
    preds += model.predict_proba(test, num_iteration=model.best_iteration_)[:,1] / skf.get_n_splits(train,target)
    oof[valid_idx] = model.predict_proba(X_valid, num_iteration=model.best_iteration_)[:,1]

print(f"LightGBM CV Gini Normalized Score: {np.mean(metric_evaluations):0.3f} ({np.std(metric_evaluations):0.3f})")

submission['target'] = preds
submission.to_csv('lgb_submission.csv')

oofs = target.to_frame()
oofs['target'] = oof
oofs.to_csv('lgb_oof.csv')