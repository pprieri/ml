clf = lgb.LGBMClassifier(boosting_type='gbdt',
                         metric='auc',
                         objective='binary',
                         n_jobs=1,
                         verbose=-1,
                         random_state=0)

search_spaces = {
    'learning_rate': Real(0.01, 1.0, 'log-uniform'),     # Boosting learning rate
    'n_estimators': Integer(30, 5000),                   # Number of boosted trees to fit
    'num_leaves': Integer(2, 512),                       # Maximum tree leaves for base learners
    'max_depth': Integer(-1, 256),                       # Maximum tree depth for base learners, <=0 means no limit
    'min_child_samples': Integer(1, 256),                # Minimal number of data in one leaf
    'max_bin': Integer(100, 1000),                       # Max number of bins that feature values will be bucketed
    'subsample': Real(0.01, 1.0, 'uniform'),             # Subsample ratio of the training instance
    'subsample_freq': Integer(0, 10),                    # Frequency of subsample, <=0 means no enable
    'colsample_bytree': Real(0.01, 1.0, 'uniform'),      # Subsample ratio of columns when constructing each tree
    'min_child_weight': Real(0.01, 10.0, 'uniform'),     # Minimum sum of instance weight (hessian) needed in a child (leaf)
    'reg_lambda': Real(1e-9, 100.0, 'log-uniform'),      # L2 regularization
    'reg_alpha': Real(1e-9, 100.0, 'log-uniform'),       # L1 regularization
    'scale_pos_weight': Real(1.0, 500.0, 'uniform'),     # Weighting of the minority class (Only for binary classification)
        }


# Although the number of hyperparameters to tune when using
# LightGBM may appear daunting, in reality only a few of them
# maĴer a lot. Given a fixed number of iterations and learning rate,
# just a few are the most impactful ( feature_fraction, num_leaves,
# subsample, reg_lambda, reg_alpha, min_data_in_leaf), as explained in this
# blog article by Kohei Ozaki, a Kaggle Grandmaster:
# https://medium.com/optuna/lightgbm-tuner-new-optunaintegration-for-hyperparameter-optimization-
# 8b7095e99258. Kohei Ozaki leverages this fact in order to create a
# fast-tuning procedure for Optuna (you’ll find more on the Optuna
# optimizer at the end of this chapter
