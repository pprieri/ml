##it is absed on Bayesian Opt

##scikit-optimize works only for minimization

# The BayesSearchCV function offered by Scikit-optimize is certainly convenient, because it wraps and arranges all the elements of a
# hyperparameter search by itself, but it also has limitations. For instance, you may find it useful in a competition to:
# Have more control over each search iteration, for instance mixing random search and Bayesian search
# Be able to apply early stopping on algorithms
# Customize your validation strategy more
# Stop experiments that do not work early (for instance, immediately evaluating the performance of the single cross validation folds when it is available, instead of waiting to have
# all folds averaged at the end)
# Create clusters of hyperparameter sets that perform in a similar way (for instance, in order to create multiple models differing  only in the hyperparameters used, to be used for a blending # ensemble

# Fixing a problem with Skopt (see https://github.com/scikit-optimize/scikit-optimize/issues/981)
!conda install scipy=='1.5.3' --y

!pip install scikit-learn=='0.23.2'

# Importing core libraries
import numpy as np
import pandas as pd
from time import time
import pprint
import joblib
from functools import partial

# Suppressing warnings because of skopt verbosity
import warnings
warnings.filterwarnings("ignore")

# Classifier/Regressor
from xgboost import XGBRegressor

# Model selection
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

# Metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer

# Skopt functions
from skopt import BayesSearchCV
from skopt.callbacks import DeadlineStopper, DeltaYStopper
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args # decorator to convert a list of parameters to named arguments
from skopt import gp_minimize, forest_minimize
from skopt import gbrt_minimize, dummy_minimize

# Data processing
from sklearn.preprocessing import OrdinalEncoder

# Setting the scoring function
scoring = partial(mean_squared_error, squared=False)

# Setting the cv strategy
kf = KFold(n_splits=5, shuffle=True, random_state=0)

# Setting the search space: note that we didn't put the
#n_estimators in the search space as we will use early stopping!
space = [Real(0.01, 1.0, 'uniform', name='learning_rate'),
         Integer(1, 8, name='max_depth'),
         Real(0.1, 1.0, 'uniform', name='subsample'),
         Real(0.1, 1.0, 'uniform', name='colsample_bytree'),  # subsample ratio of columns by tree
         Real(0, 100., 'uniform', name='reg_lambda'),      # L2 regularization
         Real(0, 100., 'uniform', name='reg_alpha'),       # L1 regularization
         Real(1, 30, 'uniform', name='min_child_weight'),     # minimum sum of instance weight (hessian)
]

#
model = XGBRegressor(n_estimators=10_000,
                     booster='gbtree', random_state=0)


# The objective function to be minimized
def make_objective(model, X, y, space, cv, scoring, validation=0.2):
    # This decorator converts your objective function with named arguments into one that
    # accepts a list as argument, while doing the conversion automatically.
    @use_named_args(space)
    def objective(**params):
        model.set_params(**params)
        print("\nTesting: ", params)
        validation_scores = list()
        for k, (train_index, test_index) in enumerate(kf.split(X, y)):
            val_index = list()
            train_examples = len(train_index)
            train_examples = int(train_examples * (1 - validation))
            train_index, val_index = train_index[:train_examples], train_index[train_examples:]

            start_time = time()
            model.fit(X.iloc[train_index,:], y[train_index],
                      early_stopping_rounds=50,
                      eval_set=[(X.iloc[val_index,:], y[val_index])],
                      verbose=0
                    )
            end_time = time()

            rounds = model.best_iteration

            test_preds = model.predict(X.iloc[test_index,:])
            test_score = scoring(y[test_index], test_preds)
            print(f"CV Fold {k+1} rmse:{test_score:0.5f} - {rounds} rounds - it took {end_time-start_time:0.0f} secs")
            validation_scores.append(test_score)

            if len(history[k]) >= 10:
                threshold = np.percentile(history[k], q=25)
                if test_score > threshold:
                    print(f"Early stopping for under-performing fold: threshold is {threshold:0.5f}")
                    return np.mean(validation_scores)

            history[k].append(test_score)
        return np.mean(validation_scores)

    return objective


objective = make_objective(model,
                           X_train, y_train,
                           space=space,
                           cv=kf,
                           scoring=scoring)

def onstep(res):
    global counter
    x0 = res.x_iters   # List of input points
    y0 = res.func_vals # Evaluation of input points
    print('Last eval: ', x0[-1],
          ' - Score ', y0[-1])
    print('Current iter: ', counter,
          ' - Best Score ', res.fun,
          ' - Best Args: ', res.x)
    joblib.dump((x0, y0), 'checkpoint.pkl') # Saving a checkpoint to disk
    counter += 1

counter = 0
history = {i:list() for i in range(5)}
used_time = 0
gp_round = dummy_minimize(func=objective,
                          dimensions=space,
                          n_calls=30,
                          callback=[onstep],
                          random_state=0)


x0, y0 = joblib.load('checkpoint.pkl')
print(len(x0))

x0, y0 = joblib.load('checkpoint.pkl')

gp_round = gp_minimize(func=objective,
                       x0=x0,              # already examined values for x
                       y0=y0,              # observed values for x0
                       dimensions=space,
                       acq_func='gp_hedge',
                       n_calls=30,
                       n_initial_points=0,
                       callback=[onstep],
                       random_state=0)

x0, y0 = joblib.load('checkpoint.pkl')
print(len(x0))

print(f"Best score: {gp_round.fun:0.5f}")
print("Best hyperparameters:")
for sp, x in zip(gp_round.space, gp_round.x):
    print(f"{sp.name:25} : {x}")

#source: https://github.com/PacktPublishing/The-Kaggle-Book/blob/main/chapter_08/hacking-bayesian-optimization.ipynb
