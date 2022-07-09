# Setting the search space
space = [Real(0.01, 1.0, 'uniform', name='learning_rate'),
         Integer(1, 8, name='max_depth'),
         Real(0.1, 1.0, 'uniform', name='subsample'),
         Real(0.1, 1.0, 'uniform', name='colsample_bytree'),  # subsample ratio of columns by tree
         Real(0, 100., 'uniform', name='reg_lambda'),      # L2 regularization
         Real(0, 100., 'uniform', name='reg_alpha'),       # L1 regularization
         Real(1, 30, 'uniform', name='min_child_weight'),     # minimum sum of instance weight (hessian)
]

# Setting the search space: note that we didn't put the
#n_estimators in the search space as we will use early stopping!
