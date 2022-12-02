import numpy as np 
 
from functools import partial 
from scipy.optimize import fmin 
from sklearn import metrics

class OptimizeLogLoss: 
    """ 
    Class for optimizing LogLoss. 
    This class is all you need to find best weights for  
    any model and for any metric and for any types of predictions. 
    With very small changes, this class can be used for optimization of  
    weights in ensemble models of _any_ type of predictions 
    """ 
    def __init__(self): 
        self.coef_ = 0 
 
    def _logloss(self, coef, X, y): 
        """ 
        This functions calulates and returns LogLoss. 
        :param coef: coef list, of the same length as number of models 
        :param X: predictions, in this case a 2d array 
        :param y: targets, in our case binary 1d array 
        """ 
        # multiply coefficients with every column of the array 
        # with predictions. 
        # this means: element 1 of coef is multiplied by column 1 
        # of the prediction array, element 2 of coef is multiplied  
        # by column 2 of the prediction array and so on! 
        x_coef = X * coef 
 
        # create predictions by taking row wise sum 
        predictions = np.sum(x_coef, axis=1) 
         
        # calculate LogLoss score 
        LogLoss_score = log_loss(y, predictions) 
 
        # return negative LogLoss 
        return LogLoss_score 
 
    def fit(self, X, y): 
        # remember partial from hyperparameter optimization chapter? 
        loss_partial = partial(self._logloss, X=X, y=y) 
         
        # dirichlet distribution. you can use any distribution you want 
        # to initialize the coefficients 
        # we want the coefficients to sum to 1 
        initial_coef = np.random.dirichlet(np.ones(X.shape[1]), size=1) 
 
        # use scipy fmin to minimize the loss function, in our case LogLoss 
        self.coef_ = fmin(loss_partial, initial_coef, disp=True) 
 
    def predict(self, X): 
        # this is similar to _LogLoss function 
        x_coef = X * self.coef_ 
        predictions = np.sum(x_coef, axis=1) 
        return predictions 
    
#use

X = probas
y = train_labels
X_train, X_valid, y_train, y_valid = train_test_split(X,y,random_state=0)

opt = OptimizeLogLoss()
opt.fit(X_train,y_train)
opt_probas = opt.predict(X_valid)