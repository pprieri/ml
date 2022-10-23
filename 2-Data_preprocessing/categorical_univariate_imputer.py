import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import datetime as dt
import random
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression,Lasso, Ridge, BayesianRidge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, MissingIndicator, KNNImputer, IterativeImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, PowerTransformer, QuantileTransformer, StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, PolynomialFeatures, KBinsDiscretizer


class CategoricalUnivariateImputer(BaseEstimator, TransformerMixin):


    def __init__(self,categories='auto',imputer_type=None,
                si_strategy='constant',si_fill_value=None,
                random_state=None):

        #These below are parameters
        if type(categories)==str and categories!='auto':
            self.categories = [categories] #in case we have only one, we convert it into a list
        else:
            self.categories = categories
        
        self.imputer_type = imputer_type #['constant','most_frequent']. 'constant' requires 'fill_value' (default is 'missing_value')
        self.si_strategy = si_strategy
        self.si_fill_value = si_fill_value
        self.random_state = random_state

        #These below are learnt attributes
        self.imputers = dict()
        
    def fit(self,X,y=None):

        if type(self.categories)=='auto':
            self.categories = np.where(X.dtypes == type(object()))[0]
        
        temp = X.loc[:, self.categories].copy()
        temp['target'] = y

        if self.imputer_type is None:
            pass
            #complete

        else:
            for variable in self.categories:

                if self.imputer_type is None:
                    pass

                elif self.imputer_type=='simpleimputer':
                    imputer = SimpleImputer(strategy=self.si_strategy,fill_value=self.si_fill_value)
                
                try:
                    self.imputers[variable] = imputer.fit(temp.loc[:,variable])
                except:
                    pass
                
        return self

    def transform(self, X):
        Xt = X.copy()
        
        for variable in self.categories:
            
            if self.imputer_type is None:
                pass
            else:
                Xt.loc[:,variable] = (self.imputers[variable]).transform(Xt.loc[:,variable])

        return Xt

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)