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


class NumericalUnivariateImputer(BaseEstimator, TransformerMixin):


    def __init__(self,numeric='auto',imputer_type=None,
                si_strategy='mean'):
        
        if type(numeric)==str and numeric!='auto':
            self.numeric = [numeric]
        else:
            self.numeric = numeric
        
        self.imputer_type = imputer_type
        self.si_strategy = si_strategy # ['mean','median','most_frequent']

        #These below are learnt attributes
        self.imputers = dict()

    def fit(self,X,y=None):
        
        if self.numeric=='auto':
            self.numeric=X.select_dtypes('number').columns.tolist()

        temp = X.loc[:,self.numeric].copy()
        temp['target'] = y

        if self.imputer_type is None:
            pass

        else:
            for variable in self.numeric:
                imputer = SimpleImputer(strategy=self.si_strategy)

                try:
                    self.imputers[variable] = imputer.fit(temp.loc[:,[variable]])
                except:
                    pass

        return self

    def transform(self,X,y=None):

        Xt = X.copy()

        for variable in self.numeric:

            if self.imputer_type is None:
                pass
            else:
                Xt.loc[:,[variable]] = (self.imputers[variable]).transform(Xt.loc[:,[variable]])

        return Xt

    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

# var = 'Humidity'

# ni = NumericalUnivariateImputer(numeric=var,imputer_type='simpleimputer')

# mask = df.loc[:,var].isnull()

# print(df[mask][var].head())

# ni.numeric
# (ni.fit_transform(df)).loc[mask,var].head()