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


class NumericalTransformer(BaseEstimator, TransformerMixin):

    def __init__(self,numeric='auto',imputer_type=None,si_strategy='mean',
                transformer_type=None,random_state=None,
                pt_method='yeo-johnson', qt_output_distribution='uniform'):
        
        if type(numeric)==str and numeric!='auto':
            self.numeric = [numeric]
        else:
            self.numeric = numeric
        
        self.imputer_type = imputer_type
        self.si_strategy = si_strategy
        self.transformer_type = transformer_type
        self.pt_method = pt_method
        self.qt_output_distribution = qt_output_distribution
        self.random_state = random_state
        self.imputers = dict()
        self.transformers = dict()
    
    def fit(self,X,y=None):
        
   
        if self.numeric=='auto':
                self.numeric=X.select_dtypes('number').columns.tolist()

        temp = X.loc[:,self.numeric].copy()
        temp['target'] = y

        for variable in self.numeric:
            
            if self.imputer_type is None:
                pass
            #complete

            elif self.imputer_type=='simpleimputer':
                imputer = SimpleImputer(strategy=self.si_strategy)

            if self.transformer_type is None:
                pass

            elif self.transformer_type=='powertransformer':
                transformer = PowerTransformer(method=self.pt_method)
            
            elif self.transformer_type=='quantiletransfomer':
                transformer = QuantileTransformer(output_distribution=self.qt_output_distribution,random_state=self.random_state)

            elif self.transformer_type=='standard':
                transformer = StandardScaler()
            
            elif self.transformer_type=='minmaxscaler':
                transformer = MinMaxScaler()

            elif self.transformer_type=='maxabsscaler':
                transformer = MaxAbsScaler()
            
            elif self.transformer_type=='robustscaler':
                transformer = RobustScaler()

            try:
                self.transformers[variable] = transformer.fit(temp.loc[:,[variable]])
            except:
                pass

        return self

    def transform(self,X,y=None):

        Xt = X.copy()

        for variable in self.numeric:

            if self.transformer_type is None:
                pass

            else:
                Xt.loc[:,[variable]] = (self.transformers[variable]).transform(Xt.loc[:,[variable]])

        return Xt

    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
    
#ex
# var = ['EDB chassis height front 6t rear 9t',
#  'Rated speed',
#  'RAT in ER',
#  'Humidity',
#  'FVX FAL txt',
#  'GB_rateio_2.08/2.12',
#  'Vehicle mass']

# # nt = Numerical_Transformer(numeric='EDB chassis height front 6t rear 9t',transformer_type='standard')
# nt = NumericalTransformer(numeric=var,transformer_type='quantiletransfomer',random_state=SEED)

# nt.numeric
# nt.fit_transform(df)[var].head()