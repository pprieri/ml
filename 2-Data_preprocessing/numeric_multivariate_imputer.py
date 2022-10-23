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


class NumericalMultivariateImputer(BaseEstimator, TransformerMixin):


    def __init__(self,numeric='auto',imputer_type=None,transformer_type=None,
                knn_n_neighbors=3,
                ii_estimator='bayesianridge',ii_initial_strategy='mean',ii_skip_complete=True,
                pt_method='yeo-johnson',
                qt_output_distribution='uniform',
                random_state=None):
        
        if type(numeric)==str and numeric!='auto':
            self.numeric = [numeric]
        else:
            self.numeric = numeric
        
        self.imputer_type = imputer_type
        self.qt_output_distribution = qt_output_distribution
        self.transformer_type = transformer_type
        self.knn_n_neighbors = knn_n_neighbors
        self.ii_initial_strategy = ii_initial_strategy
        self.ii_skip_complete = ii_skip_complete
        self.ii_estimator = ii_estimator
        self.pt_method = pt_method
        self.random_state = random_state

        #These below are learnt attributes
        self.transformers = dict()
        self.imputers = dict()

    def fit(self,X,y=None):
        
        if self.numeric=='auto':
            self.numeric=X.select_dtypes('number').columns.tolist()

        temp = X.loc[:,self.numeric].copy()
        temp['target'] = y

        # if self.transformer_type is None:
        #     pass

        # elif self.transformer_type=='powertransformer':
        #     transformer = PowerTransformer(method=self.pt_method)
        
        # elif self.transformer_type=='quantiletransfomer':
        #     transformer = QuantileTransformer(output_distribution=self.qt_output_distribution,random_state=self.random_state)

        # elif self.transformer_type=='standard':
        #     transformer = StandardScaler()
        
        # elif self.transformer_type=='minmaxscaler':
        #     transformer = MinMaxScaler()

        # elif self.transformer_type=='maxabsscaler':
        #     transformer = MaxAbsScaler()
        
        # elif self.transformer_type=='robustscaler':
        #     transformer = RobustScaler()

        # try:
        #     self.transformers[self.transformer_type] = transformer.fit(temp.loc[:,self.numeric])
        #     temp.loc[:,self.numeric] = (self.transformers[self.imputer_type]).transform(temp.loc[:,self.numeric])
        # except:
        #     pass

        if self.imputer_type is None:
            pass

        #First we work with the multivariate (global) imputers
        elif self.imputer_type=='iterativeimputer':
            
            #iterative imputer expects rather 'normal' distribution, we use yeo-johnson by default
            if self.ii_estimator=='randomforestregressor':
                self.ii_estimator=RandomForestRegressor()
            else:
                self.ii_estimator=BayesianRidge()

            imputer = IterativeImputer(estimator=self.ii_estimator,
                            initial_strategy=self.ii_initial_strategy,
                            skip_complete=self.ii_skip_complete)
            self.imputers[self.imputer_type] = imputer.fit(temp.loc[:,self.numeric])
        
        elif self.imputer_type=='knnimputer':

            imputer = KNNImputer(n_neighbors=self.knn_n_neighbors)
            self.imputers[self.imputer_type] = imputer.fit(temp.loc[:,self.numeric])

        return self

    def transform(self,X,y=None):
        
        Xt = X.loc[:,self.numeric].copy()
        
        if self.imputer_type is None:
            pass

        else:
            Xt = pd.DataFrame(data=(self.imputers[self.imputer_type]).transform(Xt),
                                index=Xt.index,columns=Xt.columns)

        return Xt

    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
    
#use
#A)1 col
# var = 'Humidity'

# ni = NumericalMultivariateImputer(numeric=var,imputer_type='iterativeimputer')

# mask = df.loc[:,var].isnull()

# print(df[mask][var].head())

# ni.numeric
# (ni.fit_transform(df)).loc[mask,var].head()

# #B) multiple cols

# var = ['EDB chassis height front 6t rear 9t',
#  'Rated speed',
#  'RAT in ER',
#  'Humidity',
#  'FVX FAL txt',
#  'GB_rateio_2.08/2.12',
#  'Vehicle mass']


# # ni = NumericalUnivariateImputer(numeric=var,imputer_type='simpleimputer',si_strategy='mean')
# ni = NumericalMultivariateImputer(numeric=var,imputer_type='iterativeimputer')

# mask = df.loc[:,var].isnull().any(axis=1)

# print(df[mask][var].head())

# ni.numeric
# (ni.fit_transform(df)).loc[mask,var].head()