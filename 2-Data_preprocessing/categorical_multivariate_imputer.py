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


class CategoricalMultivariateImputer(BaseEstimator, TransformerMixin):


    def __init__(self,categories='auto',imputer_type=None,
                knn_n_neighbors=3,
                ii_estimator='bayesianridge',ii_initial_strategy='most_frequent',ii_skip_complete=True,
                random_state=None):

        #These below are parameters
        if type(categories)==str and categories!='auto':
            self.categories = [categories] #in case we have only one, we convert it into a list
        else:
            self.categories = categories
        
        self.imputer_type = imputer_type #['constant','most_frequent']. 'constant' requires 'fill_value' (default is 'missing_value')
        self.knn_n_neighbors = knn_n_neighbors
        self.ii_initial_strategy = ii_initial_strategy
        self.ii_skip_complete = ii_skip_complete
        self.ii_estimator = ii_estimator
        self.random_state = random_state

        #These below are learnt attributes
        self.encoders = dict()
        self.imputers = dict()
        
    def fit(self,X,y=None):

        if type(self.categories)=='auto':
            self.categories = np.where(X.dtypes == type(object()))[0]
        
        temp = X.loc[:, self.categories].copy()
        
        self.encoder=LabelEncoder()

        for variable in self.categories: #we need to label encoder preserving the NaN

                series = temp[variable]
                label_encoder = LabelEncoder()
                temp[variable] = pd.Series(
                    label_encoder.fit_transform(series[series.notnull()]),
                    index=series[series.notnull()].index
                )
                self.encoders[variable] = label_encoder

        if self.imputer_type is None:
            pass
            #complete

        elif self.imputer_type=='iterativeimputer':
                
            if self.ii_estimator=='randomforestclassifier':
                self.ii_estimator=RandomForestClassifier()
            else:
                self.ii_estimator=LogisticRegression()

            imputer = IterativeImputer(estimator=self.ii_estimator,
                            initial_strategy=self.ii_initial_strategy,
                            skip_complete=self.ii_skip_complete)

            self.imputers[self.imputer_type] = imputer.fit(temp.loc[:,self.categories])
        
        elif self.imputer_type=='knnimputer':
            imputer = KNNImputer(n_neighbors=self.knn_n_neighbors)
            self.imputers[self.imputer_type] = imputer.fit(temp.loc[:,self.categories])

                
        return self

    def transform(self, X):

        Xt = X.loc[:,self.categories].copy()

        for variable in self.categories: #we need to label encoder preserving the NaN

            series = Xt[variable]
            label_encoder = LabelEncoder()
            Xt[variable] = pd.Series(
                label_encoder.fit_transform(series[series.notnull()]),
                index=series[series.notnull()].index
            )
            self.encoders[variable] = label_encoder

        Xt = Xt.apply(lambda series: pd.Series(
        LabelEncoder().fit_transform(series[series.notnull()]),
        index=series[series.notnull()].index
    ))

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
# var = ['Rear axle type txt','Gearbox syst txt','Engine tech txt','txt XFUEL','CAB txt','CJX','ASO-txt']
# ci = CategoricalMultivariateImputer(categories=var,imputer_type='iterativeimputer')

# mask = df.loc[:,var].isnull().any(axis=1)

# print(df[mask][var].head())

# (ci.fit_transform(df)).loc[mask,var].head()