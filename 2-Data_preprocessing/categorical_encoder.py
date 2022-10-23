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


class CategoricalEncoder(BaseEstimator, TransformerMixin):


    def __init__(self,categories='auto',imputer_type=None,
                encoding_type='oh',k=1,f=1,noise_level=0,random_state=None):

        #These below are parameters
        if type(categories)==str and categories!='auto':
            self.categories = [categories] #in case we have only one, we convert it into a list
        else:
            self.categories = categories
            
        self.encoding_type=encoding_type
        self.k = k
        self.f = f
        self.noise_level = noise_level
        self.random_state = random_state
        self.imputer_type = imputer_type

        #These below are learnt attributes
        self.imputers = dict()
        self.encodings = dict()
        self.reverse_encodings = dict()
        self.dict_reverse_encodings=dict()
        self.output_ohe=dict()
        self.prior = None

        
    
    def add_noise(self, series, noise_level):
        return series * (1 + noise_level *
                         np.random.randn(len(series)))
    
    def fit(self,X,y=None):

        if type(self.categories)=='auto':
            self.categories = np.where(X.dtypes == type(object()))[0]
        temp = X.loc[:, self.categories].copy()
        temp['target'] = y

        X = X.copy()

        X.loc[:, self.categories]=X.loc[:, self.categories].fillna('Missing') #to be removed once we have a 'Simple imputer' with fill_value='Missing'



        if self.encoding_type=='oh':
            
            for variable in self.categories:
                    # print(variable)
                    categories_oh = [np.sort(X[variable].unique())] #note the extra brackets [], which otherwise won't work. The correct way to construct it is
                                                                    # ohe = OneHotEncoder(categories=[['material_5', 'material_7'],
                                                                    # ['material_5', 'material_6', 'material_8']],
                                                                    #drop='first', sparse=False, handle_unknown='ignore')

                    ohe=(OneHotEncoder(handle_unknown='ignore',drop='first',
                        sparse=False,categories=categories_oh))
                    
                    self.output_ohe[variable]=[str(variable)+f'_{label}' for label in categories_oh[0][1:]] #because of drop='First'
                    self.encodings[variable] = ohe.fit(X.loc[:,[variable]]) #double brackets to have 2d array = (n,1), expected by OneHotEncoder class

        elif self.encoding_type=='target_enc':
            
            self.prior = np.mean(y)
            for variable in self.categories:
                avg = (temp.groupby(by=variable)['target']
                            .agg(['mean', 'count']))
                # Compute smoothing
                smoothing = (1 / (1 + np.exp(-(avg['count'] - self.k) /
                                self.f)))
                # The bigger the count the less full_avg is accounted
                target_enc = dict(self.prior * (1 -
                                    smoothing) + avg['mean'] * smoothing)
                self.encodings[variable] = target_enc
                self.reverse_encodings[variable] = {i:k for k,i in self.encodings[variable].items()} 

        elif self.encoding_type=='label_ordered':

            for variable in self.categories:

                # temp[variable]=temp[variable].fillna('Missing')
                ordered_labels = temp.groupby(variable)['target'].mean().sort_values().index.values
                ordinal_mapping = {k: i for i, k in enumerate(ordered_labels, 0)}
                ordinal_mapping['Other']=-1
                self.encodings[variable] = ordinal_mapping
                self.reverse_encodings[variable] = {i:k for k,i in self.encodings[variable].items()} 
                
                 
        return self

    def transform(self, X):
        Xt = X.copy()
        
        for variable in self.categories:
            
            if self.encoding_type=='oh':
                
                X_oh = pd.DataFrame(data=(self.encodings[variable]).transform(Xt[[variable]]),index=Xt.index,columns=self.output_ohe[variable])
                # df2_variable_oh = pd.DataFrame(ohe.transform(df2[[variable]]),index=df2.index,columns=output_ohe)
                Xt = pd.concat([Xt,X_oh],axis=1).drop(variable,axis=1)

            else:

                Xt[variable].replace(self.encodings[variable],
                                    inplace=True)

                if self.encoding_type=='label_ordered':
                
                    unknown_value = {value:self.encodings[variable]['Other'] for value in
                        X[variable].unique()
                        if value not in
                        self.encodings[variable].keys()}
                    
                    if len(unknown_value) > 0:
                        Xt[variable].replace(unknown_value, inplace=True)
                    Xt[variable] = Xt[variable].astype(float)

                if self.encoding_type=='target_enc':

                    unknown_value = {value:self.prior for value in
                                    X[variable].unique()
                                    if value not in
                                    self.encodings[variable].keys()}
                    if len(unknown_value) > 0:
                        Xt[variable].replace(unknown_value, inplace=True)
                    Xt[variable] = Xt[variable].astype(float)

                    if self.noise_level > 0:
                        if self.random_state is not None:
                            np.random.seed(self.random_state)
                        Xt[variable] = self.add_noise(Xt[variable],
                                                    self.noise_level)
        return Xt

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
    
# #ex use
# var = ['Rear axle type txt','RWX']
# cat_transf = CategoricalEncoder(encoding_type='oh',categories=var)
# df2 = cat_transf.fit_transform(df,df[target])

# var = 'Rear axle type txt'
# pd.concat([df['Rear axle type txt'],df2[cat_transf.output_ohe[var]]],axis=1).head(5)

# var = ['Rear axle type txt','RWX']
# cat_transf = CategoricalEncoder(encoding_type='label_ordered' ,categories=var)
# df2 = cat_transf.fit_transform(df,df[target])
# pd.concat([df['Rear axle type txt'],df2['Rear axle type txt']],axis=1).head(5)

# cat_transf = CategoricalEncoder(encoding_type='target_enc',categories=var)
# df2 = cat_transf.fit_transform(df,df[target])
# pd.concat([df['Rear axle type txt'],df2['Rear axle type txt']],axis=1).head(5)