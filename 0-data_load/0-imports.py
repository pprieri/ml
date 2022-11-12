import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_rows', 600)
pd.set_option('display.max_columns', 50)
import numpy as np # linear algebra

import os
import warnings
from pathlib import Path
import math

# Data Manipulation 

from pandas.api.types import CategoricalDtype
from IPython.display import display

# Visualization 
import matplotlib.pyplot as plt
import missingno
import seaborn as sns
from pandas.plotting import scatter_matrix
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px

# Feature Selection and Encoding
from sklearn.feature_selection import RFE, RFECV, mutual_info_regression
from sklearn.svm import SVR
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, MissingIndicator, KNNImputer, IterativeImputer

#Feature imputation and 
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, PowerTransformer, QuantileTransformer, StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, PolynomialFeatures, KBinsDiscretizer

from category_encoders import MEstimateEncoder

# Dimensionality Reduction / Clustering
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

#CV
from sklearn.model_selection import train_test_split,cross_val_score, GroupKFold, LeaveOneOut, StratifiedKFold, KFold
# !pip install scikit-multilearn
#from skmultilearn.model_selection import IterativeStratification

# Machine learning 
from sklearn.base import BaseEstimator, TransformerMixin
from xgboost import XGBRegressor
from sklearn.cross_decomposition import PLSRegression
import sklearn.ensemble as ske
from sklearn import datasets, model_selection, tree, preprocessing, metrics, linear_model
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, SGDClassifier,LinearRegression,Lasso, BayesianRidge, LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.pipeline import FeatureUnion, make_pipeline
# import tensorflow as tf

# Grid and Random Search
import scipy.stats as st
from scipy.stats import randint as sp_randint
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# Metrics
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc, r2_score, mean_absolute_error

# Set Matplotlib defaults
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)

# Plot the Figures Inline
%matplotlib inline

# Mute warnings
warnings.filterwarnings('ignore')

#!pip install scikit-learn-intelex
from sklearnex import patch_sklearn
patch_sklearn()

print("Setup done")