import os
import warnings
from pathlib import Path
import math

# Data Manipulation 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas.api.types import CategoricalDtype
from IPython.display import display

# Visualization 
import matplotlib.pyplot as plt
import missingno
import seaborn as sns
from pandas.plotting import scatter_matrix
from mpl_toolkits.mplot3d import Axes3D

# Feature Selection and Encoding
from sklearn.feature_selection import RFE, RFECV, mutual_info_regression
from sklearn.svm import SVR

#Feature imputation
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize
from category_encoders import MEstimateEncoder

# Dimensionality Reduction / Clustering
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

#Model selection
from sklearn.model_selection import KFold, cross_val_score, train_test_split

# Machine learning 
from xgboost import XGBRegressor
import sklearn.ensemble as ske
from sklearn import datasets, model_selection, tree, preprocessing, metrics, linear_model
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
# import tensorflow as tf

# Grid and Random Search
import scipy.stats as st
from scipy.stats import randint as sp_randint
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# Metrics
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc

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

print("Setup done")