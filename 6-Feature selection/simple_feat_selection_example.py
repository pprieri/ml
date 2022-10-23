from sklearn.feature_selection import SelectKBest, mutual_info_regression,f_regression
from functools import partial

SEED = 0
n_features = 5
X = df[features]
y = df_full_na[target]

# select = SelectKBest(score_func=mutual_info_regression, k=7)
select = SelectKBest(score_func=partial(mutual_info_regression,random_state=SEED), k=n_features)
X_selected= select.fit_transform(X, y)
X_selected.shape

mask = select.get_support()
X_selected.loc[:,mask].head(8)