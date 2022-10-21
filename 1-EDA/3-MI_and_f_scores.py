import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression,f_regression

features = []

num_cols=df[features].select_dtypes('number').columns.tolist()

cat_cols=df[features].select_dtypes('object').columns.tolist()
assert len(features) == len(num_cols)+len(cat_cols)

for col in num_cols:

    median_col = df_univariate[col].median()

    df_univariate[col]=df_univariate[col].fillna(median_col)


dict_reverse_encodings=dict()

for col in cat_cols:

    df_univariate[col]=df_univariate[col].fillna('Missing')

    ordered_labels = pd.concat([df_univariate[col],df[target]],axis=1).groupby(col)[target].mean().sort_values().index.values
    ordinal_mapping = {k: i for i, k in enumerate(ordered_labels, 0)}
    ordinal_mapping['Other']=-1
    reverse_ordinal_mapping = {i:k for k,i in ordinal_mapping.items()}
    
    dict_reverse_encodings[col]=reverse_ordinal_mapping #save for later

    df_univariate[col] = df_univariate[col].map(ordinal_mapping)

f_test, _ = f_regression(X, y)
f_test /= np.max(f_test)

mi = mutual_info_regression(X, y)
mi /= np.max(mi)

fig = plt.figure(figsize=(20, 15))
for i in range(X.shape[1]):
    plt.subplot(5, 5, i + 1)
    feature = X.loc[:,mask].columns[i]
    if feature in cat_cols:
        feature_decoded_sorted = pd.Series(X[:, i],index=df.index).sort_values().map(dict_reverse_encodings[X.loc[:,mask].columns[i]])
        # feature_decoded_sorted = df[col].sort_values()
        idx_feature_decoded_sorted = feature_decoded_sorted.index
        plt.scatter(feature_decoded_sorted, y.loc[idx_feature_decoded_sorted], edgecolor="black", s=20)
    else:
        plt.scatter(X[:, i], y, edgecolor="black", s=20)

    plt.xlabel(feature, fontsize=14)
    plt.xticks(rotation = 45)
    if i == 0:
        plt.ylabel("$y$", fontsize=14)
    plt.title("F-test={:.2f}, MI={:.2f}".format(f_test[i], mi[i]), fontsize=16)
fig.tight_layout()  
plt.show()