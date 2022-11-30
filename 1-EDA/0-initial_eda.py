
## Check nulls, and see if there is a pattern/structure
df.isnull().sum(axis=1)
df.isnull().sum(axis=0) 
# proportions of nulls
df.isnull().sum()/df.shape[1]

## Check uniques values
#By using nunique() and .loc try to see if columns with certain unique values proportion tend to be grouped
df.nunique(dropna=False)
df.nunique(dropna=False)[df.nunique(dropna=False)==1] #check features (columns) w/ constant value to drop them
df.nunique(axis=1)==1 #Check rows w/ constant values: (if there are duplicates, understand why)

##
# The predictions should be probabilities between 0 and 1, but X contains some values outside this range.
# The labels are all 0 or 1, as is to be expected.
# There are no null values.

print('Minimum and maximum X values:', X.min(), X.max())
print('Unique labels:               ', np.unique(y))
print('Null values in X and y:      ', np.isnan(X).sum(), np.isnan(y).sum())

print(f"Values below zero:           {(X < 0).sum()}")
print(f"Values above one:           {(X > 1).sum()}")
print(f"Rows containing outliers:    {((X < 0) | (X > 1)).any(axis=1).sum()} of {X.shape[0]}")
print(f"Columns containing outliers:  {((X < 0) | (X > 1)).any(axis=0).sum()} of {X.shape[1]}")
#Check duplicated columns
import cPickle as pickle
dup_cols={}
for i,c1 in enumerate(df_train.columns):
    for c2 in df_train.columns[i+1:]:
        if c2 not in dup_cols and np.all(df_train[c1] == df_train[c2]):
            dup_cols[c2] = c1

pickle.dump(dup_cols, open(‘dup_cols.p’,’w’),protocol=pickle.HIGHEST_PROTOCOL)
traintest_drop(dup_cols.keys(),axis=1,inplace=True)

#Generate descriptive statistics. For numeric, check percentiles
df.describe(include='all')

#c) Hackish, competition-like graps

#Plot index vs features to check if dataset was shuffled and for constant features (that we can omit)
#c.1)Row Index vs feature value: 
plt.plot(x,’.’)
#Row index vs feature value + color with y:
plt.scatter(range(len(x), x, c=y). To check if data is separated by class for ex
#Row index vs Feature index + color with Feature value
#- Plot index vs feature statistics => to detect if there are groups (ex: Kaggle compt course). 
df.mean().plot(style=’.’) / df.mean().sort_values().plot(style=’.’)
# This last one is very useful We could use it to our imagination to create some interactions