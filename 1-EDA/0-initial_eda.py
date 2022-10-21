
## Check nulls, and see if there is a pattern/structure
df.isnull.sum(axis=1)
df.isnull(axis=0) 
# proportions of nulls
df.isnull().sum()/df.shape[1]

## Check uniques values
#By using nunique() and .loc try to see if columns with certain unique values proportion tend to be grouped
df.nunique(dropna=False)
df.nunique(dropna=False)[df.nunique(dropna=False)==1] #check features (columns) w/ constant value to drop them
df.nunique(axis=1)==1 #Check rows w/ constant values: (if there are duplicates, understand why)

#Check duplicated columns
dup_cols={}
for i,c1 in enumerate(df_train.columns):
    for c2 in df_train.columns[i+1:]:
        if c2 not in dup_cols and np.all(df_train[c1] == df_train[c2]):
            dup_cols[c2] = c1


import cPickle as pickle
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
df.mean().plot(style=’.’) / df.mean().sort_values().plot(style=’.’). We could use it to our imagination to create some interactions