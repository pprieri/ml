## Categorical

#A) Univariate

feature = ''
target = ''

plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(20,1))
sns.countplot(y=target, data=df)


#target mean
df.target.mean()

#mean by features
df.groupby(features).target.mean()

#mean and size by feature
pd.concat([df.groupby(feature).failure.mean(),
            df.groupby(features).failure.size()
          ],axis=1)

#B) Bivariate

#pointplot
sns.pointplot(x=feature,y=target,data=df)

#boxplot
sns.boxplot(x=feature,y=target,data=df)

#histogram of continuous feature colored by target

plt.style.use('seaborn-whitegrid')
fig = plt.figure(figsize=(20,5)) 
sns.histplot(x=feature,hue=target,data=df)

#interactions of 2 categoricals with target

feature_1=''
feature_2=''

#a)Pointplot
sns.pointplot(data=df_train,x=feature_1, y=target,hue=feature_2)

#b)Boxplot


plt.style.use('seaborn-whitegrid')
g = sns.FacetGrid(df_train, col=feature_1, size=4, aspect=.7)
g = g.map(sns.boxplot, target, feature_2)

#interaction of 2 continuous with target
sns.lmplot(data=df_train,x=feature_1, y=feature_2,hue=target)