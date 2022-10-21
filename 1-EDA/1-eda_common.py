#check nulls
import missingno
missingno.matrix(df)
missingno.heatmap(df)
missingno.dendrogram(df_train)

#check nulls in train/test
nulls_train = df_train.isnull().sum()/df_train.shape[1]
nulls_test = df_test.isnull().sum()/df_test.shape[1]

#plot null values of features sorted
df_train[null_cols].isnull().mean().sort_values(
    ascending=False).plot.bar(figsize=(10, 4))
plt.ylabel('Percentage of missing data')
plt.show()


fig,axs = plt.subplots(1,2,figsize=(20,12),sharey=True)
sns.barplot(nulls_train.index,nulls_train.values,ax=axs[0])
axs[0].set_xticklabels(nulls_train.index,rotation=45);
sns.barplot(nulls_test.index,nulls_test.values,ax=axs[1])
axs[1].set_xticklabels(nulls_test.index,rotation=45);


#Check influence of null values in target

nulls_train = df.isnull().sum()/df.shape[1]
null_cols = list(nulls_train[nulls_train!=0].index)

def analyse_na_value(df, var):

    df = df.copy()

    # let's make an interim variable that indicates 1 if the
    # observation was missing or 0 otherwise
    df[var] = np.where(df[var].isnull(), 1, 0)

    tmp = df.groupby(var)['failure'].agg(['mean', 'std'])

    # plot into a bar graph
    tmp.plot(kind="barh", y="mean", legend=False,
             xerr="std", title="failure", color='green')

    plt.show()

for col in null_cols:
    analyse_na_value(df, col)

## Continuous

import math
import warnings
warnings.filterwarnings("ignore")

def plot_distribution(dataset, cols=5, width=20, height=15, hspace=0.2, wspace=0.5):
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(width,height))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)
    rows = math.ceil(float(dataset.shape[1]) / cols)
    for i, column in enumerate(dataset.columns):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_title(column)
        if dataset.dtypes[column] == object:
            g = sns.countplot(y=column, data=dataset)
            substrings = [s.get_text()[:18] for s in g.get_yticklabels()]
            g.set(yticklabels=substrings)
            plt.xticks(rotation=25)
        else:
            g = sns.distplot(dataset[column])
            plt.xticks(rotation=25)
            
plot_distribution(df_test, cols=3, width=30, height=30, hspace=0.5, wspace=0.25)



# Plot a count of the categories from each categorical feature split by our prediction class:
#Note: use 'percentages=True' to normalize, specially for cat variables

def plot_bivariate_bar(dataset, hue, cols=5, width=20, height=15, hspace=0.2, wspace=0.5,percentages=False):
#     dataset = dataset.select_dtypes(include=[np.object])
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(width,height))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)
    rows = math.ceil(float(dataset.shape[1]) / cols)
    for i, column in enumerate(dataset.columns):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_title(column)
#         if dataset.dtypes[column] == np.object:
        if percentages==False:
            g = sns.countplot(y=column, hue=hue, data=dataset)
            substrings = [s.get_text()[:10] for s in g.get_yticklabels()]
            g.set(yticklabels=substrings)

        else:
            g = sns.histplot(x=dataset[column], hue=dataset[hue], stat="probability", multiple="fill", shrink=.8)

            
plot_bivariate_bar(df_train[int_cols+cat_cols+['failure']], hue='failure', cols=3, width=20, height=12, hspace=0.4, wspace=0.5,percentages=False)