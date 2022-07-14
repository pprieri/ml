sns.set(rc={'figure.figsize':(15,15)})
for i, column in enumerate(list(df.columns), 1):
    plt.subplot(5,6,i)
    p=sns.histplot(x=column,data=df.sample(1000),stat='count',kde=True,color='green')
