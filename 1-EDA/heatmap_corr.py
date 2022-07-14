import seaborn as sns
import matplotlib.pyplot as plt
mask = np.triu(np.ones_like(df.corr(), dtype=bool))

sns.set(rc={'figure.figsize':(24,20)})
sns.heatmap(df.corr(),annot=True,mask=mask,fmt='.2f')
