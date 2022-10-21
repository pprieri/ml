##
# For Regression, we could use a Pairplot between y (target) and x:
# => if relationship is not linear between Y and X, apply a transform (eg: square, log, exp)
# => if we plot 2 Xi features, we could color by Y’s value or size of points by Y’s value, to detect feature interactions
# => if evidence of influence of a 3rd Categorical variable, explore interaction variable (see green “Interaction” rectangle

## Continuous

#a) Heatmap_corr: check heatmap_corr.py

#b) MI and F scores for regression: check MI_and_f_scores.py

#c) Relplot and lmplots

feature = ''
hue = '' 
col = '' 
row = ''
extra_feat = ''
extra_feat_2 = ''

sns.lmplot(data=df,x=feature,y=target) #1 feature
sns.lmplot(data=df,x=feature,y=target,hue=hue,col=colmrow=row,robust=False)

#=> CHECK FOR OUTLIERS FAR FROM THE LINE

#d) Plottly: check plottly.py

#e) Pairplot
sns.pairplot(df[features], 
             hue="target", 
             diag_kind="kde",
             size=4);

#f) Advanced: #Try PCA / t-SNE / UMAP to scatterplot a 2D-projection and color it by target value, it can give hints about possible strategies for dealing with subgroups.
#Note: t-SNE requires PCA before
#check tSNE.py