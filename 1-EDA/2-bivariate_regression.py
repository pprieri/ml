##
# For Regression, we could use a Pairplot between y (target) and x:
# => if relationship is not linear between Y and X, apply a transform (eg: square, log, exp)
# => if we plot 2 Xi features, we could color by Y’s value or size of points by Y’s value, to detect feature interactions
# => if evidence of influence of a 3rd Categorical variable, explore interaction variable (see green “Interaction” rectangle

## Continuous

#a) Heatmap_corr: check heatmap_corr.py

#b) MI and F scores for regression: check MI_and_f_scores.py

#c.1) Relplot and lmplots: 1 variable

feature = ''
hue = '' 
col = '' 
row = ''
extra_feat = ''
extra_feat_2 = ''

sns.lmplot(data=df,x=feature,y=target) #1 feature
sns.lmplot(data=df,x=feature,y=target,hue=hue,col=colmrow=row,robust=False)

#=> CHECK FOR OUTLIERS FAR FROM THE LINE

#c.2) Multi graphs lmplots

def plot_numeric(df,variable,log_variable=False,hue=None,log_hue=False,col=None,log_col=False,row=None,y=None,group_criteria=None,model=PLSRegression(),**kwargs):

    print('Univariate distributions\n')
    if log_variable==True:

        df[f'log_{variable}'] = np.log1p(df[variable])
        variable = f'log_{variable}'

    fix, axs = plt.subplots(1,2,figsize=(15,5))
    sns.kdeplot(data=df,x=variable,ax=axs[0])
    axs[0].set_title('Univariate distribution')
    sns.kdeplot(data=df,x=variable,hue=hue,ax=axs[1])
    axs[1].set_title(f'Univariate distribution by {hue}')

    if col is not None:
        
        sns.displot(data=df,x=variable,hue=hue,col=col,kind='kde')

        if row is not None:
            sns.displot(data=df,x=variable,hue=hue,col=col,row=row,kind='kde')

    if y is not None:
        
        model_1 = model.set_params(**kwargs)
        # model_1 = PLSRegression(n_components=1,scale=True)
        X_1 = df[variable].values.reshape(-1,1)
        y = df[target].values.reshape(-1,1)
        model_1.fit(X_1,y)
        
        r2_score_1 = np.round(model_1.score(X_1,y),2)
        CV_score_1 = np.round(get_score(df,variable,n_components=1),3)

        sns.lmplot(data=df,x=variable,y=target)
        ax = plt.gca()
        ax.set_title(f'R2 score {r2_score_1}')
        # ax.set_title(f'R2 score {r2_score_1} /  CV Score: {CV_score_1}')

        if hue is not None:
            model_2 = model.set_params(**kwargs)
            # model_2 = PLSRegression(n_components=2,scale=True)
            features = [variable,hue]
            X_2 = df[features]

            poly_transformer = PolynomialFeatures(degree=2,interaction_only=True,include_bias=False)
            X_poly= pd.DataFrame(poly_transformer.fit_transform(X_2),columns=poly_transformer.get_feature_names(features),index=X_2.index)

            model_2.fit(X_poly,y)
            r2_score_2 = np.round(model_2.score(X_poly,y),2)
            CV_score_2 = np.round(get_score(pd.concat([X_poly,df[[target,group_criteria]]],axis=1),X_poly.columns,n_components=2),2)

            sns.lmplot(data=df,x=variable,y=target,hue=hue)
            ax = plt.gca()
            ax.set_title(f'R2 score {r2_score_2}')
            # ax.set_title(f'R2 score {r2_score_2} /  CV Score: {CV_score_2}')


            if col is not None:

                if log_col==True:

                    df[f'log_{col}'] = np.log1p(df[col])
                    col = f'log_{col}'
                
                model_3 = model.set_params(**kwargs)
                # model_3 = PLSRegression(n_components=2,scale=True)
                features = [variable,hue,col]
                X_3 = df[features]

                poly_transformer = PolynomialFeatures(degree=2,interaction_only=True,include_bias=False)
                X_poly= pd.DataFrame(poly_transformer.fit_transform(X_3),columns=poly_transformer.get_feature_names(features),index=X_3.index)

                model_3.fit(X_poly,y)
                r2_score_3 = np.round(model_3.score(X_poly,y),2)
                CV_score_3 = np.round(get_score(pd.concat([X_poly,df[[target,group_criteria]]],axis=1),X_poly.columns,n_components=2),2)

                sns.lmplot(data=df,x=variable,y=target,hue=hue,col=col)
                ax = plt.gca()
                plt.suptitle(f'R2 score {r2_score_3}/CV Score: {CV_score_3}',y=1.05)     

#use
# variable = ''
# hue =''   
# col = ''
# row = ''
# plot_numeric(df,variable,log=False,hue=hue,col=col,row=None,y=target,model=PLSRegression(),n_components=2,scale=True))

#d) Plottly: check plottly.py

#e) Pairplot
sns.pairplot(df[features], 
             hue="target", 
             diag_kind="kde",
             size=4);

#f) Advanced: #Try PCA / t-SNE / UMAP to scatterplot a 2D-projection and color it by target value, it can give hints about possible strategies for dealing with subgroups.
#Note: t-SNE requires PCA before
#check tSNE.py