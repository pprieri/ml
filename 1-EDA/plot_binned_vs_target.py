def plot_binned_vs_target(df,feature,target,log=False,group_criteria=None,n_bins=10,strategy='uniform',model=PLSRegression(),**kwargs):


    if log==True: #only one feature

        df[f'log_{feature}'] = np.log1p(df[feature])
        feature = f'log_{feature}' #we make it a list for step below

    fig,axs=plt.subplots(1,3,figsize=(20,7))

    line = np.linspace(df[feature].min(),df[feature].max(),1000).reshape(-1,1)

    model = model.set_params(**kwargs)
    
    try:
        reg = model.fit(df[feature].values.reshape(-1,1),df[target])
    except: 
        model = model.set_params(n_components=1)
        reg = model.fit(df[feature].values.reshape(-1,1),df[target])

    axs[0].plot(df[feature],df[target],'o',c='steelblue')
    axs[0].plot(line,reg.predict(line),label="linear regression",c='orange')
    axs[0].set_xlabel(feature)
    axs[0].set_ylabel(target)
    axs[0].set_title(f'Score {np.round(reg.score(df[feature].values.reshape(-1,1),df[target]),2)}')
    #if there is a function giving cv score like 'get_score', use:
    # axs[0].set_title(f'Score {np.round(reg.score(df[feature].values.reshape(-1,1),df[target]),2)}, CV Score: {np.round(get_score(df,feature),3)}')


    ##

    kb = KBinsDiscretizer(n_bins=n_bins,strategy=strategy,encode='onehot-dense')
    X_binned=pd.DataFrame(kb.fit_transform(df[feature].values.reshape(-1,1)),columns=kb.get_feature_names_out([feature]),index=df.index)
    line_binned = kb.transform(line)
    model = model.set_params(**kwargs)
    reg = model.fit(X_binned,df[target])

    axs[1].plot(df[feature],df[target],'o',c='steelblue')
    axs[1].plot(line,reg.predict(line_binned),label="Binned feature",c='orange')
    axs[1].set_xlabel(feature)
    axs[1].set_ylabel(target)
    axs[1].legend(loc="best")
    axs[1].set_title(f'R2 score {np.round(reg.score(X_binned,df[target]),2)}, Q2 Score: {np.round(get_score(pd.concat([X_binned,df[[target,group_criteria]]],axis=1),X_binned.columns,n_components=1),3)}') #we use concat to add both target and group criteria (in this exercice we use GroupKfold)

    ##

    X_product=pd.concat([X_binned,X_binned.mul(df[feature],axis=0)],axis=1)
    line_product = np.hstack([line_binned,line * line_binned])
    model = model.set_params(**kwargs)
    reg = model.fit(X_product,df[target])

    axs[2].plot(df[feature],df[target],'o',c='steelblue')
    axs[2].plot(line,reg.predict(line_product),label="Binned feature + feature * Binned feature",c='orange')
    axs[2].set_xlabel(feature)
    axs[2].set_ylabel(target)
    axs[2].legend(loc="best")
    axs[2].set_title(f'R2 score {np.round(reg.score(X_product,df[target]),2)}, Q2 Score: {np.round(get_score(pd.concat([X_product,df[[target,group_criteria]]],axis=1),X_product.columns,n_components=1),3)}')
    
#use

TARGET = ''
feature = ''

model_args = {n_components:2,scale:True}
# plot_binned_vs_target(df,feature=feature,target=TARGET,log=True,n_bins=10,strategy='uniform',**model_args)