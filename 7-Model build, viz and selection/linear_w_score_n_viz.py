id_col = '' #'ROW ID'
group_criteria='' #'Chassi ID'
model =  #PLSRegression()
type_cv= #'groupkfold' 'stratified' 'random'
log=False

def get_score(df,features,log=False,type_cv='groupkfold',group_criteria='Chassi ID',model=PLSRegression(),**kwargs):
    
    groups = df.loc[:,group_criteria]
    n_groups = len(groups.unique())

    if type(features)==str:
        features = [features]
    else:
        features = features
    
    if log==True and len(features)==1: #only one feature

        df[f'log_{features}'] = np.log1p(df[features])
        features = [f'log_{features}'] #we make it a list for step below

    group_criteria = group_criteria
    X = df.loc[:,features]
    X = X.fillna(0) #shortcut but should be corrected
    
    y = df.loc[:,target]
    groups = df.loc[:,group_criteria]
    n_groups = len(groups.unique())
    # print(f'Nb of different {group_criteria}: {n_groups}')

    predictions = pd.DataFrame(columns=[id_col,'predictions'])

    if type_cv=='groupkfold':
        cv=GroupKFold(n_splits=n_groups).split(X,y,groups=groups)
    elif type_cv=='stratified':
        cv=StratifiedKFold(n_splits=5,random_state=SEED,shuffle=True).split(X,groups)
    elif type_cv=='random':
        cv=KFold(n_splits=7,shuffle=True,random_state=SEED).split(X)

    for train_index, val_index in cv:
        
        X_train, X_validation = X.reset_index(drop=True).iloc[train_index], X.reset_index(drop=True).iloc[val_index]
        y_train, y_validation  = y.reset_index(drop=True).iloc[train_index], y.reset_index(drop=True).iloc[val_index]

        if len(features)==1:
            X_train = X_train.values.reshape(-1,1)
            X_validation = X_validation.values.reshape(-1,1)
            y_train = y_train.values.reshape(-1,1)
            y_validation = y_validation.values.reshape(-1,1)

        model = model.set_params(**kwargs)

        model.fit(X_train,y_train)
        predictions = pd.concat([
            predictions,
            pd.DataFrame({'ROW ID':X.reset_index().iloc[val_index]['ROW ID'].values,'predictions':model.predict(X_validation).ravel()})
        ],
        axis=0)
        
    output = pd.merge(predictions,df.reset_index()[['ROW ID', target]],on='ROW ID',how='inner')
    return r2_score(output[target],output.predictions)

#use
# get_score(df,variable,log=False,type_cv='groupkfold',group_criteria='Chassi ID',model=PLSRegression(),n_components=1,scale=True)
# get_score(df,variable,log=False,type_cv='stratified',group_criteria='Rear axle type txt',model=PLSRegression(),n_components=1,scale=True)