X = 
y = 
group_criteria = '' # 'Chassi ID'
id_col

groups = df.loc[X.index,group_criteria]
n_groups = len(groups.unique())
print(f'Nb of different {group_criteria}: {n_groups}')

predictions = pd.DataFrame(columns=[id_col,'predictions'])

for train_index, val_index in GroupKFold(n_splits=n_groups).split(X,y,groups=groups):
    
    X_train, X_validation = X.reset_index(drop=True).iloc[train_index], X.reset_index(drop=True).iloc[val_index]
    y_train, y_validation  = y.reset_index(drop=True).iloc[train_index], y.reset_index(drop=True).iloc[val_index]    

    model = PLSRegression(n_components=2,scale=True)
    # scaler = StandardScaler()

    model.fit(X_train,y_train)
    predictions = pd.concat([
        predictions,
        pd.DataFrame({id_col:X.reset_index().iloc[val_index][id_col].values,'predictions':model.predict(X_validation).ravel()})
    ],
    axis=0)
    
output = pd.merge(predictions,df.reset_index()[[id_col, target]],on=id_col,how='inner')
r2_score(output[target],output.predictions)