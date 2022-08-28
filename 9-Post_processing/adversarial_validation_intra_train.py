import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict, train_test_split

gkf = GroupKFold(n_splits=5)
# cv = KFold(n_splits=5, shuffle=True, random_state=1)

summary_features = pd.DataFrame(index=X.columns)

for f, ( train_i, valid_i ) in enumerate( gkf.split(X,groups=X['product_code']) ):

    print("# fold {}".format( f + 1))

    X_train = X.iloc[train_i].drop(['product_code','attribute_0','attribute_1','attribute_2','attribute_3'],axis=1)
    X_valid = X.iloc[valid_i].drop(['product_code','attribute_0','attribute_1','attribute_2','attribute_3'],axis=1)

    y_train = y.iloc[train_i]
    y_valid = y.iloc[valid_i]
    
#     ohe_feat = ['attribute_0','attribute_1']
#     output_cols_oh= []
#     X_train,X_valid = oh_encode(X_train,X_valid,ohe_feat=ohe_feat,output_cols=output_cols_oh)   
    
    X_concat = X_train.append(X_valid)
    y_concat = [0] * len(X_train) + [1] * len(X_valid)
    
    X_concat_train, X_concat_test, y_concat_train, y_concat_test = train_test_split(X_concat, y_concat, test_size=0.33, random_state=42)
    
    model = RandomForestClassifier(n_estimators = 100,n_jobs=-1,random_state=0)
    
    model.fit(X_concat_train,y_concat_train)
    preds = model.predict_proba(X_concat_test)
    
    roc = roc_auc_score(y_true=y_concat_test, y_score=preds[:,1])
    
        
    print(f'ROC: {roc} \n')
    
    summary_features = pd.concat([summary_features,pd.Series(data=model.feature_importances_,index=X_train.columns)],axis=1)

summary_features.columns = ['Fold_'+str(i) for i in range(1,6)]
summary_features['Folds_mean']=summary_features.mean(axis=1)
summary_features.sort_values(by='Folds_mean',ascending=False)