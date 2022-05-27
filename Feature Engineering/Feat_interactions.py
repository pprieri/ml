#To Do: generalize to 'n_order'

def binary_interactions(df,model):

    from sklearn.tree import DecisionTreeClassifier
    df = df.copy()

    #initialize interaction matrix
    if isinstance(df,pd.DataFrame):
            df_feat_interactions = pd.DataFrame(index=df.columns,columns=df.columns).fillna(0)

    elif isinstance(df,np.ndarray):
            df_feat_interactions = pd.DataFrame(index=range(len(X_train.T)),columns=range(len(X_train.T))).fillna(0)

    #fill interaction matrix
    #A) if only a single DT model:
    if isinstance(model,DecisionTreeClassifier):
        feat_tree = model.tree_.feature[model.tree_.feature!=-2]
        for i in range(len(feat_tree)-1):
            if isinstance(df,pd.DataFrame):

                df_feat_interactions.loc[df.columns[feat_tree[i]],df.columns[feat_tree[i+1]]] +=1

            else:
                df_feat_interactions.loc[feat_tree[i],feat_tree[i+1]]+=1

    #B) Ensemble of DT:
    else:
        for e in range(len(model.estimators_)):
            feat_tree = model.estimators_[e].tree_.feature[model.estimators_[e].tree_.feature!=-2]
            for i in range(len(feat_tree)-1):
                if isinstance(df,pd.DataFrame):

                    df_feat_interactions.loc[df.columns[feat_tree[i]],df.columns[feat_tree[i+1]]] +=1

                else:
                    df_feat_interactions.loc[feat_tree[i],feat_tree[i+1]]+=1

    #consolidate interactions in a Series
    feat_indexes = [(row,column)
            for row in df_feat_interactions.index
            for column in df_feat_interactions.columns if row<column]

    feat_summary = pd.Series(dtype='int32',index=feat_indexes)

    for i,j in feat_indexes:
        feat_summary.loc[[(i,j)]]=df_feat_interactions.loc[i,j] + df_feat_interactions.loc[j,i]

    #normalize values
    feat_summary=feat_summary/feat_summary.sum()

    return feat_summary.sort_values(ascending=False)

def trinary_interactions(df,model):

    from sklearn.tree import DecisionTreeClassifier
    df = df.copy()

    #initialize interaction matrix
    if isinstance(df,pd.DataFrame):

            features=df.columns
#             df_feat_interactions = pd.DataFrame(index=df.columns,columns=df.columns).fillna(0)

    elif isinstance(df,np.ndarray):
            features=range(len(df.T))
#             df_feat_interactions = pd.DataFrame(index=range(len(df.T)),columns=range(len(df.T))).fillna(0)

    iterables = [features,features,features]

    df_feat = (
                pd.DataFrame(
                            index=pd.MultiIndex.from_product(
                            iterables, names=["first", "second","third"]
                            ),dtype='float32'
                            )
              )

    df_feat['interactions']=0

    #fill interaction matrix
    #A) if only a single DT model:
    if isinstance(model,DecisionTreeClassifier):
        feat_tree = model.tree_.feature[model.tree_.feature!=-2]
        for i in range(len(feat_tree)-2):

            triplet = feat_tree[i:i+3]
            sorted_triplet = np.sort(triplet) # [3,2,3] becomes [2,3,3]

            if isinstance(df,pd.DataFrame):

                sorted_features = df.columns[sorted_triplet]
                df_feat.loc(axis=0)[sorted_features[0],sorted_features[1],sorted_features[2]]+=1

            else:

                df_feat.loc(axis=0)[sorted_triplet[0],sorted_triplet[1],sorted_triplet[2]]+=1

#             if isinstance(df,pd.DataFrame):

#                 df_feat_interactions.loc[df.columns[feat_tree[i]],df.columns[feat_tree[i+1]]] +=1

#             else:
#                 df_feat_interactions.loc[feat_tree[i],feat_tree[i+1]]+=1
#                 df_feat.loc(axis=0)[sorted_triplet[0],sorted_triplet[1],sorted_triplet[2]]+=1

    #B) Ensemble of DT:
    else:
        for e in range(len(model.estimators_)):
            feat_tree = model.estimators_[e].tree_.feature[model.estimators_[e].tree_.feature!=-2]
            for i in range(len(feat_tree)-2):

                triplet = feat_tree[i:i+3]
                sorted_triplet = np.sort(triplet) # [3,2,3] becomes [2,3,3]

                if isinstance(df,pd.DataFrame):

                    sorted_features = np.sort(df.columns[sorted_triplet])

                    df_feat.loc(axis=0)[sorted_features[0],sorted_features[1],sorted_features[2]]+=1

                else:

                    df_feat.loc(axis=0)[sorted_triplet[0],sorted_triplet[1],sorted_triplet[2]]+=1

#                 if isinstance(df,pd.DataFrame):

#                     df_feat_interactions.loc[df.columns[feat_tree[i]],df.columns[feat_tree[i+1]]] +=1

#                 else:
#                     df_feat_interactions.loc[feat_tree[i],feat_tree[i+1]]+=1

    df_feat.interactions = df_feat.interactions/df_feat.interactions.sum(axis=0)

    return df_feat.sort_values(by='interactions',ascending=False)
