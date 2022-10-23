## TO DO: 
## Add a nested CV to include the 'best preprocessing' for each feature

from sklearn.model_selection import GroupKFold, cross_val_score
from matplotlib.ticker import MaxNLocator

n_iterations, backward = X.shape[1], False
selected_features=[]
if n_iterations != 0:
    n_features = X.shape[1]
    current_mask = np.zeros(shape=n_features, dtype=bool)
    history = []
    for _ in range(n_iterations):

    # for _ in range(n_iterations):
        candidate_feature_indices = np.flatnonzero(~current_mask)
        scores = {}
        for feature_idx in candidate_feature_indices:
            candidate_mask = current_mask.copy()
            candidate_mask[feature_idx] = True
            # X_new = X.values[:, ~candidate_mask if backward else candidate_mask]
            X_new = X.loc[:, ~candidate_mask if backward else candidate_mask]

            predictions = pd.DataFrame(columns=['ROW ID','predictions'])
            fold = 0
            for train_index, val_index in GroupKFold(n_splits=n_groups).split(X,y,groups=groups):
            # for train_index, val_index in StratifiedKFold(n_splits=5,random_state=SEED,shuffle=True).split(X_new,groups):

                # print(f'fold: {fold+1}')
                X_train, X_validation = X_new.reset_index(drop=True).iloc[train_index], X_new.reset_index(drop=True).iloc[val_index]
                y_train, y_validation  = y.reset_index(drop=True).iloc[train_index], y.reset_index(drop=True).iloc[val_index]

                # X_train, X_validation = X_new.iloc[train_index], X_new.iloc[val_index]
                # y_train, y_validation  = y.iloc[train_index], y.iloc[val_index]
                try: 
                    estimator.fit(X_train,y_train)
                        
                    predictions = pd.concat([
                            predictions,
                            pd.DataFrame({'ROW ID':X_new.reset_index().iloc[val_index]['ROW ID'].values,'predictions':estimator.predict(X_validation).ravel()})
                        ],
                        axis=0)
                except:
                    pass
                    # print(f'fold: {fold+1}')
                
            try:
                output = pd.merge(predictions,df.reset_index()[['ROW ID', target]],on='ROW ID',how='inner')
                score = r2_score(output[target],output.predictions)
            except:
                score=0
                
            scores[feature_idx] = score
            # scores[feature_idx] = cross_val_score(
            #     estimator=estimator,
            #     X=X_new,
            #     y=y,
            #     cv=StratifiedKFold(n_splits=7,random_state=SEED,shuffle=True),
            #     groups=groups,
            #     scoring='r2'
            # ).mean()
            #print(f"{str(X.columns[feature_idx]):30} {scores[feature_idx]:.3f}")
        new_feature_idx = max(scores, key=lambda feature_idx: scores[feature_idx])
        current_mask[new_feature_idx] = True
        history.append(scores[new_feature_idx])
        new = 'Deleted' if backward else 'Added'
        print(f'{new} feature: {str(X.columns[new_feature_idx]):30}'
              f' {scores[new_feature_idx]:.3f}')
        selected_features.append(str(X.columns[new_feature_idx]))

    print()
    plt.figure(figsize=(12, 6))
    plt.scatter(np.arange(len(history)) + (0 if backward else 1), history)
    plt.ylabel('R2')
    plt.xlabel('Features removed' if backward else 'Features added')
    plt.title('Sequential Feature Selection')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()

    if backward:
        current_mask = ~current_mask
    selected_columns = np.array(features)[current_mask]
    print(selected_columns)