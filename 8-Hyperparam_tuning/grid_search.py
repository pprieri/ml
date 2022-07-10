##for algo with few param to optimize like SVM or Lasso/Ridge/Elastic Net


##SVM
from sklearn import model_selection
search_grid = [
               {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
               {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001],
               'kernel': ['rbf']}
               ]

scorer = 'accuracy'
search_func = model_selection.GridSearchCV(estimator=svc,
                                           param_grid=search_grid,
                                           scoring=scorer,
                                           n_jobs=-1,
                                           cv=5)
search_func.fit(X, y)

print (search_func.best_params_)
print (search_func.best_score_)

#source: https://github.com/PacktPublishing/The-Kaggle-Book/blob/main/chapter_08/basic-optimization-practices.ipynb
