##more useful when we have a lot of hyper param to optimize,
## even compared to Bayesian optimization (AutoML uses Bayesian opt except when n > 16 which switche to Random)
import scipy.stats as stats
from sklearn.utils.fixes import loguniform
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV

search_dict = {'kernel': ['linear', 'rbf'],
               'C': loguniform(1, 1000),
               'gamma': loguniform(0.0001, 0.1)
               }

scorer = 'accuracy'

search_func =  HalvingRandomSearchCV(estimator=svc,
                                     param_distributions=search_dict,
                                     resource='n_samples',
                                     max_resources=100,
                                     aggressive_elimination=True,
                                     scoring=scorer,
                                     n_jobs=-1,
                                     cv=5,
                                     random_state=0)
                                                 )

search_func.fit(X, y)

print (search_func.best_params_)
print (search_func.best_score_)

#source: https://github.com/PacktPublishing/The-Kaggle-Book/blob/main/chapter_08/basic-optimization-practices.ipynb
