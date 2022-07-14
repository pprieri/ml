#The assumption is that the number of rows (or the sorted total) of the clusters
#is almost the same between each run but the number assigned (0 or 1 or 2…) to each cluster
# is different at each run, so we have to reorganize each run to get the same cluster numbers
# to be able to add predictions correctly.



import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import PowerTransformer
from sklearn.mixture import BayesianGaussianMixture
data = pd.read_csv('../input/tabular-playground-series-jul-2022/data.csv')
best_data =[
'f_07','f_08', 'f_09', 'f_10',
'f_11', 'f_12', 'f_13', 'f_22',
'f_23', 'f_24', 'f_25','f_26',
'f_27', 'f_28']
#Data must be transformed into gaussian shape : (directly)
pt = PowerTransformer()
data_scaled = pt.fit_transform(data[best_data])
data_scaled = pd.DataFrame(data_scaled, columns = best_data)
data_scaled.head(3)

# Bizen made a hard voting (with mode), I changed it to a soft voting :

values = [0,1,2,3,4,5,6]
pred_test = pd.DataFrame(np.zeros((data_scaled.shape[0],7)), columns = values)


#3 steps for soft voting:
#we use value counts to calculate the order of size of clusters => this will help us identify despite
# they don't have the same label from one run (seed) to another
# we always assign 0 to the biggest, 1 to 2nd biggest and so on to 6
# for probabilities part, we need to change the name of columns and reorder them (using reindex)
#so that 0 becomes before 1, that comes before 2,...

for seed in tqdm(range(100)):

    df = pd.DataFrame(index = data.index)
    gmm = BayesianGaussianMixture(
            n_components=7,
            random_state = seed,
            tol = 0.01,
            covariance_type = 'full',
            max_iter = 100,
            n_init=3
          )

    # fitting and probability prediction
    gmm.fit(data_scaled)
    pred_seed = gmm.predict_proba(data_scaled) # predict_proba for probabilities

    # the clusters prediction for the current seed :
    MAX = np.argmax(pred_seed, axis=1)
    df[f'pred_{seed}'] = MAX

    # Sort of the prediction by same value of cluster (for addition of every seed)
    pred_keys = df[f'pred_{seed}'].value_counts().index.tolist()
    pred_dict = dict(zip(pred_keys, values))
    df[f'pred_{seed}'] = df[f'pred_{seed}'].map(pred_dict)

    pred_new = pd.DataFrame(pred_seed).rename(columns = pred_dict)
    pred_new = pred_new.reindex(sorted(pred_new.columns), axis=1)
    pred_test += pred_new # Soft voting by probabiliy addition

predictions = np.argmax(np.array(pred_test), axis=1)
