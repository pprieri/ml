## Simple heatmap

import seaborn as sns
import matplotlib.pyplot as plt
mask = np.triu(np.ones_like(df.corr(), dtype=bool))

sns.set(rc={'figure.figsize':(24,20)})
sns.heatmap(df.corr(),annot=True,mask=mask,fmt='.2f')

##edited heatmap
cols = ''
import seaborn as sns
import matplotlib.pyplot as plt
mask = np.triu(np.ones_like(df_train[cols].corr(), dtype=bool))

sns.set(rc={'figure.figsize':(24,20)})
sns.heatmap(df_train[cols].corr(),annot=True,mask=mask,fmt='.2f')

## Clustered heatmap

import scipy
import scipy.cluster.hierarchy as sch

def cluster_corr(corr_array, inplace=False):
    """
    Rearranges the correlation matrix, corr_array, so that groups of highly 
    correlated variables are next to eachother 
    
    Parameters
    ----------
    corr_array : pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix 
        
    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix with the columns and rows rearranged
    """
    pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method='complete')
    cluster_distance_threshold = pairwise_distances.max()/2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, 
                                        criterion='distance')
    idx = np.argsort(idx_to_cluster_array)
    
    if not inplace:
        corr_array = corr_array.copy()
    
    if isinstance(corr_array, pd.DataFrame):
        return corr_array.iloc[idx, :].T.iloc[idx, :]
    return corr_array[idx, :][:, idx]

#sns.heatmap(df.corr()) # unclustered version
sns.heatmap(cluster_corr(df.corr()))