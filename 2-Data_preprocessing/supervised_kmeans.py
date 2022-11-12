#from this notebook:https://www.kaggle.com/code/xuedaolao/experimentation-3/edit

class KMeansFeaturizer:

    def __init__(self, k=100, target_scale=1, random_state=None):
        self.k = k
        self.target_scale = target_scale
        self.random_state = random_state

    def fit(self, X, y=None):
        """Runs k-means on the input data and finds centroids.
        """
        if y is None:
        # No target variable, just do plain k-means
            km_model = KMeans(n_clusters=self.k,
            n_init=20,
            random_state=self.random_state)
            km_model.fit(X)

            self.km_model_ = km_model
            self.cluster_centers_ = km_model.cluster_centers_
            return self

        # There is target information. Apply appropriate scaling and include
        # it in the input data to k-means.
        data_with_target = np.hstack((X, y[:,np.newaxis]*self.target_scale))

        # Build a pre-training k-means model on data and target
        km_model_pretrain = KMeans(n_clusters=self.k,
        n_init=20,
        random_state=self.random_state)
        km_model_pretrain.fit(data_with_target)

        # Run k-means a second time to get the clusters in the original space
        # without target info. Initialize using centroids found in pre-training.
        # Go through a single iteration of cluster assignment and centroid
        # recomputation.
        km_model = KMeans(n_clusters=self.k,
        init=km_model_pretrain.cluster_centers_[:,:-1],
        n_init=1,
        max_iter=1)
        km_model.fit(X)

        self.km_model = km_model
        self.cluster_centers_ = km_model.cluster_centers_
        return self

    def transform(self, X, y=None):
        """Outputs the closest cluster ID for each input data point.
        """
        clusters = self.km_model.transform(X)
        return pd.DataFrame(clusters,index=X.index)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X, y)
    
    def predict(self,X,y=None):
        
        return self.km_model.predict(X)