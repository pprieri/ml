import sklearn.metrics as metrics
for i in range(4,13):
    labels=KMeans(n_clusters=i,init="k-means++",random_state=200).fit(scaled_data).labels_
    print ("Silhouette score for k(clusters) = "+str(i)+" is "
           +str(metrics.silhouette_score(scaled_data,labels,metric="euclidean",sample_size=1000,random_state=200)))
