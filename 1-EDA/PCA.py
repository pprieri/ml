import matplotlib.pyplot as plt

#plot PCA, controling the colors with  prop_cycle

# Compute the PCA
pca = PCA(n_components=3)
p = pca.fit_transform(scaled)
# PCA projection, random drawing order of points
prop_cycle = plt.rcParams['axes.prop_cycle']
c = [prop_cycle.by_key()['color'][i % 10] for i in y]

plt.figure(figsize=(8, 8))
plt.scatter(p[:,0], p[:,1], s=1, label=f"Cluster {i}", c=c)
plt.xlabel('PCA[0]')
plt.ylabel('PCA[1]')
plt.legend()
plt.title('PCA projection')
plt.show()

## Heatmap of the components vs features
pca_components = pd.DataFrame(data=np.transpose(pca.components_), columns=X_pca.columns)
plt.figure(figsize=(25, 4))
sns.heatmap(pca_components.transpose(), cmap='RdBu_r', center = 0.0)

#Plot in 3d
fig = plt.figure(figsize=(10,8))
ax = plt.subplot(111, projection='3d', label="bla")
ax.scatter(x, y, z, s=40, c=PCA_ds["Clusters"], marker='o', cmap = cmap )
ax.set_title("The Plot Of The Clusters")
plt.show()
