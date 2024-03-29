import matplotlib.pyplot as plt

%%time
pca = PCA()
Xt = pca.fit_transform(X)
plt.plot(pca.explained_variance_ratio_.cumsum())
plt.title('Principal components analysis')
plt.xlabel('Component')
plt.ylabel('Cumulative explained variance ratio')
plt.show()

print('Cumulative explained variance ratio for the first five components:', pca.explained_variance_ratio_.cumsum()[:5].round(2))
#plot PCA, controling the colors with  prop_cycle

df = 
scaler = 
df_scaled = 

# Compute the PCA
pca = PCA(n_components=3)
X_pca = pca.fit_transform(df_scaled)
# PCA projection, random drawing order of points
prop_cycle = plt.rcParams['axes.prop_cycle']
c = [prop_cycle.by_key()['color'][i % 10] for i in y]

plt.figure(figsize=(8, 8))
plt.scatter(X_pca[:,0], X_pca[:,1], s=1, label=f"Cluster {i}", c=c)
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
