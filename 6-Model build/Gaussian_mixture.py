## Analyze which covariances works best:

fig, ax = plt.subplots(figsize=(8,4))

n_components = range(2, 12)
covariance_types = ['spherical', 'tied', 'diag', 'full']

colores = ['red','blue','orange','green']

j = 0
for covariance_type in covariance_types:
    valores_bic = []
    valores_aic = []

    for i in n_components:
        modelo = GaussianMixture(n_components=i, covariance_type=covariance_type)
        modelo = modelo.fit(scaled_data)
        valores_bic.append(modelo.bic(scaled_data))
        valores_aic.append(modelo.aic(scaled_data))

    ax.plot(n_components, valores_bic, label=covariance_type, color = colores[j])
    ax.plot(n_components, valores_aic, color = colores[j])
    j += 1

ax.set_title("Valores AIC / BIC")
ax.set_xlabel("Número componentes")
ax.legend();

## once defined covariance, follow below

components_min, components_max = 4, 25

result_list = []
for n_components in range(components_min, components_max):
    for seed in range(10):
        gm = GaussianMixture(n_components=n_components, random_state=seed, verbose=0, n_init=1)
        y = gm.fit_predict(scaled)
        bic = gm.bic(scaled)
        aic = gm.aic(scaled)
        #print(f"{n_components:2} {bic:16.5f} {aic:16.5f}")
        result_list.append((n_components, seed, bic, aic, y, gm))

results = pd.DataFrame(result_list, columns=['n_components', 'seed', 'bic', 'aic', 'y', 'gm'])
results = results.set_index(['n_components', 'seed'])

#Having the models ("gm") column allows also to select the best model according to a given metric (ex: lowest bic), like so:

gm = results.loc[7, 1].gm
y = results.loc[7, 1].y
n_clusters = len(gm.means_)

#Compare BIC
# BIC diagram
plt.figure(figsize=(16, 5))
plt.scatter(results.reset_index().n_components, results.bic)
m = results.reset_index().groupby('n_components').bic.min()
plt.plot(m.index, m)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.ylabel('BIC score')
plt.xlabel('Number of clusters')
plt.title('BIC: Can you see an elbow? Is it really at 7 components?', fontsize=20)
plt.show()

#Compare clusters
def compare_clusterings(y1, y2, title=''):
    """Show the adjusted rand score and plot the two clusterings in color"""
    ars = adjusted_rand_score(y1, y2)
    n1 = y1.max() + 1
    n2 = y2.max() + 1
    argsort = np.argsort(y1*100 + y2) if n1 >= n2 else np.argsort(y2*100 + y1)
    plt.figure(figsize=(16, 0.5))
    for i in range(6, 11):
        plt.scatter(np.arange(len(y1)), np.full_like(y1, i), c=y1[argsort], s=1, cmap='tab10')
    for i in range(5):
        plt.scatter(np.arange(len(y2)), np.full_like(y2, i), c=y2[argsort], s=1, cmap='tab10')
    plt.gca().axis('off')
    plt.title(f'{title}\nAdjusted Rand score: {ars:.5f}')
    plt.savefig(title + '.png', bbox_inches='tight')
    plt.show()

compare_clusterings(results.loc[8, 0].y, 7 - results.loc[8, 0].y, '8 clusters vs. relabeling')

#diff sizes:
compare_clusterings(results.loc[7, 1].y, results.loc[8, 1].y, '7 clusters vs. 8 clusters')
compare_clusterings(results.loc[7, 0].y, results.loc[8, 0].y, '7 clusters vs. 8 clusters')

#across seeds:
compare_clusterings(results.loc[7, 1].y, results.loc[10, 2].y, '7 clusters vs. 10 clusters')
compare_clusterings(results.loc[7, 2].y, results.loc[10, 0].y, '7 clusters vs. 10 clusters')

#Histograms colored by Cluster

fig, axs = plt.subplots(6, 4, figsize=(16, 20))
axs = axs.ravel()
float_columns = [col for col in data.columns if data[col].dtype == 'float']
for ax, f in zip(axs, float_columns):
    for i in range(n_clusters):
        h, edges = np.histogram(data[f][y == i], bins=np.linspace(-5, 5, 26))
        ax.plot((edges[:-1] + edges[1:]) / 2, h, label=f"Cluster {i}", lw=3)
    ax.set_title(f)
axs[-2].axis('off')
axs[-1].axis('off')
plt.suptitle('Histograms of the float features of the 7 clusters', y=0.95, fontsize=20)
plt.show()

#Alternative: graph only the mean by features
# Cluster means for every feature
# Features where all cluster means coincide tend to be useless
plt.figure(figsize=(16, 4))
for i in range(gm.means_.shape[0]):
    plt.scatter(np.arange(scaled.shape[1]), gm.means_[i])
plt.xticks(ticks=np.arange(scaled.shape[1]), labels=scaled.columns)
plt.title('Cluster means')
plt.show()

## Compute the PCA
pca = PCA(n_components=3)
p = pca.fit_transform(scaled)
# PCA projection, random drawing order of points
c = [prop_cycle.by_key()['color'][i % 10] for i in y]

plt.figure(figsize=(8, 8))
plt.scatter(p[:,0], p[:,1], s=1, label=f"Cluster {i}", c=c)
plt.xlabel('PCA[0]')
plt.ylabel('PCA[1]')
plt.legend()
plt.title('PCA projection')
plt.show()

#Compare 2 by 2 features colored by clusters
# Projections of clusters to feature pairs
# We see several strange shapes. The strange shapes suggest that we haven't yet found the best clustering.
for f, g in [(f"f_{i:02d}", f"f_{j:02d}") for i in range(22, 29) for j in range(i+1, 29)]:
    fig, axs = plt.subplots(1, n_clusters, figsize=(16, 5), sharex=True, sharey=True)
    for i in range(n_clusters):
        axs[i].scatter(data[f][y == i], data[g][y == i], s=1, label=f"Cluster {i}", color=prop_cycle.by_key()['color'][i % 10])
        axs[i].set_xlabel(f)
        axs[i].set_aspect('equal')
    axs[0].set_ylabel(g)
    if f == 'f_24' and g == 'f_25': plt.savefig(f'{f}_{g}.png')
    plt.show()

##Plot covariance matrix

# A covariance heatmap per cluster
for i in range(len(gm.covariances_)):
    print(f'Cluster {i}')
    plt.figure(figsize=(16, 16))
    sns.heatmap(gm.covariances_[i], annot=True, fmt='.1f', center=0)
    plt.show()
