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
