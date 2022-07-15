##More control

nr_rows=6
nr_cols=4

bins_min=-5
bins_max=5
fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(16, 20))
axs = axs.ravel() ##transform from 2D (nr rows, nr cols) into nr rows * nr cols, which is easier to iterate
float_columns = [col for col in data.columns if data[col].dtype == 'float']
for ax, f in zip(axs, float_columns):
    for i in range(n_clusters):
        h, edges = np.histogram(data[f][y == i], bins=np.linspace(bins_min, bins_max, 26))
        ax.plot((edges[:-1] + edges[1:]) / 2, h, label=f"Cluster {i}", lw=3) #to center the value in bins we do : (edges[:-1] + edges[1:]) / 2
    ax.set_title(f)
axs[-2].axis('off')
axs[-1].axis('off')
plt.suptitle('Histograms of the float features of the 7 clusters', y=0.95, fontsize=20)
plt.show()

## Easier way

fig, axes = plt.subplots(nrows = 5, ncols = 6, figsize = (22,12))
for i, col in enumerate(aux[1:].columns):
    sns.kdeplot(data = aux, x = col, hue = 'clusters', palette = 'Spectral', ax = axes[(i // 6)][(i % 6)])

fig.tight_layout(h_pad=1.0, w_pad=0.5)
