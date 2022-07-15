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

## Easier way: continuous features (but also work for discrete)
## a) with histo
cont_feats=[f'f_0{i}' for i in range(7)]
cont_feats=cont_feats + [f'f_{i}' for i in range(14,29)]

# Figure with subplots
fig=plt.figure(figsize=(15,14))

for i, f in enumerate(cont_feats):
    # New subplot
    plt.subplot(6,4,i+1)
    sns.histplot(x=data[f])

    # Aesthetics
    plt.title(f'Feature: {f}')
    plt.xlabel('')

# Overall aesthetics
fig.suptitle('Continuous feature distributions',  size=20)
fig.tight_layout()  # Improves appearance a bit
plt.show()

##with kde 
fig, axes = plt.subplots(nrows = 5, ncols = 6, figsize = (22,12))
for i, col in enumerate(aux[1:].columns):
    sns.kdeplot(data = aux, x = col, hue = 'clusters', palette = 'Spectral', ax = axes[(i // 6)][(i % 6)])

fig.tight_layout(h_pad=1.0, w_pad=0.5)

## Easier way: discrete features (but also work for discrete)
# Figure with subplots
fig=plt.figure(figsize=(15,14))

for i in range(7):
    # New subplot
    plt.subplot(4,2,i+1)
    feat_num=i+7
    sns.countplot(x=data.iloc[:,feat_num])

    # Aesthetics
    plt.title(f'Feature: 0{feat_num}')
    plt.xlim([-1,44])      # same scale for all plots
    plt.ylim([0,11000])   # same scale for all plots
    plt.xticks(np.arange(0,44,2))
    plt.xlabel('')
fig.suptitle('Discrete feature distributions',  size=20)
fig.tight_layout()  # Improves appearance a bit
plt.show()
