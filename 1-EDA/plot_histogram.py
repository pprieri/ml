## Easier way: continuous features (but also work for discrete)
## a) with histo
data=
cols = ''

# Figure with subplots
fig=plt.figure(figsize=(15,14))

for i, f in enumerate(cols):
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
data = 
n_rows = 
n_cols

fig, axes = plt.subplots(nrows = n_rows, ncols = n_cols, figsize = (22,12))
for i, col in enumerate(data.columns):
    sns.kdeplot(data = data, x = col, hue = 'clusters', palette = 'Spectral', ax = axes[(i // n_cols)][(i % n_cols)])

fig.tight_layout(h_pad=1.0, w_pad=0.5)

## Easier way: discrete features (but also work for discrete)
# Figure with subplots

x_lim = [-1,44]
y_lim = [0,11000]
fig=plt.figure(figsize=(15,14))

for i in range(7):
    # New subplot
    plt.subplot(4,2,i+1)
    feat_num=i+7
    sns.countplot(x=data.iloc[:,feat_num])

    # Aesthetics
    plt.title(f'Feature: 0{feat_num}')
    plt.xlim(x_lim)      # same scale for all plots
    plt.ylim(y_lim)   # same scale for all plots
    plt.xticks(np.arange(0,x_lim[-1],2))
    plt.xlabel('')
fig.suptitle('Discrete feature distributions',  size=20)
fig.tight_layout()  # Improves appearance a bit
plt.show()


##More control on bin and edges

data = ''
y = #containing the target, or a group, or clusters
n_unique_y=  '' #can be groups or clusters in the data


n_rows= #6
n_cols= #4
bins_min= #min of cols
bins_max= #max of cols 5

fig, axs = plt.subplots(n_rows, n_cols, figsize=(16, 20))
axs = axs.ravel() ##transform from 2D (n rows, n cols) into nr rows * nr cols, which is easier to iterate
cols =  #[col for col in data.columns if data[col].dtype == 'float']

for ax, f in zip(axs, cols):
    for i in range(n_unique_y):
        h, edges = np.histogram(data[f][y == i], bins=np.linspace(bins_min, bins_max, 26))
        ax.plot((edges[:-1] + edges[1:]) / 2, h, label=f"Group {i}", lw=3) #to center the value in bins we do : (edges[:-1] + edges[1:]) / 2
    ax.set_title(f)
#axs[-2].axis('off') to control the axis depending on the nunmber of rows
#axs[-1].axis('off')
plt.suptitle('Histograms of the float features of the 7 Groups', y=0.95, fontsize=20)
plt.show()