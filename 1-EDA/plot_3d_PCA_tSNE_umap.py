#3d:
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap #need to use umap-learn

df =
scaled_df =
sample_size = 
n_components = 
y = 

transformer = PCA(n_components=n_components)
# transformer = TSNE(n_components=n_components)
# transformer = umap.UMAP(n_components=3)
components = tsne.fit_transform(scaled_df.iloc[:sample_size,:])

# 3D scatterplot
fig = px.scatter_3d(
    components, x=0, y=1, z=2, color=y[:sample_size], size=0.1*np.ones(len(scaled_df.iloc[:5000,:])), opacity = 1,
    title='t-SNE plot in 3D',
    labels={'0': 'comp. 1', '1': 'comp. 2', '2': 'comp. 3'},
    width=650, height=500
)
fig.show()