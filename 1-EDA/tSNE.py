

#3d:
import plotly.express as px
from sklearn.manifold import TSNE
tsne = TSNE(n_components=3)
components = tsne.fit_transform(scaled_data_crop.iloc[:5000,:])

# 3D scatterplot
fig = px.scatter_3d(
    components, x=0, y=1, z=2, color=preds_bgmm[:5000], size=0.1*np.ones(len(scaled_data_crop.iloc[:5000,:])), opacity = 1,
    title='t-SNE plot in 3D',
    labels={'0': 'comp. 1', '1': 'comp. 2', '2': 'comp. 3'},
    width=650, height=500
)
fig.show()
