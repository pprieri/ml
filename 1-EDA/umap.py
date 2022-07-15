

##UMAP 3D
import plotly.express as px
# UMAP
um = umap.UMAP(n_components=3)
components_umap = um.fit_transform(scaled_data_crop)

# 3D scatterplot
fig = px.scatter_3d(
    components_umap, x=0, y=1, z=2, color=preds_bgmm, size=0.1*np.ones(len(scaled_data_crop)), opacity = 1,
    title='UMAP plot in 3D',
    labels={'0': 'comp. 1', '1': 'comp. 2', '2': 'comp. 3'},
    width=650, height=500
)
fig.show()
