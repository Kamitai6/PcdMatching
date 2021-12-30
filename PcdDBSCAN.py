import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import plotly.graph_objects as go
import plotly.offline as offline
import matplotlib.pyplot as plt


path = '../Documents/rs_color.pcd'
# path = './g_code.pcd'
source = o3d.io.read_point_cloud(path, remove_nan_points = True, remove_infinite_points = True, print_progress = True)

point_array = np.asarray(source.points)
vectorized = point_array.reshape((-1,3))
vectorized = np.float32(vectorized)

nearest_neighbors = NearestNeighbors(n_neighbors=5)
nearest_neighbors.fit(vectorized)
distances, indices = nearest_neighbors.kneighbors(vectorized)
distances = np.sort(distances, axis=0)[:, 1]
print(distances)
plt.plot(distances)
plt.show()

dbscan = DBSCAN(eps=0.05, min_samples=100, metric='euclidean').fit(vectorized)

unique_labels = np.unique(dbscan.labels_)
data = []
zobjects = {}

for label in unique_labels:
    if label == -1: continue
    x = vectorized[np.where(dbscan.labels_==label)]
    trace=go.Scatter3d(
        x=x[:, 0],
        y=x[:, 1],
        z=x[:, 2],
        mode='markers',
        showlegend=True,
        name=str(label),
        marker=dict(
            size=3,
            color=label,
            colorscale='Viridis',
            opacity=0.8
        )
    )
    zobject = {str(label) : np.mean(x[:, 2])}
    print('z-mean: {}'.format(zobject))
    zobjects.update(zobject)
    data.append(trace)

godlabel = max(zobjects, key=zobjects.get)
print('z-max: {}'.format(godlabel))
x = vectorized[np.where(dbscan.labels_==int(godlabel))]
print(x)

x_np = np.array(x)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(x_np)

o3d.io.write_point_cloud('./RS_DBSCAN.pcd', pcd, True)

fig = dict(data=data)
offline.plot(fig, include_plotlyjs="cdn", auto_open=True, filename='sample plotly.html', config={
    'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape']}, )