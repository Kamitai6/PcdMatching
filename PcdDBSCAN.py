import numpy as np
import numba as nb
import cv2
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
import plotly.graph_objects as go
import plotly.offline as offline
import matplotlib.pyplot as plt


# width  = 640
# height = 480
width  = 1344
height = 756
fx = 340
fy = 370

def padding_position(x, y, w, h, p):
    return x - p, y - p, w + p * 2, h + p * 2

def pcd2image(pcd):
    dst = np.zeros((height, width, 3), np.uint8)
    for i in range(pcd.shape[0]):
        z = pcd[i, 2] * 1.0
        x = int(-pcd[i, 0] * fx / z + (width  / 2))
        y = int( pcd[i, 1] * fy / z + (height / 2))
        if x > width or x < 0:
            print('error x: {}'.format(x))
        if y > height or y < 0:
            print('error y: {}'.format(y))
        for j in range(3):
            dst[y, x, j] = 255
    return dst

# path = './rs_color_b.pcd'
path = './coldep.pcd'
source = o3d.io.read_point_cloud(path, remove_nan_points = True, remove_infinite_points = True, print_progress = True)

point_array = np.asarray(source.points)
vectorized = point_array.reshape((-1,3))
vectorized = np.float32(vectorized)

# nearest_neighbors = NearestNeighbors(n_neighbors=5)
# nearest_neighbors.fit(vectorized)
# distances, indices = nearest_neighbors.kneighbors(vectorized)
# distances = np.sort(distances, axis=0)[:, 1]
# plt.plot(distances)
# plt.show()

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
    print(np.sqrt(x[:, 0]**2 + x[:, 1]**2))
    zobject = {str(label) : np.mean(np.sqrt(x[:, 0]**2 + x[:, 1]**2))}
    print('z-mean: {}'.format(zobject))
    zobjects.update(zobject)
    data.append(trace)

godlabel = min(zobjects, key=zobjects.get)
print('z-max: {}'.format(godlabel))

x = vectorized[np.where(dbscan.labels_==int(godlabel))]
x_np = np.array(x)
img = pcd2image(x_np)
blur_img = cv2.blur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),(9,9))
ret, otsu_img = cv2.threshold(blur_img, 0, 255, cv2.THRESH_OTSU)
contours, hierarchy = cv2.findContours(otsu_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
area = np.array(list(map(cv2.contourArea, contours)))
if (len(area) != 0 and np.nanmax(area) / (width*height) > 0.05):
    max_idx = np.argmax(area)
    x, y, w, h = cv2.boundingRect(contours[max_idx])
    x, y, w, h = padding_position(x, y, w, h, 5)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)  #rectangle
plt.imshow(img)
plt.show()

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(x_np)
colors = np.zeros(np.asarray(pcd.points).shape)
colors[:] = np.array([1.0, 0, 0])
pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.io.write_point_cloud('./RS_DBSCAN.pcd', pcd, True)
o3d.visualization.draw_geometries([pcd])

layout = go.Layout(yaxis=dict(scaleanchor='x'))
fig = dict(data=data, layout=layout)
offline.plot(fig, include_plotlyjs="cdn", auto_open=True, filename='sample plotly.html', config={
    'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape']}, )