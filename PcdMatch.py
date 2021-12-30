import copy
import numpy as np
import open3d as o3d
from probreg import cpd, filterreg, gmmtree


target = o3d.io.read_point_cloud('./RS_DBSCAN.pcd', remove_nan_points = True, remove_infinite_points = True, print_progress = True)
source = o3d.io.read_point_cloud('./g_code.pcd', remove_nan_points = True, remove_infinite_points = True, print_progress = True)

# target = copy.deepcopy(source)
# th = np.deg2rad(30.0)
# target.transform(np.array([[np.cos(th), -np.sin(th), 0.0, 0.0],
#                            [np.sin(th), np.cos(th), 0.0, 0.0],
#                            [0.0, 0.0, 1.0, 0.0],
#                            [0.0, 0.0, 0.0, 1.0]]))

source = source.voxel_down_sample(voxel_size=0.0025)
target = target.voxel_down_sample(voxel_size=0.01)

tf_param, _, _ = cpd.registration_cpd(source, target, tf_type_name='rigid', w=0.0, maxiter=100, tol=0.0001)
result = copy.deepcopy(source)
result.points = tf_param.transform(result.points)

source.paint_uniform_color([1, 0, 0])
target.paint_uniform_color([0, 1, 0])
result.paint_uniform_color([0, 0, 1])

o3d.visualization.draw_geometries([source, target, result])