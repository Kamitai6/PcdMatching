import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


path = './RS_DBSCAN.pcd'
# path = './rs_color.pcd'
# path = './g_code.pcd'
source = o3d.io.read_point_cloud(path, remove_nan_points = True, remove_infinite_points = True, print_progress = True)
o3d.visualization.draw_geometries([source])
