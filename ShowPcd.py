import numpy as np
import open3d as o3d


path = '../Downloads/rs_color2.pcd'
# path = './g_code.pcd'
source = o3d.io.read_point_cloud(path, remove_nan_points = True, remove_infinite_points = True, print_progress = True)
o3d.visualization.draw_geometries([source])