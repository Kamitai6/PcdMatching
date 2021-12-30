import numpy as np
import cv2
import open3d as o3d

from pygcode import *
from pygcode import Line


word_bank  = []
layer_bank = []
type_bank = []
line_bank = []
parsed_Num_of_layers = 0
gcode_type = 0

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
    return inlier_cloud

with open('./CE3_Stanford_Bunny.gcode', 'r') as fh:
    layer_start = False
    for line_text in fh.readlines():
        line = Line(line_text)
        w = line.block.words # splits blocks into XYZEF, omits comments
        if line.comment:
            if (line.comment.text[0:7] == "LAYER:0"):
                layer_start = True
                parsed_Num_of_layers += 1
                gcode_type = 0
            if (line.comment.text[0:6] == "LAYER:" and layer_start):
                parsed_Num_of_layers += 1
                gcode_type = 0
            if (line.comment.text[0:15] == "TYPE:WALL-OUTER"):
                gcode_type = 1
            if (line.comment.text[0:15] == "TYPE:WALL-INNER"):
                gcode_type = 2
            if (line.comment.text[0:9] == "TYPE:FILL"):
                gcode_type = 3
            if (line.comment.text[0:12] == "TYPE:SUPPORT"):
                gcode_type = 4
            if (line.comment.text[0:22] == "TYPE:SUPPORT-INTERFACE"):
                gcode_type = 5
        if(np.shape(w)[0]):
            word_bank.append(w) # <Word: G01>, <Word: X15.03>, <Word: Y9.56>, <Word: Z0.269>, ...
            layer_bank.append(parsed_Num_of_layers)
            type_bank.append(gcode_type)
            line_bank.append(line_text)

index = {'X':0.0, 'Y':0.0, 'Z':0.0}
pre_index = {'X':0.0, 'Y':0.0, 'Z':0.0}
gcode_points = []

for i in range(len(layer_bank)): # for each line in file
    if type_bank[i] in [1, 2, 3]:
        if (str(word_bank[i][0])[:1] == 'G'):
            G_num = int(str(word_bank[i][0])[1:])
            if (G_num == 0 or G_num == 1):
                for j in range(1, len(word_bank[i])):
                    key = str(word_bank[i][j])[:1]
                    if key in index:
                        index[key] = float(str(word_bank[i][j])[1:]) / 1000.0
                if (index['X'] * index['Y'] * index['Z'] != 0.0 and (pre_index['X'] != index['X'] or pre_index['Y'] != index['Y'] or pre_index['Z'] != index['Z'])):
                    # print('index:{}'.format(index))
                    gcode_points.append(index.copy())
                    pre_index = index.copy()

sphere = []
for p in gcode_points:
    sphere.append([p['X'], p['Y'], p['Z']])
sphere_np = np.array(sphere)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(sphere_np)

pcd = pcd.voxel_down_sample(voxel_size=0.001)

cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
pcd = display_inlier_outlier(pcd, ind)

cl, ind = pcd.remove_radius_outlier(nb_points=16, radius=0.05)
pcd = display_inlier_outlier(pcd, ind)

o3d.io.write_point_cloud('./g_code.pcd', pcd, True)
o3d.visualization.draw_geometries([pcd])