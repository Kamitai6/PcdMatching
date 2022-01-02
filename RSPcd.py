import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)
    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    # inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
    return inlier_cloud

align = rs.align(rs.stream.color)
config = rs.config()
pipeline = rs.pipeline()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
profile = pipeline.start(config)

depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)
print(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)
print(depth_scale)

while True:
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    color_frame = aligned_frames.get_color_frame()
    depth_frame = aligned_frames.get_depth_frame()
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.04), cv2.COLORMAP_JET)
    cv2.imshow("RealSense  Color Image", color_image)
    cv2.imshow("RealSense  Depth Image", depth_colormap)

    if cv2.waitKey(1) != -1:
        print('finish')
        break

color_frame = aligned_frames.get_color_frame()
depth_frame = aligned_frames.get_depth_frame()
color_image = np.asanyarray(color_frame.get_data())
depth_image = np.asanyarray(depth_frame.get_data())
depth = o3d.geometry.Image(np.asanyarray(depth_frame.get_data()))
color = o3d.geometry.Image(color_image)

rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth)
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole_camera_intrinsic, depth_scale=depth_scale)
pcd = pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

pipeline.stop()
pcd = pcd.voxel_down_sample(voxel_size=0.001)

cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
pcd = display_inlier_outlier(pcd, ind)

cl, ind = pcd.remove_radius_outlier(nb_points=16, radius=0.05)
pcd = display_inlier_outlier(pcd, ind)

o3d.io.write_point_cloud('./rs_color.pcd', pcd, True)
o3d.visualization.draw_geometries([pcd])