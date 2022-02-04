import time
import math
import random
import cv2
import numpy as np
import pandas as pd
import open3d as o3d
import pyrealsense2 as rs
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import plotly.graph_objects as go
import plotly.offline as offline


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)
    outlier_cloud.paint_uniform_color([1, 0, 0])
    # inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
    return inlier_cloud

def padding_position(x, y, w, h, width, height):
    x_, y_, w_, h_ = x, y, w, h
    rate = 1.2

    if x - w*(rate-1)/2 >= 1:
        x_ = x - w*(rate-1)/2
        if x_ + w*rate < width:
            w_ = w*rate
        else:
            w_ = width - 2
    else:
        x_ = 1
        if w*rate - (w*(rate-1)/2-x) < width:
            w_ = w*rate - (w*(rate-1)/2-x+x_)
        else:
            w_ = width - 2

    if y - h*(rate-1)/2 >= 1:
        y_ = y - h*(rate-1)/2
        if y_ + h*rate < height:
            h_ = h*rate
        else:
            h_ = height - 2
    else:
        y_ = 1
        if h*rate - (h*(rate-1)/2-y) < height:
            h_ = h*rate - (h*(rate-1)/2-y+y_)
        else:
            h_ = height - 2

    return int(x_), int(y_), int(w_), int(h_)

def pcd2image(pcd, height, width, fx, fy):
    dst = np.zeros((height, width, 3), np.uint8)
    for i in range(pcd.shape[0]):
        z = pcd[i, 2] * 1.0
        x = int(-pcd[i, 0] * fx / z + (width  / 2))
        y = int( pcd[i, 1] * fy / z + (height / 2))
        if x >= width:
            x = width - 1
        if x < 0:
            x = 0
        if y >= height:
            y = height - 1
        if y < 0:
            y = 0
        for j in range(3):
            dst[y, x, j] = 255
    return dst

def detect_cut(img, rect_param, num):
    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (rect_param[0],rect_param[1],rect_param[2],rect_param[3])
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,2,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img_ = img*mask2[:,:,np.newaxis]
    if img_.any():
        img = img_
    img2 = img.copy()
    print(rect)
    cv2.rectangle(img2, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)  #rectangle
    cv2.imwrite('cut'+str(num)+'.png', img2)
    return img

def detect_canny(img, num):
    def edges2polylines(edges, th_n=6, th_c=None):
        def distance(a, b):
            return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

        def convert_imshow2plot(p):
            tmp = p.copy()
            tmp[0] = p[1]
            tmp[1] = - p[0]
            return tmp
        
        def convert_plot2imshow(p):
            tmp = p.copy()
            tmp[0] = - p[1]
            tmp[1] = p[0]
            tmp = tmp.astype(np.int64)
            return tmp

        def cost_function(p, line_end, d):
            r = p - line_end
            r_norm = math.sqrt(np.dot(r, r))
            res = r_norm ** 2 - 0.99 * np.dot(r, d)
            return res
    
        edges_tmp = edges.copy()
        th_c = th_c or th_n ** 2
        il, jl = np.where(edges_tmp == 255)
        edgepoints = [np.array([i, j]) for i, j in zip(il, jl)]
        init_n_points = len(edgepoints)
        polylines = []
        count = 0
        while count < init_n_points:
            il, jl = np.where(edges_tmp == 255)
            edgepoints = [np.array([i, j]) for i, j in zip(il, jl)]
            n_points = len(edgepoints)
            init_idx = random.randrange(n_points)
            polyline = []
            polyline.append(convert_imshow2plot(edgepoints[init_idx]))
            edges_tmp[edgepoints[init_idx][0], edgepoints[init_idx][1]] = 0
            count += 1
            could_reverse = True
            while True:
                nearest = []
                pl = []
                ri = math.floor(th_n)
                xl = np.arange(-ri, ri + 1)
                for x in xl:
                    y = math.sqrt(th_n ** 2 - x ** 2)
                    yi = math.floor(y)
                    yl = np.arange(-yi, yi + 1)
                    pl.extend([np.array([x, y]) + polyline[-1] for y in yl])
                for p in pl:
                    p_img = convert_plot2imshow(p)
                    if p_img[0] < 0 or p_img[1] < 0:
                        continue
                    try:
                        if edges_tmp[p_img[0], p_img[1]] == 255:
                            nearest.append(p)
                    except IndexError:
                        continue
                nearest = sorted(nearest, key=lambda p: distance(p, polyline[-1]))
                if not nearest:
                    if could_reverse:
                        polyline.reverse()
                        could_reverse = False
                        continue
                    else:
                        break
                if len(polyline) <= 2:
                    next_point = nearest[0]
                else:
                    d = (polyline[-1] - polyline[-2]) + 0.47 * (polyline[-2] - polyline[-3])
                    d = d / np.linalg.norm(d)
                    nearest = sorted(nearest, key=lambda p: cost_function(p, polyline[-1], d))
                    if cost_function(nearest[0], polyline[-1], d) > th_c:
                        if could_reverse:
                            polyline.reverse()
                            could_reverse = False
                            continue
                        else:
                            break
                    next_point = nearest[0]
                polyline.append(next_point)
                edges_tmp[convert_plot2imshow(next_point)[0], convert_plot2imshow(next_point)[1]] = 0
                count += 1
            polylines.append(polyline)
        return polylines
    
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    med_val = np.median(img2)
    sigma = 0.33  # 0.33
    min_val = int(max(0, (1.0 - sigma) * med_val))
    max_val = int(max(255, (1.0 + sigma) * med_val))
    edges = cv2.Canny(img2, threshold1 = min_val, threshold2 = max_val)
    lines = edges2polylines(edges)

    figure_size = 15
    plt.figure(figsize=(figure_size,figure_size))
    plt.subplot(1,2,1),plt.imshow(edges,cmap = 'gray')
    plt.title('Canny Edge')
    ax = plt.subplot(1,2,2)
    dst = np.zeros_like(img2)
    for line in lines:
        plus_x, minus_y = zip(*line)
        x = np.array(plus_x)
        y = -np.array(minus_y)
        if len(x) > 50:
            cv2.polylines(dst, [np.hstack([x.reshape(-1, 1), y.reshape(-1, 1)])], False, (255, 255, 255))
            ax.plot(x, y)
    ax.imshow(dst)
    plt.title('Edge Polyline')
    plt.savefig('edge'+str(num)+'.jpg', dpi=300)
    # plt.show()
    return dst

def detect_feature(img, edges, num):
    detector = cv2.AKAZE_create()
    kp = detector.detect(edges)
    dst = cv2.drawKeypoints(img, kp, None)
    data = []
    for i in range(len(kp)):
        x = kp[i].pt[0]
        y = kp[i].pt[1]
        data.append([x, y])
    data_norm = np.array(data)
    # print(data_norm)

    dbscan = DBSCAN(eps=20, min_samples=20, metric='euclidean').fit(data_norm)
    
    rate = 0.1
    spag = False
    figure_size = 15
    plt.figure(figsize=(figure_size,figure_size))
    plt.subplot(1,2,1),plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    plt.title('AKAZE Detect')
    ax = plt.subplot(1,2,2)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    unique_labels = np.unique(dbscan.labels_)
    for label in unique_labels:
        if label == -1: continue
        x = data_norm[np.where(dbscan.labels_==label)]
        if np.abs((max(x[:, 0]) - min(x[:, 0])) * (max(x[:, 1]) - min(x[:, 1]))) > img.shape[0]*img.shape[1]*rate:
            spag = True
        plt.scatter(x[:, 0], x[:, 1], label=label)
    plt.legend()
    plt.title('DBSCAN Clustering')
    plt.savefig('akaze'+str(num)+'.jpg', dpi=300)
    # plt.show()
    return spag

def spaghetti(depth_image, color_image, intr, pinhole, scale, num):
    depth = o3d.geometry.Image(depth_image)
    color = o3d.geometry.Image(color_image)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole)
    pcd = pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    pcd = pcd.voxel_down_sample(voxel_size=0.001)
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=300, std_ratio=0.1)
    pcd = display_inlier_outlier(pcd, ind)
    o3d.io.write_point_cloud('./rs_color'+str(num)+'.pcd', pcd, True)
    # o3d.visualization.draw_geometries([pcd])

    if pcd.has_points():
        point_array = np.asarray(pcd.points)
        vectorized = point_array.reshape((-1,3))
        vectorized = np.float32(vectorized)

        dbscan = DBSCAN(eps=0.03, min_samples=100, metric='euclidean').fit(vectorized)

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
            x_wid = np.abs((max(x[:, 0]) - min(x[:, 0])))
            y_wid = np.abs((max(x[:, 1]) - min(x[:, 1])))
            print(x_wid)
            print(y_wid)
            if x_wid < 0.05 and y_wid < 0.05:
                continue
            zobject = {str(label) : np.mean(np.sqrt(x[:, 0]**2 + x[:, 1]**2))}
            print('z-mean: {}'.format(zobject))
            zobjects.update(zobject)
            data.append(trace)

        godlabel = min(zobjects, key=zobjects.get)
        print('z-max: {}'.format(godlabel))

        x = vectorized[np.where(dbscan.labels_==int(godlabel))]
        x_np = np.array(x)
        img = pcd2image(x_np, intr.height, intr.width, intr.fx, intr.fy)
        blur_img = cv2.blur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),(9,9))
        ret, otsu_img = cv2.threshold(blur_img, 0, 255, cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(otsu_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area = np.array(list(map(cv2.contourArea, contours)))
        rect = []
        if (len(area) != 0 and np.nanmax(area) / (intr.width*intr.height) > 0.03):
            max_idx = np.argmax(area)
            rect = cv2.boundingRect(contours[max_idx])
            rect = padding_position(rect[0], rect[1], rect[2], rect[3], intr.width - 1, intr.height - 1)
            cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)  #rectangle
        else:
            rect = [int(intr.width / 4), int(intr.height / 4), int(intr.width / 2), int(intr.height / 2)]
        cv2.imwrite('rect'+str(num)+'.png', img)
        # plt.show()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(x_np)
        colors = np.zeros(np.asarray(pcd.points).shape)
        colors[:] = np.array([1.0, 0, 0])
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.io.write_point_cloud('./RS_DBSCAN'+str(num)+'.pcd', pcd, True)
        # o3d.visualization.draw_geometries([pcd])

        fig = dict(data=data)
        offline.plot(fig, include_plotlyjs="cdn", auto_open=True, filename='sample plotly.html', config={
            'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'drawclosedpath', 'drawcircle', 'drawrect', 'eraseshape']}, )
    else:
        rect = [int(intr.width / 4), int(intr.height / 4), int(intr.width / 2), int(intr.height / 2)]
    
    cut = detect_cut(color_image, rect, num)
    edges = detect_canny(cut, num)
    spag = detect_feature(cut, edges, num)

    return spag

def detection(depth_image, color_image, intr, pinhole, scale, num):
    spag = spaghetti(depth_image, color_image, intr, pinhole, scale, num)
    print(spag)

def pipe():
    num = 0
    align = rs.align(rs.stream.color)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config) # Start streaming
    sensor_dep = profile.get_device().first_depth_sensor()
    depth_scale = sensor_dep.get_depth_scale()

    sensor_dep.set_option(rs.option.visual_preset, 3) # 5 is short range, 3 is low ambient light
    sensor_dep.set_option(rs.option.confidence_threshold, 3)
    sensor_dep.set_option(rs.option.noise_filtering, 6)
    sensor_dep.set_option(rs.option.laser_power, 100) # 0 ~ 100
    sensor_dep.set_option(rs.option.receiver_gain, 18) # 8 ~ 18
    sensor_dep.set_option(rs.option.min_distance, 1) # mm

    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)
    
    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            detection(depth_image, color_image, intr, pinhole_camera_intrinsic, depth_scale, num)
            num+=1

    except Exception as e:
        print("error:", e)
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

pipe()