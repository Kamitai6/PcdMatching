import copy
import numpy as np
import open3d as o3d


def show(model, scene, model_to_scene_trans=np.identity(4)):
    model_t = copy.deepcopy(model)
    scene_t = copy.deepcopy(scene)

    model_t.paint_uniform_color([1, 0, 0])
    scene_t.paint_uniform_color([0, 0, 1])

    model_t.transform(model_to_scene_trans)

    o3d.visualization.draw_geometries([model_t, scene_t])

scene = o3d.io.read_point_cloud('./RS_DBSCAN.pcd', remove_nan_points = True, remove_infinite_points = True, print_progress = True)
model = o3d.io.read_point_cloud('./g_code.pcd', remove_nan_points = True, remove_infinite_points = True, print_progress = True)

size = np.abs((model.get_max_bound() - model.get_min_bound())).max() / 10
kdt_n = o3d.geometry.KDTreeSearchParamHybrid(radius=size, max_nn=50)
kdt_f = o3d.geometry.KDTreeSearchParamHybrid(radius=size * 50, max_nn=50)

model.estimate_normals(kdt_n)
scene.estimate_normals(kdt_n)
show(model, scene)

model_d = model.voxel_down_sample(voxel_size=size)
scene_d = scene.voxel_down_sample(voxel_size=size)
model_d.estimate_normals(kdt_n)
scene_d.estimate_normals(kdt_n)
show(model_d, scene_d)

model_f = o3d.pipelines.registration.compute_fpfh_feature(model_d, kdt_f)
scene_f = o3d.pipelines.registration.compute_fpfh_feature(scene_d, kdt_f)

checker = [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
           o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(size * 2)]

est_ptp = o3d.pipelines.registration.TransformationEstimationPointToPoint()
est_ptpln = o3d.pipelines.registration.TransformationEstimationPointToPlane()

criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration=40000)

# RANSAC
result1 = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                model_d, scene_d,
                model_f, scene_f,
                True, size * 2.0,
                estimation_method=est_ptp,
                ransac_n=4,
                checkers=checker,
                criteria=criteria
            )
show(model_d, scene_d, result1.transformation)

# ICP
result2 = o3d.pipelines.registration.registration_icp(model, scene, size, result1.transformation, est_ptpln)
show(model, scene, result2.transformation)