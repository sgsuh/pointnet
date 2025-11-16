"""
Create: 2022.06.15
Author: SG.SUH
Python: 3.7
"""

import numpy as np

import os
import glob
import open3d as o3d # Import open3d as o3d, replacing pcl
import math
import argparse

from calibration_kitti import Calibration

def draw_box(pyplot_axis, vertices, axes = [0, 1, 2], color = 'red'):
    vertices = vertices[axes, :]
    connections = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]

    for connection in connections:
        pyplot_axis.plot(*vertices[:, connection], c = color, lw = 0.5)

def draw_point_cloud(cloud, ax, title, axes = [0, 1, 2], xlim3d = None, ylim3d = None, zlim3d = None):
    cloud = np.array(cloud)
    no_points = np.shape(cloud)[0]
    point_size = 10 ** (3 - int(np.log10(no_points)))

    if np.shape(cloud)[1] == 4:
        ax.scatter(*np.transpose(cloud[:, axes]), s = point_size, c = cloud[:, 3], cmap = 'gray')
    elif np.shape(cloud)[1] == 3:
        ax.scatter(*np.transpose(cloud[:, axes]), s = point_size, c = 'b', alpha = 0.7)

    ax.set_xlabel('{} axis'.format(axes_str[axes[0]]))
    ax.set_ylabel('{} axis'.format(axes_str[axes[1]]))

    if len(axes) > 2:
        ax.set_xlim3d(axes_limits[axes[0]])
        ax.set_ylim3d(axes_limits[axes[1]])
        ax.set_zlim3d(axes_limits[axes[2]])
        ax.set_zlabel('{} axis'.format(axes_str[axes[2]]))
    else:
        ax.set_xlim(*axes_limits[axes[0]])
        ax.set_ylim(*axes_limits[axes[1]])

    if xlim3d != None:
        ax.set_xlim3d(xlim3d)
    
    if ylim3d != None:
        ax.set_ylim3d(ylim3d)

    if zlim3d != None:
        ax.set_zlim3d(zlim3d)

    ax.set_title(title)

def voxel_filter(cloud, leaf_sizes):
    """
    Converts PCL's VoxelGrid filtering to Open3D's voxel_down_sample.
    Args:
        cloud (o3d.geometry.PointCloud): Input Open3D Point Cloud.
        leaf_sizes (list): [size_x, size_y, size_z] (PCL) -> leaf_size (Open3D)
    Returns:
        o3d.geometry.PointCloud: Voxel-filtered point cloud.
    """
    # Open3D uses a single cubic size (leaf_size).
    # Since the original PCL code used [0.16, 0.16, 4], we approximate with the smallest relevant size (0.16).
    leaf_size = leaf_sizes[0]
    cloud_voxel_filtered = cloud.voxel_down_sample(voxel_size=leaf_size)

    return cloud_voxel_filtered

def roi_filter(cloud, x_roi, y_roi, z_roi):
    """
    Converts PCL's CropBox filtering to Open3D's crop function.
    Args:
        cloud (o3d.geometry.PointCloud): Input Open3D Point Cloud.
        x_roi, y_roi, z_roi (list): [min, max] range for each axis.
    Returns:
        o3d.geometry.PointCloud: ROI filtered point cloud.
    """
    xc_min, xc_max = x_roi
    yc_min, yc_max = y_roi
    zc_min, zc_max = z_roi

    min_bound = np.array([xc_min, yc_min, zc_min])
    max_bound = np.array([xc_max, yc_max, zc_max])

    # Create AxisAlignedBoundingBox (AABB)
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    
    # Crop the point cloud
    cloud_roi_filtered = cloud.crop(bbox)

    return cloud_roi_filtered

def plane_segmentation(cloud, dist_thold, max_iter):
    """
    Converts PCL's SACMODEL_NORMAL_PLANE segmentation to Open3D's segment_plane.
    Note: Open3D's segment_plane implements RANSAC for a general plane model (SACMODEL_PLANE).
    Args:
        cloud (o3d.geometry.PointCloud): Input Open3D Point Cloud.
        dist_thold (float): Distance threshold for inliers.
        max_iter (int): Maximum RANSAC iterations.
    Returns:
        tuple: (inlier_indices, coefficients)
    """
    # Use Open3D's RANSAC plane segmentation
    inlier_indices, coefficients = cloud.segment_plane(
        distance_threshold=dist_thold,
        ransac_n=3,
        num_iterations=max_iter
    )

    return indices, coefficients

def clustering(cloud, tol, min_size, max_size):
    """
    Converts PCL's Euclidean Cluster Extraction to Open3D's cluster_dbscan.
    Args:
        cloud (o3d.geometry.PointCloud): Input Open3D Point Cloud (obstacles).
        tol (float): Clustering tolerance (DBSCAN's eps).
        min_size (int): Minimum cluster size (DBSCAN's min_points).
        max_size (int): Maximum cluster size (Post-filtering applied).
    Returns:
        list: List of cluster indices (list of lists, where each inner list contains point indices).
    """
    # Use Open3D's DBSCAN clustering:
    # eps: ClusterTolerance (Max distance between points)
    # min_points: MinClusterSize (Minimum number of points to form a cluster)
    
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error) as vcm:
        labels = np.array(cloud.cluster_dbscan(eps=tol, min_points=min_size, print_progress=False))

    cluster_indices = []
    
    # Extract cluster indices and manually apply MaxClusterSize filtering
    max_label = labels.max()
    for i in range(max_label + 1):
        indices = np.where(labels == i)[0]
        if len(indices) <= max_size:
            cluster_indices.append(indices.tolist())

    return cluster_indices

def get_cluster_box_list(cluster_indices, cloud_obsts):
    """
    Creates cluster point clouds and AABB center/size lists from cluster indices.
    Args:
        cluster_indices (list): List of cluster indices.
        cloud_obsts (o3d.geometry.PointCloud): Obstacle Open3D point cloud.
    Returns:
        tuple: (cloud_cluster_list, box_coord_list)
    """
    cloud_cluster_list = []
    box_coord_list = []

    points_obsts_np = np.asarray(cloud_obsts.points)

    for indices in cluster_indices:
        # Extract points from obstacle cloud using indices
        points = points_obsts_np[indices]

        # Create Open3D PointCloud object
        cloud_cluster = o3d.geometry.PointCloud()
        cloud_cluster.points = o3d.utility.Vector3dVector(points)
        cloud_cluster_list.append(cloud_cluster)

        # Calculate Axis-Aligned Bounding Box (AABB) dimensions
        x_max, x_min = np.max(points[:, 0]), np.min(points[:, 0])
        y_max, y_min = np.max(points[:, 1]), np.min(points[:, 1])
        z_max, z_min = np.max(points[:, 2]), np.min(points[:, 2])

        # PCL box format: [center_x, center_y, center_z, length, width, height, rotation_z]
        center_x = (x_max + x_min) / 2
        center_y = (y_max + y_min) / 2
        center_z = (z_max + z_min) / 2
        
        length = (x_max - x_min)
        width = (y_max - y_min)
        height = (z_max - z_min)

        # AABB has zero rotation
        rotation_z = 0

        box = [center_x, center_y, center_z, length, width, height, rotation_z]
        box_coord_list.append(box)

    return cloud_cluster_list, box_coord_list

def draw_box(pyplot_axis, vertices, axes = [0, 1, 2], color = 'red'):
    """Draws a 3D bounding box on a matplotlib pyplot axis."""
    vertices = vertices[axes, :]
    connections = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]

    for connection in connections:
        pyplot_axis.plot(*vertices[:, connection], c = color, lw = 0.5)

def draw_point_cloud(cloud, ax, title, axes = [0, 1, 2], xlim3d = None, ylim3d = None, zlim3d = None):
    """Draws the point cloud using matplotlib scatter."""
    cloud = np.array(cloud)
    no_points = np.shape(cloud)[0]
    point_size = 10 ** (3 - int(np.log10(no_points)))

    # Dummy definitions for axes_str and axes_limits (assuming they should be defined globally or passed)
    axes_str = ['X', 'Y', 'Z'] 
    axes_limits = [[-80, 80], [-80, 80], [-5, 5]]
    
    if np.shape(cloud)[1] == 4:
        ax.scatter(*np.transpose(cloud[:, axes]), s = point_size, c = cloud[:, 3], cmap = 'gray')
    elif np.shape(cloud)[1] == 3:
        ax.scatter(*np.transpose(cloud[:, axes]), s = point_size, c = 'b', alpha = 0.7)

    ax.set_xlabel('{} axis'.format(axes_str[axes[0]]))
    ax.set_ylabel('{} axis'.format(axes_str[axes[1]]))

    if len(axes) > 2:
        ax.set_xlim3d(axes_limits[axes[0]])
        ax.set_ylim3d(axes_limits[axes[1]])
        ax.set_zlim3d(axes_limits[axes[2]])
        ax.set_zlabel('{} axis'.format(axes_str[axes[2]]))
    else:
        ax.set_xlim(*axes_limits[axes[0]])
        ax.set_ylim(*axes_limits[axes[1]])

    if xlim3d != None:
        ax.set_xlim3d(xlim3d)
    
    if ylim3d != None:
        ax.set_ylim3d(ylim3d)

    if zlim3d != None:
        ax.set_zlim3d(zlim3d)

    ax.set_title(title)

def get_gt_box(label_path, calib_path):
    """Reads ground truth boxes from KITTI label format and converts them to Lidar coordinates."""
    calib = Calibration(calib_path)
    class_names = {'Car': 1, 'Pedestrian': 2, 'Cyclist': 3, 'Van': 1, 'Person_sitting': 2, 'Truck': 1, 'DontCare': -1, 'Misc': -1, 'Tram': -1}
    label_file_path = label_path

    P2 = np.concatenate([calib.P2, np.array([[0., 0., 0., 1.]])], axis = 0)
    R0_4x4 = np.zeros([4, 4], dtype = calib.R0.dtype)
    R0_4x4[3, 3] = 1.
    R0_4x4[:3, :3] = calib.R0
    V2C_4x4 = np.concatenate([calib.V2C, np.array([[0., 0., 0., 1.]])], axis = 0)
    calib_info = {'P2': P2, 'R0_rect': R0_4x4, 'Tr_velo_to_cam': V2C_4x4}

    box_list = []

    with open(label_file_path, 'r') as f:
        for line in f:
            content = line.split('\n')[0].split(' ')
            
            name = content[0]
            class_id = class_names[name]
            truncated = float(content[1])
            occluded = int(content[2])
            alpha = float(content[3])
            bbox = []
            bbox.append(float(content[4]))
            bbox.append(float(content[5]))
            bbox.append(float(content[6]))
            bbox.append(float(content[7]))
            
            h = float(content[8])
            w = float(content[9])
            l = float(content[10])

            location = []
            location.append(float(content[11]))
            location.append(float(content[12]))
            location.append(float(content[13]))
            rotation_y = float(content[14])

            loc_lidar = calib.rect_to_lidar(np.array(location).reshape(1, 3))[0]
            loc_lidar[2] += h / 2

            gt_box = []

            gt_box.append(loc_lidar[0])
            gt_box.append(loc_lidar[1])
            gt_box.append(loc_lidar[2])

            gt_box.append(l)
            gt_box.append(w)
            gt_box.append(h)

            gt_box.append(-(np.pi / 2 + rotation_y))

            if class_id == -1:
                continue

            box_list.append(gt_box)

    return box_list

def draw_points(points, cluster_box_list, gt_box_list):
    """Visualizes the point cloud and bounding boxes (clusters in red, GT in blue) using Open3D."""
    vis = o3d.visualization.Visualizer()

    vis.create_window()

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 1.0, origin = [0, 0, 0])

    vis.add_geometry(axis_pcd)

    pts = o3d.geometry.PointCloud()
    pts.points = o3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)

    pts.colors = o3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))

    for cluster_box in cluster_box_list:
        center = cluster_box[0:3]
        lwh = cluster_box[3:6]
        # Convert Z-rotation angle to rotation matrix via axis-angle representation
        axis_angle = np.array([0, 0, cluster_box[6] + 1e-10])

        rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angle)
        box3d = o3d.geometry.OrientedBoundingBox(center, rot, lwh)

        line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

        lines = np.asarray(line_set.lines)
        # Concatenate extra lines as defined in the original code
        lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis = 0) 

        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.paint_uniform_color((1, 0, 0)) # Red for detected clusters/FP boxes

        vis.add_geometry(line_set)

    for gt_box in gt_box_list:
        center = gt_box[0:3]
        lwh = gt_box[3:6]
        axis_angle = np.array([0, 0, gt_box[6] + 1e-10])

        rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angle)
        box3d = o3d.geometry.OrientedBoundingBox(center, rot, lwh)

        line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

        lines = np.asarray(line_set.lines)
        # Concatenate extra lines as defined in the original code
        lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis = 0)

        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.paint_uniform_color((0, 0, 1)) # Blue for Ground Truth boxes

        vis.add_geometry(line_set)

    vis.run()
    vis.destroy_window()

def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--fold_path",
                        type=str,
                        default="training")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = make_parser()

    fold_path = args.fold_path

    lidar_fold = os.path.join(fold_path, 'velodyne')
    label_fold = os.path.join(fold_path, 'label_2')
    calib_fold = os.path.join(fold_path, 'calib')
    save_fold = os.path.join(fold_path, 'save')

    file_list = glob.glob(lidar_fold + '/*.bin')
    file_list.sort()

    for file_path in file_list:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        label_file_path = os.path.join(label_fold, file_name + '.txt')
        calib_file_path = os.path.join(calib_fold, file_name + '.txt')

        cloud_np = np.fromfile(file_path, dtype = np.float32).reshape(-1, 4)

        cloud_x = cloud_np[:, 0]
        cloud_y = cloud_np[:, 1]
        cloud_z = cloud_np[:, 2]

        x_max, x_min = np.max(cloud_x), np.min(cloud_x)
        y_max, y_min = np.max(cloud_y), np.min(cloud_y)
        z_max, z_min = np.max(cloud_z), np.min(cloud_z)

        # Create Open3D PointCloud object from (x, y, z)
        cloud_xyz = o3d.geometry.PointCloud()
        cloud_xyz.points = o3d.utility.Vector3dVector(cloud_np[:, 0:3])

        colors = {'Car': 'b', 'Tram': 'r', 'Cyclist': 'g', 'Van': 'c', 'Truck': 'm', 'Pedestrian': 'y', 'Sitter': 'k'}
        axes_limits = [[int(x_min * 1.2), int(x_max * 1.2)], [int(y_min * 1.2), int(y_max * 1.2)], [-5, 5]]
        axes_str = ['X', 'Y', 'Z']

        # 1. Voxel Filtering
        cloud_voxel_filtered = voxel_filter(cloud_xyz, [0.16, 0.16, 4]) 

        # 2. Region of Interest (ROI) Filtering
        cloud_roi_filtered = roi_filter(cloud_voxel_filtered, [15, 70], [-15, 15], [-3, 1])

        # 3. Ground Plane Segmentation
        indices, coefficients = plane_segmentation(cloud_roi_filtered, 0.3, 100)

        if len(indices) == 0:
            print('Could not estimate a planar model for the given dataset.')
            continue

        # Extract plane and obstacles
        cloud_plane = cloud_roi_filtered.select_by_index(indices, invert=False)
        cloud_obsts = cloud_roi_filtered.select_by_index(indices, invert=True)

        # 4. Clustering (Obstacles)
        cluster_indices = clustering(cloud_obsts, 0.7, 10, 200)
        cloud_cluster_list, box_coord_list = get_cluster_box_list(cluster_indices, cloud_obsts)

        # Load Ground Truth Boxes
        gt_box_coord_list = get_gt_box(label_file_path, calib_file_path)

        detect_box = np.array(box_coord_list)
        gt_box = np.array(gt_box_coord_list)

        fp_box = []

        for k in range(detect_box.shape[0]):
            detect_x1 = detect_box[k][0] - detect_box[k][3] / 2
            detect_y1 = detect_box[k][1] - detect_box[k][4] / 2
            detect_x2 = detect_box[k][0] + detect_box[k][3] / 2
            detect_y2 = detect_box[k][1] + detect_box[k][4] / 2

            overlap = 0

            for n in range(gt_box.shape[0]):
                gt_x1 = gt_box[n][0] - gt_box[n][3] / 2
                gt_y1 = gt_box[n][1] - gt_box[n][4] / 2
                gt_x2 = gt_box[n][0] + gt_box[n][3] / 2
                gt_y2 = gt_box[n][1] + gt_box[n][4] / 2

                inter_w = min(detect_x2, gt_x2) - max(detect_x1, gt_x1)

                if inter_w > 0:
                    inter_h = min(detect_y2, gt_y2) - max(detect_y1, gt_y1)

                    if inter_h > 0:
                        overlap = 1                   

                        break

            if overlap > 0:
                continue

            fp_box.append(detect_box[k])

        for box in fp_box:
            cx = box[0]
            cy = box[1]
            cz = box[2]

            dx = box[3]
            dy = box[4]
            dz = box[5]

            rz = box[6]

            cosa = math.cos(-rz)
            sina = math.sin(-rz)

            crop_point = []

            for point in cloud_np:
                # print(point)        

                x = point[0]
                y = point[1]
                z = point[2]

                if abs(z - cz) > dz / 2.:
                    continue

                local_x = (x - cx) * cosa + (y - cy) * (-sina)
                local_y = (x - cx) * sina + (y - cy) * cosa

                if abs(local_x) < dx / 2.0 + 1e-2 and abs(local_y) < dy / 2.0 + 1e-2:
                    new_point = np.array([x - cx, y - cy, z - cz, point[3]], dtype = np.float32)

                    crop_point.append(new_point)

            if len(crop_point) < 128:
                continue

            crop_point = np.array(crop_point)

            class_fold = os.path.join(save_fold, 'Neg')

            if not os.path.isdir(class_fold):
                os.makedirs(class_fold)

            save_path = os.path.join(class_fold, 'Kitti_{}_{}_{}_{}_{}.npy'.format(file_name, int(cx), int(cy), int(cz), crop_point.shape[0]))

            print(save_path)

        draw_points(cloud_np, fp_box, gt_box_coord_list)