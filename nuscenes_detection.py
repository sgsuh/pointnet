"""
Create: 2022.07.06
Author: SG.SUH
Python: 3.7
"""

import os
import pickle
import numpy as np
import open3d as o3d # Import Open3D as o3d for convenience
import math
import argparse

def draw_points(points, cluster_box_list, gt_box_list):
    """
    Visualizes the point cloud and bounding boxes using Open3D. (Original visualization logic maintained)
    """
    # Initialize Open3D visualizer
    vis = o3d.visualization.Visualizer()

    vis.create_window()

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    # Create coordinate frame
    axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 1.0, origin = [0, 0, 0])

    vis.add_geometry(axis_pcd)

    # Create PointCloud object and set points
    pts = o3d.geometry.PointCloud()
    pts.points = o3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)

    # Set point colors to white (default)
    pts.colors = o3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))

    # Draw clustered detection boxes (Red)
    for cluster_box in cluster_box_list:
        center = cluster_box[0:3]
        lwh = cluster_box[3:6]
        # Rotation around Z-axis
        axis_angle = np.array([0, 0, cluster_box[6] + 1e-10])

        rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angle)
        box3d = o3d.geometry.OrientedBoundingBox(center, rot, lwh)

        line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

        # Add additional lines (based on original pcl visualization logic)
        lines = np.asarray(line_set.lines)
        lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis = 0)

        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.paint_uniform_color((1, 0, 0)) # Red

        vis.add_geometry(line_set)

    # Draw Ground Truth boxes (Blue)
    for gt_box in gt_box_list:
        center = gt_box[0:3]
        lwh = gt_box[3:6]
        # Rotation around Z-axis
        axis_angle = np.array([0, 0, gt_box[6] + 1e-10])

        rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angle)
        box3d = o3d.geometry.OrientedBoundingBox(center, rot, lwh)

        line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

        # Add additional lines (based on original pcl visualization logic)
        lines = np.asarray(line_set.lines)
        lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis = 0)

        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.paint_uniform_color((0, 0, 1)) # Blue

        vis.add_geometry(line_set)

    vis.run()
    vis.destroy_window()

def voxel_filter(cloud, leaf_sizes):
    """
    Replaces pcl.VoxelGridFilter with open3d.voxel_down_sample.
    Note: Open3D's standard Voxel Downsampling uses a single voxel size for all axes.
    We use the minimum leaf size provided as a single voxel size for approximation.
    Args:
        cloud (o3d.geometry.PointCloud): Input point cloud.
        leaf_sizes (list): [size_x, size_y, size_z] voxel sizes.
    Returns:
        o3d.geometry.PointCloud: Voxel-filtered point cloud.
    """
    # Use the minimum leaf size as the representative voxel size
    voxel_size = np.min(leaf_sizes)
    cloud_voxel_filtered = cloud.voxel_down_sample(voxel_size=voxel_size)
    return cloud_voxel_filtered

def roi_filter(cloud, x_roi, y_roi, z_roi):
    """
    Replaces pcl.CropBox with open3d.geometry.AxisAlignedBoundingBox.
    Args:
        cloud (o3d.geometry.PointCloud): Input point cloud.
        x_roi (list): [x_min, x_max] for X-axis ROI.
        y_roi (list): [y_min, y_max] for Y-axis ROI.
        z_roi (list): [z_min, z_max] for Z-axis ROI.
    Returns:
        o3d.geometry.PointCloud: ROI-filtered point cloud.
    """
    xc_min, xc_max = x_roi
    yc_min, yc_max = y_roi
    zc_min, zc_max = z_roi

    # Define min/max bounds
    min_bound = np.array([xc_min, yc_min, zc_min])
    max_bound = np.array([xc_max, yc_max, zc_max])

    # Create Open3D AxisAlignedBoundingBox and crop
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    cloud_roi_filtered = cloud.crop(bbox)
    
    return cloud_roi_filtered

def plane_segmentation(cloud, dist_thold, max_iter):
    """
    Replaces pcl.SACMODEL_NORMAL_PLANE segmentation with open3d.segment_plane (RANSAC).
    Note: Open3D does not provide a direct Normal-based Plane Segmentation.
    `max_iter` is passed to `num_iterations` in Open3D's function.
    Args:
        cloud (o3d.geometry.PointCloud): Input point cloud.
        dist_thold (float): Distance threshold.
        max_iter (int): Maximum RANSAC iterations.
    Returns:
        tuple: (list of inlier indices, plane equation coefficients [a, b, c, d]).
    """
    # Open3D RANSAC plane segmentation: returns coefficients and inlier indices
    plane_model, indices = cloud.segment_plane(
        distance_threshold=dist_thold,
        ransac_n=3,
        num_iterations=max_iter
    )
    
    # Return as (indices, coefficients) for consistency with original pcl function
    return indices, plane_model

def clustering(cloud, tol, min_size, max_size):
    """
    Replaces pcl.EuclideanClusterExtraction with open3d.cluster_dbscan.
    Args:
        cloud (o3d.geometry.PointCloud): Input point cloud.
        tol (float): Clustering distance tolerance (eps).
        min_size (int): Minimum cluster size (min_points).
        max_size (int): Maximum cluster size (filtered in post-processing).
    Returns:
        list of lists: List of point indices for each cluster.
    """
    # Perform DBSCAN clustering
    # eps=tol, min_points=min_size
    labels = np.array(cloud.cluster_dbscan(eps=tol, min_points=min_size, print_progress=False))
    
    max_label = labels.max()
    cluster_indices = []
    
    # Iterate through all labels (excluding noise, which is -1)
    for i in range(max_label + 1):
        indices = np.where(labels == i)[0]
        # Apply max_size filtering (to mimic pcl's functionality)
        if len(indices) <= max_size:
            cluster_indices.append(indices.tolist())
            
    return cluster_indices

def get_cluster_box_list(cluster_indices, cloud_obsts):
    """
    Calculates Axis-Aligned Bounding Boxes (AABB) for each cluster.
    Args:
        cluster_indices (list of lists): List of point indices for each cluster.
        cloud_obsts (o3d.geometry.PointCloud): Obstacle point cloud (Open3D object).
    Returns:
        tuple: (List of cluster PointCloud objects, List of AABB coordinates [cx, cy, cz, dx, dy, dz, rz]).
    """
    cloud_cluster_list = []
    box_coord_list = []
    
    # Get numpy array of points from Open3D object
    points_obsts = np.asarray(cloud_obsts.points)

    for j, indices in enumerate(cluster_indices):
        # Extract points belonging to the current cluster
        points = points_obsts[indices]
        
        # Create Open3D PointCloud object for the cluster
        cloud_cluster = o3d.geometry.PointCloud()
        cloud_cluster.points = o3d.utility.Vector3dVector(points)
        cloud_cluster_list.append(cloud_cluster)

        # Calculate AABB (Axis-Aligned Bounding Box)
        x_max, x_min = np.max(points[:, 0]), np.min(points[:, 0])
        y_max, y_min = np.max(points[:, 1]), np.min(points[:, 1])
        z_max, z_min = np.max(points[:, 2]), np.min(points[:, 2])

        box = []

        # Center (cx, cy, cz)
        box.append((x_max + x_min) / 2)
        box.append((y_max + y_min) / 2)
        box.append((z_max + z_min) / 2)

        # Dimensions (dx, dy, dz)
        box.append((x_max - x_min))
        box.append((y_max - y_min))
        box.append((z_max - z_min))

        # Rotation (rz = 0 for AABB)
        box.append(0)

        box_coord_list.append(box)

    return cloud_cluster_list, box_coord_list

def get_fp_box(detect_box, gt_box):
    """
    Calculates False Positive (FP) boxes by checking overlap (IoU/2D box check)
    between detected boxes and ground truth boxes.
    Args:
        detect_box (np.ndarray): Detected boxes [cx, cy, cz, dx, dy, dz, rz].
        gt_box (np.ndarray): Ground truth boxes [cx, cy, cz, dx, dy, dz, rz].
    Returns:
        list: List of detected boxes considered False Positives.
    """
    fp_box = []

    for k in range(detect_box.shape[0]):
        # Calculate 2D coordinates for the detected box
        detect_x1 = detect_box[k][0] - detect_box[k][3] / 2
        detect_y1 = detect_box[k][1] - detect_box[k][4] / 2
        detect_x2 = detect_box[k][0] + detect_box[k][3] / 2
        detect_y2 = detect_box[k][1] + detect_box[k][4] / 2

        overlap = 0

        # Check for overlap with any Ground Truth box
        for n in range(gt_box.shape[0]):
            # Calculate 2D coordinates for the ground truth box
            gt_x1 = gt_box[n][0] - gt_box[n][3] / 2
            gt_y1 = gt_box[n][1] - gt_box[n][4] / 2
            gt_x2 = gt_box[n][0] + gt_box[n][3] / 2
            gt_y2 = gt_box[n][1] + gt_box[n][4] / 2

            # Calculate intersection width
            inter_w = min(detect_x2, gt_x2) - max(detect_x1, gt_x1)

            if inter_w >= 0:
                # Calculate intersection height
                inter_h = min(detect_y2, gt_y2) - max(detect_y1, gt_y1)

                # Check if there is significant overlap (inter_h >= -5 seems like a relaxed threshold check)
                if inter_h >= -5:
                    overlap = 1
                    break

        # If there is overlap (i.e., it's a True Positive/hit), skip
        if overlap > 0:
            continue

        # If no overlap, it's considered a False Positive
        fp_box.append(detect_box[k])

    return fp_box

def filter_gt_boxes(gt_boxes, gt_names):
    """
    Filters ground truth boxes based on object names (e.g., ignoring barriers, traffic cones).
    Args:
        gt_boxes (np.ndarray): All ground truth boxes.
        gt_names (list): List of ground truth object names.
    Returns:
        np.ndarray: Filtered ground truth boxes.
    """
    class_names = {'pedestrian': 2, 'ignore': -1, 'car': 1, 'motorcycle': 3, 'bicycle': 3, 'bus': 1, 'truck': 1, 'construction_vehicle': 1, 'trailer': 1, 'barrier': -1, 'traffic_cone': -1}

    new_box = []

    for i, gt_box in enumerate(gt_boxes):
        name = gt_names[i]
        class_id = class_names[name]

        # Filter out classes with ID -1 (ignored)
        if class_id == -1:
            continue

        new_box.append(gt_box)

    return np.array(new_box)

def save_fp_box(save_fold,
                cloud_np, 
                fp_box, 
                idx, 
                train = True):
    """
    Crops the original point cloud based on False Positive boxes, transforms points
    to local box coordinates, and saves the resulting point cloud data.
    Args:
        cloud_np (np.ndarray): The full point cloud data [x, y, z, intensity].
        fp_box (list): List of False Positive boxes.
        idx (int): Current frame index.
        train (bool): Flag to save data in 'train' or 'val' folder.
    """

    for box in fp_box:
        cx = box[0]
        cy = box[1]
        cz = box[2]

        dx = box[3]
        dy = box[4]
        dz = box[5]

        rz = box[6]

        # Calculate rotation matrix components for rotation in the opposite direction (-rz)
        cosa = math.cos(-rz)
        sina = math.sin(-rz)

        crop_point = []

        for point in cloud_np:
            x = point[0]
            y = point[1]
            z = point[2]

            # Z-axis check (height)
            if abs(z - cz) > dz / 2.:
                continue

            # Translate and rotate (around Z-axis) to local box coordinates
            local_x = (x - cx) * cosa + (y - cy) * (-sina)
            local_y = (x - cx) * sina + (y - cy) * cosa

            # XY-plane check
            if abs(local_x) < dx / 2.0 + 1e-2 and abs(local_y) < dy / 2.0 + 1e-2:
                # Store point in local box coordinates, retaining intensity
                new_point = np.array([x - cx, y - cy, z - cz, point[3]], dtype = np.float32)

                crop_point.append(new_point)

        # Skip boxes with too few points
        if len(crop_point) < 128:
            continue

        crop_point = np.array(crop_point)

        # Determine save path
        if train is True:
            class_fold = os.path.join(save_fold, os.path.join('train', '00_Neg'))
        else:
            class_fold = os.path.join(save_fold, os.path.join('val', '00_Neg'))

        if not os.path.isdir(class_fold):
            os.makedirs(class_fold)

        # Construct save file name
        save_path = os.path.join(class_fold, 'Nuscenes_{:06d}_{}_{}_{}_{}.npy'.format(idx, int(cx), int(cy), int(cz), crop_point.shape[0]))

        print(save_path)

        # Save the cropped point cloud
        np.save(save_path, crop_point)

def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--fold_path",
                        type=str,
                        default="nuscenes/v1.0-trainval")
    parser.add_argument("--train_file",
                        type=str,
                        default='nuscenes_infos_10sweeps_train.pkl')
    parser.add_argument("--val_file",
                        type=str,
                        default='nuscenes_infos_10sweeps_val.pkl')
    parser.add_argument("--save_fold",
                        type=str,
                        default="classification")
    
    return parser.parse_args()

def main():
    """
    Main execution logic: loads data, performs point cloud processing (filtering, segmentation, clustering),
    identifies False Positives, and saves the results.
    """
    args = make_parser()

    fold_path = args.fold_path
    
    train_path = os.path.join(fold_path, args.train_file)
    val_path = os.path.join(fold_path, args.val_file)

    # --- Processing Train Data ---
    with open(train_path, 'rb') as f:
        infos = pickle.load(f)

    for idx in range(len(infos)):
        info = infos[idx]
        lidar_path = os.path.join(fold_path, info['lidar_path'])

        # Load point cloud data [x, y, z, intensity, ring_index]
        cloud_np = np.fromfile(lidar_path, dtype = np.float32, count = -1).reshape([-1, 5])[:, :4]

        gt_boxes = info['gt_boxes']
        gt_names = info['gt_names']

        # Filter GT boxes (e.g., remove 'ignore' classes)
        gt_boxes = filter_gt_boxes(gt_boxes, gt_names)

        # PointCloud initialization using Open3D
        cloud_xyz = o3d.geometry.PointCloud()
        cloud_xyz.points = o3d.utility.Vector3dVector(cloud_np[:, 0:3])

        # Point Cloud Processing Pipeline (Open3D functions used)
        cloud_voxel_filtered = voxel_filter(cloud_xyz, [0.1, 0.1, 0.2])
        cloud_roi_filtered = roi_filter(cloud_voxel_filtered, [-51.2, 51.2], [5, 51.2], [-5, 3])
        indices, coefficients = plane_segmentation(cloud_roi_filtered, 0.3, 100)

        if len(indices) == 0:
            print('Could not estimate a planar model for the given dataset.')
            continue

        # Extract plane and non-plane (obstacle) points
        cloud_obsts = cloud_roi_filtered.select_by_index(indices, invert = True)
        
        # Clustering
        cluster_indices = clustering(cloud_obsts, 0.7, 10, 200)
        cloud_cluster_list, box_coord_list = get_cluster_box_list(cluster_indices, cloud_obsts)
        detect_box = np.array(box_coord_list)
        
        # Identify False Positives
        fp_box = get_fp_box(detect_box, gt_boxes)

        # Save False Positive point clouds
        save_fp_box(cloud_np, fp_box, idx, train = True)

    # --- Processing Validation Data ---
    with open(val_path, 'rb') as f:
        infos = pickle.load(f)

    for idx in range(len(infos)):
        info = infos[idx]
        lidar_path = os.path.join(fold_path, info['lidar_path'])

        print(os.path.join(fold_path, info['cam_front_path']))

        # Load point cloud data [x, y, z, intensity, ring_index]
        cloud_np = np.fromfile(lidar_path, dtype = np.float32, count = -1).reshape([-1, 5])[:, :4]

        gt_boxes = info['gt_boxes']
        gt_names = info['gt_names']

        # Filter GT boxes
        gt_boxes = filter_gt_boxes(gt_boxes, gt_names)

        # PointCloud initialization using Open3D
        cloud_xyz = o3d.geometry.PointCloud()
        cloud_xyz.points = o3d.utility.Vector3dVector(cloud_np[:, 0:3])

        # Point Cloud Processing Pipeline (Open3D functions used)
        cloud_voxel_filtered = voxel_filter(cloud_xyz, [0.1, 0.1, 0.2])
        cloud_roi_filtered = roi_filter(cloud_voxel_filtered, [-51.2, 51.2], [5, 51.2], [-5, 3])
        indices, coefficients = plane_segmentation(cloud_roi_filtered, 0.3, 100)

        if len(indices) == 0:
            print('Could not estimate a planar model for the given dataset.')
            continue

        # Extract plane and non-plane (obstacle) points
        cloud_obsts = cloud_roi_filtered.select_by_index(indices, invert = True)
        
        # Clustering
        cluster_indices = clustering(cloud_obsts, 0.7, 10, 100)
        cloud_cluster_list, box_coord_list = get_cluster_box_list(cluster_indices, cloud_obsts)
        detect_box = np.array(box_coord_list)
        
        # Identify False Positives
        fp_box = get_fp_box(detect_box, gt_boxes)

        # Save False Positive point clouds
        save_fp_box(args.save_fold, cloud_np, fp_box, idx, train = False)

if __name__ == '__main__':
    main()