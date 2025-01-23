import numpy as np
import open3d as o3d

# Load the point cloud data from the .npy file
point_clouds = np.load('results/GEN_Ours_ibs_1735542129/out.npy')

# Iterate through each point cloud and visualize it using Open3D
for i, points in enumerate(point_clouds):
    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    
    # Set the points of the PointCloud object
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd], window_name=f'Point Cloud {i+1}')