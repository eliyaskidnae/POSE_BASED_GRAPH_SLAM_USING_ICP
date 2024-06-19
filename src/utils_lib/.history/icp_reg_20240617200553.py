
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import copy
import time
# from  utils_lib.helper_function import *
from math import cos, sin
import numpy as np
from laser_geometry import LaserProjection
import sensor_msgs.point_cloud2 as pc2
def compose_transform_matrix(x, y, theta):
    # Compose a 3x3 transformation matrix from x, y, theta
    '''
    This function takes in the x, y, and theta values and returns a 3x3 transformation matrix.
    
    Inputs:
    - x: The x value of the transformation matrix
    - y: The y value of the transformation matrix
    - theta: The theta value of the transformation matrix
    
    Outputs:
    - T: The 4x4 transformation matrix'''
    T = np.array([[cos(theta), -sin(theta),0, x],
                  [sin(theta), cos(theta), 0,y],
                  [0, 0, 1 , 0],
                  [0, 0, 0, 1]])
    return T
def decompose_transform_matrix(T):
    # Decompose a 3x3 transformation matrix into x, y, theta
    '''
    This function takes in a 3x3 transformation matrix and returns the x, y, and theta values.

    Inputs:
    - T: The 4x4 transformation matrix
    Outputs:
    - x: The x value of the transformation matrix
    - y: The y value of the transformation matrix
    - theta: The theta value of the transformation matrix
    '''

    x = T[0, 3]
    y = T[1, 3]
    theta = np.arctan2(T[1, 0], T[0, 0])
    return x, y, theta

# from helper_function import compose_transform_matrix, decompose_transform_matrix 

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])
 


def ICP(MatchedScan , CurrentScan , initial_guess): #, MatchedVp, CurrentVp

    '''
    This function takes in two point clouds and returns the transformation matrix 
    that aligns the two point clouds.
    
    Inputs:
    - MatchedScan: The first point cloud to be aligned
    - CurrentScan: The second point cloud to be aligned
    - initial_guess: The initial guess for the transformation matrix
        
    Outputs:
    - zr: The transformation matrix that aligns the two point clouds
    '''

    x1 = np.copy(CurrentScan[:,0])
    y1 = np.copy(CurrentScan[:,1])
    x2 = np.copy(MatchedScan[:,0])
    y2 = np.copy(MatchedScan[:,1])

    # Add a column of zeros to the point clouds to make them 3D
    temp_column = np.zeros(CurrentScan.shape[0])
    source_points = np.hstack((CurrentScan, temp_column.reshape(-1, 1)))
 
    # Add a column of zeros to the point clouds to make them 3D
    temp_column = np.zeros(MatchedScan.shape[0])
    target_points = np.hstack((MatchedScan, temp_column.reshape(-1, 1)))

    # Create Open3D point cloud objects
    source_cloud = o3d.geometry.PointCloud()
    source_cloud.points = o3d.utility.Vector3dVector(source_points)

    target_cloud = o3d.geometry.PointCloud()
    target_cloud.points = o3d.utility.Vector3dVector(target_points)
   

    initial_guess = initial_guess.flatten()
    #create a 4x4 transformation matrix out of the initial guess

    initial_guess = compose_transform_matrix(initial_guess[0], initial_guess[1], initial_guess[2])

    #convert initial guess to float64
    initial_guess = initial_guess.astype(np.float64)

    # Perform registration
    reg_p2p = o3d.pipelines.registration.registration_icp( source_cloud,
               target_cloud, 0.1 , initial_guess,
               o3d.pipelines.registration.TransformationEstimationPointToPoint(),
               o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    
    transformation = reg_p2p.transformation
    # print("Transformation is:")
    translation = transformation[0:2, 3]
    theta = np.arctan2(transformation[1, 0], transformation[0, 0])

    x ,y , theta = decompose_transform_matrix(transformation)
    
    
    aligned_pcd1 = source_cloud.transform(transformation)
    p3 = np.asarray(aligned_pcd1.points)



    trans = np.array([x, y , theta]).reshape(3,1)
 
    
    return trans #transformation


