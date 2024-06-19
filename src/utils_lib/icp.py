from math import cos, sin
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import copy
import time
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

    # Visualize the aligned point clouds
    # o3d.visualization.draw_geometries([aligned_pcd1, target_cloud])
    # x3 = p3[:, 0]
    # y3 = p3[:, 1]
    # fig = plt.figure()
    # ax2 = fig.add_subplot()
    # ax2.scatter(x1, y1, c='green', s=1) # original scan
    # ax2.scatter(x2, y2, c='blue', s=1) # matched scan
    # ax2.scatter(x3, y3, c='red', s=1) # aligned scan
    # ax2.legend(["source scan", "target scan", "aligned scan"])
    # ax2.set_title("scan matching using ICP")
    # # plt.savefig('/home/elias/catkin_ws/src/localization_lab/src/media'+str(np.round(time.time(), 2))+'.png')
    # plt.close()


    trans = np.array([x, y , theta]).reshape(3,1)
 
    
    return trans #transformation


