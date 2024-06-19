
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import copy
import time
from  utils_lib.helper_function import *
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
    print("Fitness:" , reg_p2p.fitness)
    
    return trans #transformation


