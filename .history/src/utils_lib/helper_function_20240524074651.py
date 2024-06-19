

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

def  scan_to_cartesian(scan_msg):

    cartesian_points = np.array([])
    ranges      = scan_msg.ranges
    angle_min = scan_msg.angle_min
    angle_max = scan_msg.angle_max
    angle_increment = scan_msg.angle_increment
    max_range = scan_msg.range_max
    min_range = scan_msg.range_min

    
    for i in range(len(ranges)):
        angle = angle_min + i*angle_increment
        if ranges[i] < max_range and ranges[i] > min_range:
            x = ranges[i]*cos(angle)
            y = ranges[i]*sin(angle)
            cartesian_points = np.append(cartesian_points, [x, y])
            cartesian_points = cartesian_points.reshape(-1, 2)



    return cartesian_points

def get_scan(msg):
    laser_projector = LaserProjection()
    point_cloud_msg = laser_projector.projectLaser(msg)

    point_cloud = pc2.read_points(point_cloud_msg, field_names=("x", "y"), skip_nans=True)
    points = []

    for point in point_cloud:
        points.append([float(point[0]), float(point[1])])

    new_point_cloud = np.array(points).reshape(-1,2)
    # print("scan" , new_point_cloud)

    
    return new_point_cloud

def scan_to_robot(scan):
    # Convert the scan from the lidar  frame to the robot frame
    scan_robot = np.zeros(scan.shape)
    for i in range(scan.shape[0]):
        scan_robot[i, 0] = scan[i, 0] + 0  # difference b.w the lidar and the robot center
        scan_robot[i, 1] = scan[i, 1] + 0 # 
    return scan_robot
def scan_to_world(scan, pose):
    # Convert the scan from the robot frame to the world frame
   
    scan_world = np.zeros(scan.shape)
    for i in range(scan.shape[0]):
        scan_world[i, 0] = pose[0] + scan[i, 0]*cos(pose[2]) - scan[i, 1]*sin(pose[2])
        scan_world[i, 1] = pose[1] + scan[i, 0]*sin(pose[2]) + scan[i, 1]*cos(pose[2])
    return scan_world

def build_map(scans, poses):
    # Convert the scan from the robot frame to the world frame
    map = []
    print("scan" , scans.shape )
    print("pose" , poses.shape )

    j = len(scans)
    for l,scan in enumerate(scans):
        scan_world = np.zeros(scan.shape)
        j = l*3
        for i in range(scan.shape[0]):
            
            scan_world[i, 0] = poses[j+0][0] + scan[i, 0]*cos(poses[j+2][0]) - scan[i, 1]*sin(poses[j+2][0])
            scan_world[i, 1] = poses[j+1][0] + scan[i, 0]*sin(poses[j+2][0]) + scan[i, 1]*cos(poses[j+2][0])
        map.append(scan_world)
        
    return map


def build_map(scans, poses):
    # Convert the scan from the robot frame to the world frame
    map = []
    j = len(scans)
    for l,scan in enumerate(scans):
        scan_world = np.zeros(scan.shape)
        j = l*3
        for i in range(scan.shape[0]):
            
            scan_world[i, 0] = poses[j+0][0] + scan[i, 0]*cos(poses[j+2][0]) - scan[i, 1]*sin(poses[j+2][0])
            scan_world[i, 1] = poses[j+1][0] + scan[i, 0]*sin(poses[j+2][0]) + scan[i, 1]*cos(poses[j+2][0])
        map.append(scan_world)
        
    return map


def get_eculidean_distance(p1, p2):
    ''''''
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
def check_scan_thershold(xk, dist_th, ang_th):

    last_scan_pose = xk[-6:-3]  # 2nd last state in state vector
    curr_pose = xk[-3:]         # last state in state vector
    
    dist_since_last_scan = get_eculidean_distance(last_scan_pose[:2], curr_pose[:2]) 
    rot_since_last_scan = abs(last_scan_pose[2] - curr_pose[2])
  
    # only add pose/scan if we have move significantly
    if dist_since_last_scan > dist_th or rot_since_last_scan > ang_th:
         print("Update ")
         return True
    else:
         return False