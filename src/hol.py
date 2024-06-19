#!/usr/bin/python3
import rospy
from geometry_msgs.msg import Twist , Quaternion
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
import numpy as np
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from tf import transformations as tf 
import tf
import tf2_ros
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
import tf.transformations
from utils_lib.POSE_SLAM import PoseSLAMEKF

from utils_lib.ICP_Reg import ICP
from utils_lib.Pose import Pose3D
from std_msgs.msg import Float64MultiArray
from math import atan2 , sqrt , degrees , radians , pi , floor , cos , sin
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray  
from geometry_msgs.msg import Point
from sensor_msgs import point_cloud2
from std_msgs.msg import ColorRGBA
from numpy.linalg import eig
import scipy.linalg
import scipy
import threading
import copy
from  utils_lib.helper_function import *
import threading


class DifferentialDrive:
    def __init__(self) -> None:

        #Initilaize the state of the robot
        self.lock = threading.Lock()
        self.xk      = np.array([0.0, 0.0, 0.0]).reshape(3, 1) # Initialize the state of the robot
        # self.xk      = np.array([3., -0.78, np.pi/2]).reshape(3, 1) # Initialize the state of the robot
        self.Pk      = np.eye(3)*0.000   # Initialize the covariance of the state
        self.map     =  [] # Initialize the map
        self.scan    =  [] # Initialize the scan
        self.xk_dr   = np.array([0.0,0.0,0.0]).reshape(3, 1) # Initialize the state of the robot
        self.xk_scan = np.array([0.0,0.0,0.0]).reshape(3, 1) # Initialize the state of the robot
        self.xk_imu  = np.array([0.0,0.0,0.0]).reshape(3, 1) # Initialize the state of the robot
        self.xk_slam = np.array([0.0,0.0,0.0]).reshape(3, 1) # Initialize the state of the robot
        # Initialize the covariance of the process 
      
        # state vector for dead reckoning
        self.XK_DR = np.array([0.0,0.0,0.0]).reshape(3, 1) # Initialize the covariance of the state
        self.PK_DR = np.eye(3)*0.0# Initialize the covariance of the state
      

        self.Pk_dr = np.eye(3)*0.00# Initialize the covariance of the state
        self.Pk_slam = np.eye(3)*0.00# Initialize the covariance of the state
        self.scan_cartesian = []
        
        self.Qk= np.array([[20**2, 0 ],    
                           [0, 20**2],
                           ]) # Initialize the covariance of the process noise  
        
        self.R_icp = np.array([ [0.1**2 ,0,0],
                                [0,0.1**2,0],
                                [0,0,0.1**2]])
        # Initialize the 
        self.parent_frame = "world_ned"
        # self.child_frame  = "turtlebot/base_footprint"
        self.child_frame  = "turtlebot/kobuki/base_footprint"
        self.rplidar_frame = "turtlebot/kobuki/rplidar"
        # self.wheel_name_left = "turtlebot/wheel_left_joint"
        # self.wheel_name_right = "turtlebot/wheel_right_joint"
       
        self.wheel_name_left = "turtlebot/kobuki/wheel_left_joint"
        self.wheel_name_right = "turtlebot/kobuki/wheel_right_joint"
        self.left_wheel_velocity   = 0
        self.right_wheel_velocity  = 0
        self.left_wheel_velo_read  = False
        self.right_wheel_velo_read = False
        self.last_time = None
        self.yaw_read = False
        self.wheel_radius = 0.035
        self.wheel_base = 0.235 
        self.scan_th_distance = 1
        self.scan_th_angle =np.pi/2
        
        self.mutex = threading.Lock()

        self.pse = PoseSLAMEKF(self.xk, self.Pk, self.Qk, self.wheel_base , self.wheel_radius)
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.tf_buff = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buff)
        # Create a publisher
        # odom publisher
        self.odom_pub = rospy.Publisher("/odom", Odometry, queue_size=10)
        self.covariance_markers = rospy.Publisher('/covariance_eigen_markers', MarkerArray, queue_size=10)
        # Viewpoints visualizer
        self.viewpoints_pub = rospy.Publisher("/slam/vis_viewpoints",MarkerArray,queue_size=1)
        self.full_map_pub = rospy.Publisher('/slam/map', PointCloud2, queue_size=10)
        self.full_map_pub_dr = rospy.Publisher('/dr/map', PointCloud2, queue_size=10)


        # joint1 velocity publisher
        self.vel_pub = rospy.Publisher('/kobuki/kobuki/commands/wheel_velocities', Float64MultiArray , queue_size=10)
        self.vel_pub = rospy.Publisher('/turtlebot/kobuki/commands/wheel_velocities', Float64MultiArray , queue_size=10)
        # # # Create a subscriber
        
        rospy.Subscriber("/kobuki/kobuki/sensors/rplidar", LaserScan, self.check_scan)
        rospy.Subscriber("/turtlebot/kobuki/sensors/rplidar", LaserScan, self.check_scan)

        # joint state subscriber
        rospy.Subscriber('/turtlebot/joint_states', JointState, self.joint_state_callback)
        rospy.Subscriber('/kobuki/joint_states', JointState, self.joint_state_callback)
        # velocity subscriber
        rospy.Subscriber('/cmd_vel', Twist, self.velocity_callback)
        
         # imu subscriber
        rospy.Subscriber('/turtlebot/kobuki/sensors/imu_data', Imu, self.imu_callback)
        rospy.Subscriber('/kobuki/kobuki/sensors/imu_data', Imu, self.imu_callback)

        
    def wrap_angle(self, angle):
        """this function wraps the angle between -pi and pi

        :param angle: the angle to be wrapped
        :type angle: float

        :return: the wrapped angle
        :rtype: float
        """
        return (angle + ( 2.0 * np.pi * np.floor( ( np.pi - angle ) / ( 2.0 * np.pi ) ) ) )
    
    def write_to_file(self):
        
        #creare a publisher
        # with open('/home/elias/catkin_ws/src/localization_lab/deadre.txt', 'a') as f:
        #             # print("create prediction file ")
        #             xk1 = self.xk_dr
        #             np.savetxt(f, xk1.reshape(1,3))
       
        # with open('/home/elias/catkin_ws/src/localization_lab/imu.txt', 'a') as f:
        #             # print("create imu file ")
        #             xk3 = self.xk_imu
        #             np.savetxt(f, xk3.reshape(1,3)) 
        
        with open('/home/elias/catkin_ws/src/localization_lab/slam.txt', 'a') as f:
                    # print("create slam file ")
                    xk5 = self.xk_slam
                    np.savetxt(f, xk5.reshape(1,3))

        with open('/home/elias/catkin_ws/src/localization_lab/slam_pk.txt', 'a') as f:
                    # print("create slam file ")
                    pk1 = self.Pk_slam
                    np.savetxt(f, pk1)
        # with open('/home/elias/catkin_ws/src/localization_lab/dr_pk.txt', 'a') as f:
        #             # print("create slam file ")
        #             pk2 = self.Pk_dr
        #             np.savetxt(f, pk2)
        
    def write_to_file_scan(self):
        
        #creare a publisher
        # with open('/home/elias/catkin_ws/src/localization_lab/deadre.txt', 'a') as f:
        #             # print("create prediction file ")
        #             xk1 = self.xk_dr
        #             np.savetxt(f, xk1.reshape(1,3))
       
        with open('/home/elias/catkin_ws/src/localization_lab/scan_xk.txt', 'a') as f:
                    # print("create imu file ")
                    xk3 = self.xk_slam
                    np.savetxt(f, xk3.reshape(1,3))
        with open('/home/elias/catkin_ws/src/localization_lab/scan.txt', 'a') as f:
                    # print("create slam file ")
                    scan = self.scan_cartesian
                    f.write("scan\n")
                    np.savetxt(f, scan)
    def joint_state_callback(self, msg):
       
     
        self.time_stamp = msg.header.stamp

        if self.yaw_read and self.last_time == None:
           self.last_time = rospy.Time(msg.header.stamp.secs, msg.header.stamp.nsecs).to_sec()
                        
        if(self.yaw_read and msg.name[0] == self.wheel_name_left and msg.name[1] ==self.wheel_name_right):
            
            # print("right")
            self.right_wheel_velocity = msg.velocity[1]
            
            self.left_wheel_velocity = msg.velocity[0]  
            self.left_wheel_velo_read = False
            self.right_wheel_velo_read = False
            self.left_linear_vel = self.left_wheel_velocity * self.wheel_radius
            self.right_linear_vel = self.right_wheel_velocity * self.wheel_radius 
            self.v = (self.left_linear_vel + self.right_linear_vel) / 2
            self.w = (self.left_linear_vel - self.right_linear_vel) / self.wheel_base
            
            time_secs = msg.header.stamp.secs
            time_nsecs = msg.header.stamp.nsecs
            # print("Now" , rospy.Time.now().to_sec())
            self.current_time = rospy.Time.from_sec(msg.header.stamp.secs + msg.header.stamp.nsecs*1e-9).to_sec()
            self.dt = self.current_time - self.last_time   
            # print("time" , self.last_time , self.current_time , self.dt)     
            self.last_time = self.current_time
            # print("PK" , self.Pk)
            uk = np.array([self.v*self.dt, 0, self.w*self.dt]).reshape(3, 1)
            # print(self.v , self.w , uk)
            # print("angle" , self.xk[2] , self.yaw)
            with self.lock:
                self.xk , self.Pk = self.pse.Prediction( self.xk , self.Pk , uk , self.dt)
                self.xk_slam = self.xk[-3:].reshape(3,1)
                self.pk_slam = self.Pk[-3:,-3:]
                # self.XK_DR , self.PK_DR = self.pse.Prediction(self.XK_DR,self.PK_DR  ,uk ,self.dt)
            
            # print("XK" ,self.xk )
            # print("PK" , self.Pk)
            self.write_to_file()
            self.publish_odometry(msg)
         
        
    def publish_odometry(self ,msg):
       
        odom = Odometry()
        
        current_time = rospy.Time.from_sec(self.time_stamp.secs + self.time_stamp.nsecs * 1e-9)
     
        theta = self.xk[-1].copy()
        q = quaternion_from_euler(0, 0, float(theta))
        
        covar = [   self.Pk[-3,-3], self.Pk[-3,-2], 0.0, 0.0, 0.0, self.Pk[-3,-1],
                    self.Pk[-2,-3], self.Pk[-2,-2], 0.0, 0.0, 0.0, self.Pk[-2,-1],  
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                    self.Pk[-1,-3], self.Pk[-1,-2], 0.0, 0.0, 0.0, self.Pk[-1,-1]]

        odom.header.stamp = current_time
        odom.header.frame_id = self.parent_frame
        odom.child_frame_id = self.child_frame
    
        odom.pose.pose.position.x = self.xk[-3]
        odom.pose.pose.position.y = self.xk[-2]

        odom.pose.pose.orientation.x = q[0]
        odom.pose.pose.orientation.y = q[1]
        odom.pose.pose.orientation.z = q[2]
        odom.pose.pose.orientation.w = q[3]

        odom.twist.twist.linear.x = self.v
        odom.twist.twist.angular.z = self.w
        odom.pose.covariance = covar
        # print("odom publisher")
        self.odom_pub.publish(odom)
     
        self.tf_broadcaster.sendTransform((self.xk[-3], self.xk[-2], 0.0), q ,self.time_stamp, self.child_frame, self.parent_frame)
   
    def imu_callback(self, msg):
       
        orientation = msg.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]

        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)

        self.yaw = self.wrap_angle(-yaw)
        # print("Observation yaw"  , self.yaw ,  self.xk[2]) 
        if(self.yaw_read == False):
            with  self.lock:
                self.xk[2] = self.yaw
                # self.XK_DR[2] = self.yaw
                self.yaw_read = True
            print("Observation yaw" ,self.yaw , self.xk) 
            
        else:
            self.yaw = np.array([self.yaw]).reshape(1, 1)
            self.heading_update()
        

    def transform_cloud(self, target_frame, source_frame, scan):
        try:
            # print("transforming", target_frame, source_frame)
            # Create a LaserProjection object
            projector = LaserProjection()

            # Convert the LaserScan to PointCloud2
            cloud = projector.projectLaser(scan)    
                
            # Wait for the transform to become available
            self.tf_buff.can_transform(target_frame, source_frame,scan.header.stamp, rospy.Duration(1.0))
            
            transform = self.tf_buff.lookup_transform(target_frame, source_frame, scan.header.stamp)
            transformed_cloud = do_transform_cloud(cloud , transform )

            # Convert the PointCloud2 to a list of (x, y) tuples
            xy_points = [[point[0], point[1]] for point in point_cloud2.read_points(transformed_cloud, field_names=("x", "y"), skip_nans=True)]
            # print("transform", xy_points)
            return xy_points
            # return transformed_cloud
           
            # Continue processing with the transform...
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr("Transform error: %s", e)
            return None
       
    def check_scan(self, scan):
        
        scan = self.transform_cloud(self.child_frame , self.rplidar_frame , scan)
        # print(scan)
        self.scan_cartesian = []
        # Convert laser scan data to x, y coordinates in robot frame 
        if(scan):
             self.scan_cartesian = np.array(scan)
        
        if(self.yaw_read and  len(self.map) == 0 and len(self.scan_cartesian)):
            self.publish_covariance_marker()
            # self.scan_cartesian = scan_to_robot(self.scan_cartesian)
           
            self.scan_world = scan_to_world(self.scan_cartesian, self.xk[-3:])
            with self.lock:    
                self.scan.append(self.scan_cartesian)
                self.map.append(self.scan_world)         
                self.xk , self.Pk = self.pse.Add_New_Pose(self.xk, self.Pk)
            # self.XK_DR, self.PK_DR = self.pse.Add_New_Pose(self.XK_DR, self.PK_DR)
            
            self.create_map()   
               
            self.write_to_file()
        
        elif(self.yaw_read and  len(self.scan_cartesian) and check_scan_thershold(self.xk,self.scan_th_distance, self.scan_th_angle)):
           
            self.create_map()   
            self.publish_covariance_marker()
    
            self.scan_world = scan_to_world(self.scan_cartesian  , self.xk[-3:])
            with self.lock: 
                self.map.append(self.scan_world)
                self.scan.append(self.scan_cartesian)

                self.xk , self.Pk = self.pse.Add_New_Pose(self.xk, self.Pk)
            # self.XK_DR, self.PK_DR = self.pse.Add_New_Pose(self.XK_DR, self.PK_DR)
            
            Ho = self.pse.OverlappingScan(self.xk)
            
            print("------------------------------------------------")
            print("Ho",Ho)
            zp = np.zeros((0,1)) # predicted scan
            Rp = np.zeros((0,0)) # predicted scan covariance
            Hp = []
            i =0 
            for j in Ho:
                print("check scan 1",j)
                jXk = self.pse.hfj(self.xk, j)
                jPk = self.pse.jPk(self.xk, self.Pk,  j)

                xk_robot = self.xk[-3:].reshape(3,1)
                xk_scan = self.xk[j:j+3,:].reshape(3,1)
                distance = xk_robot - xk_scan
                distance[2] = self.wrap_angle(distance[2])
            
                matched_scan = self.scan[j]
                current_scan = self.scan[-1]
                xk  = self.xk[-3:].reshape(3,1)
                # ICP Registration 
                zr  = ICP(matched_scan, current_scan, jXk)
                matched_pose = self.xk[j:j+3,:].reshape((3,1))
                Rr = self.R_icp
                isCompatibile = self.pse.ICNN(jXk, jPk , zr, Rr )
                print("isCompatibile",isCompatibile)
                if(isCompatibile):
                    zp = np.block([[zp],[zr]])
                    Rp = scipy.linalg.block_diag(Rp, Rr)
                    Hp.append(j)
            if(len(Hp)>0):
                zk ,Rk, Hk,Vk = self.pse.ObservationMatrix(self.xk , Hp,zp,Rp )
                print("observation matrix")
                # print("zk",zk)
                # print("Rk",Rk)
                # print("Hk",Hk)
                # print("Vk",Vk)
                with self.lock:
                    self.xk , self.Pk = self.pse.Update(self.xk, self.Pk, Hk, Vk, zk, Rk,Hp)
                    self.xk_slam = self.xk[-3:].reshape(3,1)
                    self.Pk_slam = self.Pk[-3:,-3:]
   
                
            # print("--------------------------------------------------")
            self.write_to_file()
            self.write_to_file_scan()
        if(len(self.map) > 1):
            self.publish_viewpoints()   
            # self.create_map_dr()
            self.create_map()
     
#############################
        # Update step
############################
    def heading_update(self):
            # Create a row vector of zeros of size 1 x 3*num_poses
            self.compass_Vk = np.diag([1])
            # define the covariance matrix of the compass
            self.compass_Rk = np.diag([0.01**2]) 
            # print("imu update")   
            Hk = np.zeros((1, len(self.xk)))
            # Replace the last element of the row vector with 1
            Hk[0, -1] = 1
            predicted_compass_meas = self.xk[-1]  
            # Compute the kalman gain
            K = self.Pk @ Hk.T @ np.linalg.inv((Hk @ self.Pk @ Hk.T) + (self.compass_Vk @ self.compass_Rk @ self.compass_Vk.T))
            # Compute the innovation
            innovation = np.array(self.wrap_angle(self.yaw[0] - predicted_compass_meas)).reshape(1, 1)
            # Update the state vector
            I = np.eye(len(self.xk))
            with self.lock:
                self.xk = self.xk + K@innovation
                # Create the identity matrix        
                # Update the covariance matrix
                self.Pk = (I - K @ Hk) @ self.Pk @ (I - K @ Hk).T
     
            # Hkdr = np.zeros((1, len(self.XK_DR)))
            # # Replace the last element of the row vector with 1
            # Hkdr[0, -1] = 1
            # predicted_compass_meas = self.XK_DR[-1]

            # K = self.PK_DR @ Hkdr.T @ np.linalg.inv((Hkdr @ self.PK_DR @ Hkdr.T) + (self.compass_Vk @ self.compass_Rk @ self.compass_Vk.T))
            
            # I = np.eye(len(self.XK_DR))
            # self.XK_DR = self.XK_DR + K@innovation
            # self.PK_DR = (I - K @ Hkdr) @ self.PK_DR @ (I - K @ Hk).T
            # self.xk_dr = self.XK_DR[-3:]
            # self.Pk_dr = self.PK_DR[-3:,-3:]
            self.write_to_file()
       
        
    def create_map(self):
        scan_slam = self.scan.copy()
        full_map_slam = np.zeros((0,3))
        map_slam= build_map(scan_slam,self.xk)
        for m in map_slam:
            if(len(m) == 0):
                continue
            z = np.zeros((len(m), 1))
            full_map_slam = np.block([[full_map_slam] ,[m,z]])
            
        
        # Create the header for the point cloud message
        header = rospy.Header()
        header.stamp = self.time_stamp
        header.frame_id = 'world_ned'  # Set the frame ID

        # # Create the point cloud message
        point_cloud_msg = point_cloud2.create_cloud_xyz32(header, full_map_slam)

        self.full_map_pub.publish(point_cloud_msg)
    def create_map_dr(self):
        # print(self.scan)
        scan_dr = self.scan.copy()
        z = np.zeros((len(self.map), 1))
        
        full_map_dr = np.zeros((0,3))

        map = build_map(scan_dr,self.XK_DR)

        for m in map:
            if(len(m) == 0):
                continue
            z = np.zeros((len(m), 1))
            full_map_dr = np.block([[full_map_dr] ,[m,z]])

        # Create the header for the point cloud message
        header = rospy.Header()
        header.stamp = self.time_stamp
        header.frame_id = 'world_ned'  # Set the frame ID

        # # Create the point cloud message
        point_cloud_msg2 = point_cloud2.create_cloud_xyz32(header,full_map_dr)

        self.full_map_pub_dr.publish(point_cloud_msg2)
    def publish_viewpoints(self):

        marker_frontier_lines = MarkerArray()
        marker_frontier_lines.markers = []

        viewpoints_list = []

        for i in range(0,len(self.xk),3):
            myMarker = Marker()
            myMarker.header.frame_id = "world_ned"
            myMarker.header.stamp = rospy.Time.now()
            myMarker.type = myMarker.SPHERE
            myMarker.action = myMarker.ADD
            myMarker.id = i

            myMarker.pose.orientation.x = 0.0
            myMarker.pose.orientation.y = 0.0
            myMarker.pose.orientation.z = 0.0
            myMarker.pose.orientation.w = 1.0

            myPoint = Point()
            myPoint.x = self.xk[i]
            myPoint.y = self.xk[i+1]

            myMarker.pose.position = myPoint
            myMarker.color=ColorRGBA(0.224, 1, 0, 1)

            myMarker.scale.x = 0.1
            myMarker.scale.y = 0.1
            myMarker.scale.z = 0.05
            viewpoints_list.append(myMarker)

        self.viewpoints_pub.publish(viewpoints_list)
    # Define function to create marker message for eigenvalues and eigenvectors
    def publish_covariance_marker(self):
       # Create marker array message
        marker_array  = MarkerArray()
        # Define marker properties
        id = 0
        markers = []
        for j in range(0,len(self.xk),3):
            
            eigenvalues, eigenvectors = eig(self.Pk[j:j+2, j:j+2])    
            marker = Marker()
            marker.header.frame_id = "world_ned"  # Set the frame            
            marker.type   =  Marker.SPHERE
            marker.action =  Marker.ADD
            # Check if eigenvalues are valid
            marker_scale = [0.01, 0.01, 0.0001]
            if np.any(np.isnan(eigenvalues)) or np.any(np.isinf(eigenvalues)) or np.any(eigenvalues < 0):
                print("Invalid eigenvalues:", eigenvalues)
            else:
                marker_scale  = [4*np.sqrt(eigenvalues[0]), 4*np.sqrt(eigenvalues[1]) , 0.00001 ]


            marker_color  = [1.0, 1.0, 0.0, 1.0]  # Red colo
            id += 1
            marker.id = id 
            marker.scale.x = marker_scale[0]
            marker.scale.y = marker_scale[1]
            marker.scale.z = 0.1
            marker.color.r = marker_color[0]
            marker.color.g = marker_color[1]
            marker.color.b = marker_color[2]
            marker.color.a = marker_color[3]
            marker.pose.position.x = self.xk[j]
            marker.pose.position.y = self.xk[j+1]
            marker.pose.position.z = 0.0
            # Use eigenvectors to determine the orientation of the ellipsoid

            angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
            quat = tf.transformations.quaternion_from_euler(0, 0, angle)
            
       
            # Normalize the quaternion
            quat_magnitude = np.sqrt(quat[0]**2 + quat[1]**2 + quat[2]**2 + quat[3]**2)
            quat = [q/quat_magnitude for q in quat]

            marker.pose.orientation.x = quat[0]
            marker.pose.orientation.y = quat[1]
            marker.pose.orientation.z = quat[2]
            marker.pose.orientation.w = quat[3]
            markers.append(marker)

            
        marker_array.markers.extend(markers)

        self.covariance_markers.publish(marker_array)
    def velocity_callback(self, msg):
        
        lin_vel = msg.linear.x
        ang_vel = msg.angular.z

        # print("linear and angular ", lin_vel , ang_vel )
        left_linear_vel   = lin_vel  + (ang_vel*self.wheel_base/2)
        right_linear_vel = lin_vel   -  (ang_vel*self.wheel_base/2)
 
        left_wheel_velocity  = left_linear_vel / self.wheel_radius
        right_wheel_velocity = right_linear_vel / self.wheel_radius
        
        # print("left_wheel_velocity",left_wheel_velocity , right_wheel_velocity)
        
        wheel_vel = Float64MultiArray()
        wheel_vel.data = [left_wheel_velocity, right_wheel_velocity]
        self.vel_pub.publish(wheel_vel)
# Main function
if __name__ == '__main__':
    # Initialize the node
    rospy.init_node('differential_robot_node')
    # Create an instance of the DifferentialDrive class
    diff_drive = DifferentialDrive()
    # Spin
    rospy.spin()