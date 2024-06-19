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
from utils_lib.POSE_SLAM_EKF import PoseSLAMEKF
from utils_lib.icp_reg import ICP
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
from  utils_lib.helper_function import *

class DifferentialDrive:
    def __init__(self) -> None:

        #Initilaize the state of the robot
        self.xk      = np.array([0, 0, 0]).reshape(3, 1) # Initialize the state of the robot
        # self.xk      = np.array([3., -0.78, np.pi/2]).reshape(3, 1) # Initialize the state of the robot
        self.Pk      = np.eye(3)*0.00001# Initialize the covariance of the state
        self.map     =  [ ] # Initialize the map
        self.scan    =  [ ] # Initialize the scan
        self.scan_cartesian = []
        # motion model uncertainty   
        self.Qk= np.array([[0.2,0 ],    
                           [0,0.2],
                           ]) # Initialize the covariance of the process noise  
        # uncertainty of the icp registration
        self.R_icp = np.array([[0.1**2 ,0,0],
                              [0,0.1**2,0],
                              [0,0,0.1**2]])
        # compass uncertainty
        self.compass_Vk = np.diag([1])
        # define the covariance matrix of the compass
        self.compass_Rk = np.diag([0.1**2]) 
         # Initialize the 
        self.parent_frame = "world_ned"
        # self.child_frame  = "turtlebot/base_footprint"
        self.child_frame  = "turtlebot/kobuki/base_footprint"
        self.rplidar_frame  = "turtlebot/kobuki/rplidar"

        # self.wheel_name_left = "turtlebot/wheel_left_joint"
        # self.wheel_name_right = "turtlebot/wheel_right_joint"
       
        self.wheel_name_left = "turtlebot/kobuki/wheel_left_joint"
        self.wheel_name_right = "turtlebot/kobuki/wheel_right_joint"
        self.last_time = None
        self.yaw_read = False

        # self.wheel_name_left  = "turtlebot/wheel_left_joint"
        # self.wheel_name_right = "turtlebot/wheel_right_joint"
        # self.parent_frame  = "world_ned"
        # self.child_frame   = "turtlebot/base_footprint"
        # self.rplidar_frame = "turtlebot/rplidar"
        self.left_wheel_velocity   = 0
        self.right_wheel_velocity  = 0
        self.left_wheel_velo_read  = False
        self.right_wheel_velo_read = False

        self.last_time    = rospy.Time.now().to_sec()
        self.wheel_radius = 0.035 # wheel radius
        self.wheel_base   = 0.235 # wheel base

        # scan matching variables 
        self.min_scan_th_distance = 0.8  # minimum scan taking distance  
        self.min_scan_th_angle = np.pi # take scan angle thershold
        self.overlapping_check_th_dis = 1 # ovrlapping checking distance thershold 
        self.max_scan_history = 33 # maximum amount of scan history to store 
        self.num_of_scans_to_remove = 4 # number of scans to remove from the history
        self.max_scans_to_match = 5 # maximum number of scans to match

        #Initilaize the state of the robot
        self.lock = threading.Lock()
        self.mutex = threading.Lock()
        # create object of pose based slam 
        self.pse = PoseSLAMEKF(self.xk, self.Pk, self.Qk,self.compass_Rk , self.compass_Vk , self.wheel_base , self.wheel_radius , self.overlapping_check_th_dis)
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.tf_buff = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buff)
        # Create a publisher
        # odom publisher
        self.odom_pub = rospy.Publisher("/odom", Odometry, queue_size=10)
        # Viewpoints visualizer
        self.viewpoints_pub = rospy.Publisher("/slam/vis_viewpoints",MarkerArray,queue_size=1)
        self.full_map_pub = rospy.Publisher('/slam/map', PointCloud2, queue_size=10)
        self.full_map_pub_dr = rospy.Publisher('/dr/map', PointCloud2, queue_size=10)
        self.full_map_pub_gt = rospy.Publisher('/gt/map', PointCloud2, queue_size=10)
        # joint1 velocity publisher
        self.vel_pub = rospy.Publisher('/turtlebot/kobuki/commands/wheel_velocities', Float64MultiArray , queue_size=10)
        # # # Create a subscriber
        # Create ROS publisher for marker array messages
        self.covariance_markers = rospy.Publisher('/covariance_eigen_markers', MarkerArray, queue_size=10)
        rospy.Subscriber("/turtlebot/kobuki/sensors/rplidar", LaserScan, self.check_scan)

        # joint state subscriber
        rospy.Subscriber('/turtlebot/joint_states', JointState, self.joint_state_callback)
        # velocity subscriber
        rospy.Subscriber('/cmd_vel', Twist, self.velocity_callback) 
         # imu subscriber
        rospy.Subscriber('/turtlebot/kobuki/sensors/imu_data', Imu, self.imu_callback)
     
             
      
    def wrap_angle(self, angle):
        """this function wraps the angle between -pi and pi

        :param angle: the angle to be wrapped
        :type angle: float

        :return: the wrapped angle
        :rtype: float
        """
        return (angle + ( 2.0 * np.pi * np.floor( ( np.pi - angle ) / ( 2.0 * np.pi ) ) ) )
    

    def joint_state_callback(self, msg):
        '''' This function reads the joint state of the robot and computes the odometry of the robot
        Args:
            msg (JointState): The joint state message
            '''
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
            
            self.publish_odometry(msg)
    """
    Publishes the odometry message
    """
    def publish_odometry(self ,msg):
        '''This function publishes the odometry message
        Args:
            msg (JointState): The joint state message
            '''
       
        odom = Odometry()
        
        current_time = rospy.Time.from_sec(msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9)
     
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

        self.odom_pub.publish(odom)
     
        self.tf_broadcaster.sendTransform((self.xk[-3], self.xk[-2], 0.0), q , rospy.Time.now(), self.child_frame, self.parent_frame)
   
    def imu_callback(self, msg):
        '''This function reads the IMU data and updates the heading of the robot    
        Args:
            msg (Imu): The IMU message
            '''
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
            with self.lock:
                 self.xk ,self.Pk = self.pse.heading_update(self.xk , self.Pk , self.yaw)

    def transform_cloud(self, target_frame, source_frame, scan):
        ''' This function transforms the source  scan data to the target frame
        Args:
            target_frame (str): The target frame
            source_frame (str): The source frame
            scan (LaserScan): The scan data to be transformed
            Returns:
            xy_points (list): The transformed scan data
            '''

        try:
            #
            # Create a LaserProjection object
            projector = LaserProjection()

            # Convert the LaserScan to PointCloud2
            cloud = projector.projectLaser(scan)
                
            # Wait for the transform to become available
            self.tf_buff.can_transform(target_frame, source_frame, scan.header.stamp, rospy.Duration(1.0))
            
            transform = self.tf_buff.lookup_transform(target_frame, source_frame,scan.header.stamp)
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
        '''This function checks the scan data and updates the map
        Args:
            scan (LaserScan): The scan data
            '''
        scan = self.transform_cloud(self.child_frame , self.rplidar_frame , scan)
        # Convert laser scan data to x, y coordinates in robot frame 
        self.scan_cartesian = np.array(scan)
    
        if(len(self.map) == 0 and len(self.scan_cartesian)):
            # self.publish_covariance_marker()
            self.scan_world = scan_to_world(self.scan_cartesian, self.xk[-3:])
            self.scan.append(self.scan_cartesian)
            self.map.append(self.scan_world)    
            self.xk , self.Pk = self.pse.Add_New_Pose(self.xk, self.Pk)
        elif(len(self.scan_cartesian) and check_scan_thershold(self.xk,self.min_scan_th_distance, self.min_scan_th_angle)):
            # self.publish_covariance_marker()
            self.scan_world = scan_to_world(self.scan_cartesian  , self.xk[-3:])
            self.map.append(self.scan_world)
            self.scan.append(self.scan_cartesian)
            self.xk , self.Pk = self.pse.Add_New_Pose(self.xk, self.Pk)
            # check if the scan is overlapping with the previous scans
            Ho = self.pse.OverlappingScan(self.xk , self.overlapping_check_th_dis , self.max_scans_to_match)            
            zp = np.zeros((0,1)) # predicted scan
            Rp = np.zeros((0,0)) # predicted scan covariance
            Hp = []
            i =0 
            for j in Ho:
                jXk = self.pse.hfj(self.xk, j)
                jPk = self.pse.jPk(self.xk, self.Pk ,j)
                matched_scan = self.scan[j]
                current_scan = self.scan[-1]
                xk  = self.xk[-3:].reshape(3,1)
                # ICP Registration 
                zr  = ICP(matched_scan, current_scan, jXk)
                matched_pose = self.xk[j:j+3,:].reshape((3,1))
                Rr = self.R_icp
                isCompatibile = self.pse.ICNN(jXk, jPk , zr, Rr )
                if(isCompatibile):
                    zp = np.block([[zp],[zr]])
                    Rp = scipy.linalg.block_diag(Rp, Rr)
                    Hp.append(j)
            if(len(Hp)>0):
                zk ,Rk, Hk,Vk = self.pse.ObservationMatrix(self.xk , Hp,zp,Rp )
                self.xk , self.Pk = self.pse.Update(self.xk, self.Pk, Hk, Vk, zk, Rk,Hp)

            # remove the oldest scan
            if(len(self.map) > self.max_scan_history):
                with self.lock: 
                     self.xk , self.Pk = self.pse.remove_pose(self.xk, self.Pk , self.num_of_scans_to_remove) 
               
                indices = [2*i+1 for i in range(self.num_of_scans_to_remove)]
             
                for i in sorted(indices, reverse=True):
                    self.scan.pop(i)
                    self.map.pop(i)
                # for i in range(self.num_of_scans_to_remove ):
                #     print("remove")
                #     print("scan", len(self.scan))
                #     print("map", len(self.map))
                #     self.scan.pop(0)
                #     self.map.pop(0)
                #     print("scan", len(self.scan))
                #     print("map", len(self.map))

        if(len(self.scan) > 0):
            self.publish_viewpoints()   
            self.create_map()      
            
  
    def create_map(self):
        '''This function creates a point cloud message of the map'''
        z = np.zeros((len(self.map), 1))
        full_map = np.zeros((0,3))
        map = build_map(self.scan,self.xk)
        for m in map:
            if(len(m) == 0):
                continue
            z = np.zeros((len(m), 1))
            full_map = np.block([[full_map] ,[m,z]])
        # Create the header for the point cloud message
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'world_ned'  # Set the frame ID
        # # Create the point cloud message
        point_cloud_msg = point_cloud2.create_cloud_xyz32(header, full_map)
        self.full_map_pub.publish(point_cloud_msg)

    def publish_viewpoints(self):
        '''This function publishes the viewpoints as markers'''

        # Create a marker array message
        marker_frontier_lines = MarkerArray()
        marker_frontier_lines.markers = []

        viewpoints_list = []

        for i in range(0,len(self.xk),3):
            myMarker = Marker()
            myMarker.header.frame_id = "world_ned"
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
        return 
        '''This function publishes the covariance eigenvalues and eigenvectors as markers'''
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
            marker_scale = [0.1, 0.1, 0.1]
            if np.any(np.isnan(eigenvalues)) or np.any(np.isinf(eigenvalues)) or np.any(eigenvalues < 0):
                # print("Invalid eigenvalues:", eigenvalues)
                marker_scale  = [0.1, 0.1, 0.00001]
            else:
                marker_scale  = [2.4*np.sqrt(eigenvalues[0]), 2.4*np.sqrt(eigenvalues[1]) , 0.0001 ]

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
            if( not eigenvectors[0, 0]):
                angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
            else:
                angle = np.arctan2(1, 1)
            quat = tf.transformations.quaternion_from_euler(0, 0, angle)
            # quat = tf.transformations.quaternion_from_matrix(matrix)
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
        left_linear_vel   = lin_vel  - (ang_vel*self.wheel_base/2)
        right_linear_vel = lin_vel   +  (ang_vel*self.wheel_base/2)
 
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