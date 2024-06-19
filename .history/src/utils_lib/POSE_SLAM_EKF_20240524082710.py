#!/usr/bin/python3
import numpy as np
from scipy.linalg import block_diag
from  utils_lib.Pose import Pose3D
from math import cos, sin
import rospy
import scipy
import threading
import math 
class PoseSLAMEKF:
    def __init__(self, x0, P0, Q, compass_Rk , compass_Vk , wheel_base , wheel_radius , overlap_distancce = 2):    
        
        self.xk = x0
        self.Pk= P0
        self.Qk = Q
        self.compass_Rk  = compass_Rk
        self.compass_Vk = compass_Vk
        self.wheel_base = wheel_base 
        self.wheel_radius = wheel_radius
        self.xb_dim = 3
        self.alpha = 0.95
        self.overlap_distancce = overlap_distancce
        self.mutex = threading.Lock()
    def wrap_angle(self, angle):
        """this function wraps the angle between -pi and pi

        :param angle: the angle to be wrapped
        :type angle: float

        :return: the wrapped angle
        :rtype: float
        """
        return (angle + ( 2.0 * np.pi * np.floor( ( np.pi - angle ) / ( 2.0 * np.pi ) ) ) )
    
    def Add_New_Pose(self, xk, Pk):
        '''
        This function takes in the state vector and covariance matrix and adds a new pose to the state vector and covariance matrix.
        Inputs:
        - xk: The state vector
        - Pk: The covariance matrix
        Outputs:
        - xk: The updated state vector
        - Pk: The updated covariance matrix
        '''
        xk_new = np.zeros((len(xk)+3,1))
        xk_new[:-3,:] = xk
        xk_new[-3:,:] = xk[-3:,:]
        Pk_new = np.zeros((len(xk)+3,len(xk)+3))

        last_row = Pk[-3:,:] # 
        last_col = Pk[:,-3:]
        Pk_new[:-3,:-3] = Pk
        Pk_new[-3:,:-3] = last_row # make them full correlation at firist 
        Pk_new[:-3,-3:] = last_col  # make them full correlation at firist
        Pk_new[-3:,-3:] = Pk[-3:,-3:]

       
        
        return xk_new , Pk_new
    
    def Prediction(self, xk_1 , Pk_1 , uk , dt):             
        '''
        This function takes in the previous state vector, the previous covariance matrix, the control input,
          and the time step and returns the predicted state vector and covariance matrix.     
        Inputs:    
        - xk_1: The previous state vector
        - Pk_1: The previous covariance matrix
        - uk: The control input
        - dt: The time step 
        Outputs:     
        - xk: The predicted state vector  
        - Pk: The predicted covariance matrix
        '''
       
        t0 = rospy.Time.now().to_sec()
        self.xk = xk_1
        self.dt = dt
        self.uk = uk

        xk_robot = self.get_robot_pose(xk_1)
        theta = xk_robot[-1]

        Jfx = self.F1k(xk_robot)
        Jfw= self.F2k(xk_robot)
        
        
        # Predict the mean of the robot pose
        xk_robot = Pose3D.oplus(xk_robot, uk)
        xk_robot[-1] = self.wrap_angle(xk_robot[-1])

        # Add the predicted robot pose to the state vector
        xk = np.block([[xk_1[:-3,:]], [xk_robot]])

        A = Pk_1[:-3,:-3]  # Extract the covariance of the scan pose
        B = Pk_1[-3:,-3:]  # Extract the covariance of the robot pose
        C = Pk_1[-3:,:-3]  # Extract the covariance of the side
        
        P =  Jfx@B@Jfx.T + Jfw@self.Qk@Jfw.T
        Pk = np.block([[A,C.T@Jfx.T],[Jfx@C,P]])
     
        return xk ,Pk
    
    def get_robot_pose(self , xk):
        return xk[-3:,-3:]
    def F1k(self,xk_robot):
        
        Jfx = Pose3D.J_1oplus(xk_robot,self.uk)
     
        # F1x = np.block([[np.eye(len(self.xk)-3), np.zeros((len(self.xk)-3, 3))],
        #                 [np.zeros((3, len(self.xk)-3)), Jfx]])
        return Jfx
    
    def F2k(self,xk_robot):
        
        theta = xk_robot[-1,-1]
        print("theta",theta)
       
        Jfw = np.array([[self.dt*cos(theta)/2,    self.dt*cos(theta)/2],
                        [self.dt*sin(theta)/2,    self.dt*sin(theta)/2],
                        [self.dt/self.wheel_base, -self.dt/self.wheel_base]])
        print("Jfw",Jfw)
        return Jfw
    
    
    def OverlappingScan(self, xk ):
        '''
        This function returns the indices of the scans that are overlapping with the current scan.
        Inputs:
        - None
        Outputs:
        - H0: The indices of the scans that are overlapping with the current scan
        '''
        H0= [] 
        match_pose = xk[-6:-3,:][0:2,:]
        # print("self.xk",self.xk)
        # print("match_pose",match_pose)
        
        for i in range(0 , len(xk)-2*self.xb_dim , self.xb_dim ):
            # print("New match", self.xk[i:i+3,:][0:2,:])
            distance  = np.linalg.norm(xk[i:i+3,:][0:2,:] - match_pose)
            if distance < self.overlap_distancce:
                index = int(i/3)
                H0.append(index)
            
        return H0
    def remove_pose(self, xk , Pk , len = 3):
        ''' This function removes the last pose from the state vector and covariance matrix.
        Inputs:
        - xk: The state vector
        - Pk: The covariance matrix'''
        indices = []
        
        for l in range(len):
            index = l+1 #get the last even poses index from state vector 
            indices.extend( range(index*3 , index*3+3))
        xk = np.delete(xk, indices, axis=0)
        Pk = np.delete(Pk, indices, axis=0)
        Pk = np.delete(Pk, indices, axis=1)
        return xk , Pk
        
       

    def h(self, xk ,Hp):
        """
        Computes the expected feature observations given the state vector :math:`x_k`.

        :param xk: state vector
        :return: expected feature observations
        """
        hf = np.zeros((0,1))
        for i in range(len(Hp)):
            hf = np.block([[hf],[self.hfj(xk, Hp[i])]])
        return hf

    def hfj(self, xk, j ):
        '''
        This function takes in the state vector and the index of the matched scan and returns the expected observation of the matched scan in the robot frame.
        Inputs:
        - xk: The state vector
        - j: The index of the matched scan
        Outputs:
        - hfj: The expected observation of the matched scan in the robot frame
        '''
        # h(xk_bar,vk)=(-) Xk) [+] x_J+ vk
        # Get Pose vector from the filter state
        NxBk = self.get_robot_pose(xk) # current pose 
        index = int(j*3)
        NxBj = xk[index:index+3,:].reshape((3,1)) # matched pose
        
        # Matched Scan as referance frame 
        Jxn  = Pose3D.ominus(NxBj)
        hfj  = Pose3D.oplus(Jxn ,NxBk)

      

        return hfj
    
    def Jhf(self,xk,j):
        
        '''
        This function takes in the state vector and the index of the matched scan and returns the jacobian of the observation model with respect to the state vector.
        Inputs:
        - xk: The state vector
        - j: The index of the matched scan
        Outputs:
        - J: The jacobian of the observation model with respect to the state vector
        '''

        NxBk = self.get_robot_pose(xk)
        index = int(j*3)
        J1_oplus, J1_ominus, J2_oplus = self.Jhfjx(xk, j)
        J = np.zeros((self.xb_dim,np.shape(xk)[0]))

        J[:,index:index+3] = J1_oplus@J1_ominus
        print("J", J.shape ,xk.shape , J2_oplus.shape)
        J[:,-3:] = J2_oplus 
        return J

    def Jhfjx(self, xk, j):
        
        '''
        This function takes in the state vector and the index of the matched scan and returns the jacobian of the observation model with respect to the state vector.
        
        Inputs:
        - xk: The state vector
        - j: The index of the matched scan
        Outputs:
        - J1_oplus:
        - J1_ominus:
        - J2_oplus:
                
        '''
        # Get Pose vector from the filter state
        NxBk  = self.get_robot_pose(xk) # current pose 
       
        index = int(j*3)
        NxBj  = xk[index:index+3,:].reshape((3,1)) # matched pose
        
        JxN   = Pose3D.ominus(NxBj)  
        # hfj  = Pose3D.oplus(JxN,NxBk)
       
        J1_oplus  = Pose3D.J_1oplus(JxN, NxBk)
        J1_ominus = Pose3D.J_ominus(NxBj)
        J2_oplus  = Pose3D.J_2oplus(JxN)

        
        return J1_oplus, J1_ominus, J2_oplus

    
    def jPk(self, xk , Pk, j):
        '''
        This function takes in the state vector and the index of the matched scan and returns the covariance matrix of the matched scan in the robot frame.
        
        Inputs:
        - xk: The state vector
        - j: The index of the matched scan
        Outputs:
        - jPk: The jacobian of the measurement model
        '''
        index = int(j*3)
        NPk = Pk[-3:,-3:]  # Extract the covariance of the robot pose
        NPj = Pk[index:index+3,index:index+3]  # Extract the covariance of the matched scan pose
        J1_oplus, J1_ominus, J2_oplus = self.Jhfjx(xk, j)
        jPk = J1_oplus@J1_ominus@NPj@J1_ominus.T@J1_oplus.T + J2_oplus@NPk@J2_oplus.T
        
        return jPk
    
    def ObservationMatrix(self,xk,Hp ,zk ,Rk):
        """
        Computes the observation matrix for the EKF SLAM algorithm. The observation matrix is computed using the Jacobian of the observation model with respect to the state vector and the feature observation noise.

        :param xk: state vector
        :param Hp: vector of feature indices
        :param zk: vector of feature observations
        :param Rk: Covariance matrix of the feature observations
        :return: The observation matrix Hk, the observation noise covariance Vk
        """
        Hk, Vk = np.zeros((0,np.shape(xk)[0])), np.zeros((0,0))
        xk_robot = self.get_robot_pose(xk)
        Vr =  np.diag(np.ones(self.xb_dim))
       
        for i,j in enumerate(Hp):

            # Add jacobian with respect to the state vector
            Hk = np.block([[Hk], [self.Jhf(xk, j)]])
            # Add jacbian with respect to the feature observation noise
            Vk = scipy.linalg.block_diag(Vk,Vr)

        
        return zk ,Rk , Hk, Vk 
  
    def SquaredMahalanobisDistance(self, hfj, Pfj, zfi, Rfi):
        """
        Computes the squared Mahalanobis distance between the expected feature observation :math:`hf_j` and the feature observation :math:`z_{f_i}`.

        :param hfj: expected feature observation
        :param Pfj: expected feature observation covariance
        :param zfi: feature observation
        :param Rfi: feature observation covariance
        :return: Squared Mahalanobis distance between the expected feature observation :math:`hf_j` and the feature observation :math:`z_{f_i}`
        """

        # TODO: To be completed by the student
        # Compute inovation
        v_ij = zfi - hfj
        # Compute inovetion unceranity
        S_ij = Rfi + Pfj
   
        # Compute squared mahalandobis distance
        D2_ij = v_ij.T @ np.linalg.inv(S_ij) @ v_ij
        return D2_ij

    def IndividualCompatibility(self, D2_ij, dof, alpha):
        """
        Computes the individual compatibility test for the squared Mahalanobis distance :math:`D^2_{ij}`. The test is performed using the Chi-Square distribution with :math:`dof` degrees of freedom and a significance level :math:`\\alpha`.

        :param D2_ij: squared Mahalanobis distance
        :param dof: number of degrees of freedom
        :param alpha: confidence level
        :return: bolean value indicating if the Mahalanobis distance is smaller than the threshold defined by the confidence level
        """
        # TODO: To be completed by the student
        # print("D2_ij", D2_ij)
        isCompatible = D2_ij < scipy.stats.chi2.ppf(alpha, dof)

        return isCompatible

    def ICNN(self, hf, Phf, zf, Rf):
        """
        Individual Compatibility Nearest Neighbor (ICNN) data association algorithm. Given a set of expected feature
        observations :math:`h_f` and a set of feature observations :math:`z_f`, the algorithm returns a pairing hypothesis
        :math:`H` that associates each feature observation :math:`z_{f_i}` with the expected feature observation
        :math:`h_{f_j}` that minimizes the Mahalanobis distance :math:`D^2_{ij}`.

        :param hf: vector of expected feature observations
        :param Phf: Covariance matrix of the expected feature observations
        :param zf: vector of feature observations
        :param Rf: Covariance matrix of the feature observations
        :param dim: feature dimensionality
        :return: The vector of asociation hypothesis H
        """

        # TODO: To be completed by the student

        D2_ij = self.SquaredMahalanobisDistance(hf, Phf, zf, Rf)
        if self.IndividualCompatibility(D2_ij, self.xb_dim, self.alpha):
            return True
    
    def Update(self, xk, Pk, Hk, Vk, zk, Rk  , Hp):
        """
        Updates the state vector and covariance matrix using the EKF update step.

        :param xk: state vector
        :param Pk: Covariance matrix of the state vector
        :param Hk: observation matrix
        :param Vk: observation noise covariance
        :param zk: observation vector
        :param Rk: observation covariance
        :return: Updated state vector and covariance matrix
        """

        Kk = Pk @ Hk.T @ np.linalg.inv(Hk @ Pk @ Hk.T + Vk@Rk@Vk.T)
        xk = xk + Kk @ (zk - self.h(xk, Hp))
        Pk = (np.eye(len(xk)) - Kk @ Hk) @ Pk@(np.eye(len(xk)) - Kk @ Hk).T 
      
        return xk, Pk
    
    def heading_update(self , xk , Pk , yaw):
        '''This function updates the heading of the robot using the IMU data'''
        # Create a row vector of zeros of size 1 x 3*num_poses
        # print("imu update")   
        Hk = np.zeros((1, len(xk)))
        # Replace the last element of the row vector with 1
        Hk[0, -1] = 1
        predicted_compass_meas = xk[-1]
        # Compute the kalman gain
        K = Pk @ Hk.T @ np.linalg.inv((Hk @ Pk @ Hk.T) + (self.compass_Vk @ self.compass_Rk @ self.compass_Vk.T))
        # Compute the innovation
        innovation = np.array(self.wrap_angle(yaw[0] - predicted_compass_meas)).reshape(1, 1)
        # Update the state vector
        xk = xk + K@innovation
        # Create the identity matrix        
        I = np.eye(len(xk))
        # Update the covariance matrix
        Pk = (I - K @ Hk) @ Pk @ (I - K @ Hk).T

        return xk, Pk
    