#!/usr/bin/python3
import rospy
import numpy as np
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray

class Velocity_Converter:
    def __init__(self):
        # Create a subscriber to the cmd_vel topic
        self.wheel_radius = 0.033
        self.wheel_base = 0.233
        self.sub = rospy.Subscriber('/cmd_vel', Twist, self.callback)
        # Create a publisher to the left wheel
       # velocity publisher
        # self.vel_pub = rospy.Publisher('/kobuki/kobuki/commands/wheel_velocities', Float64MultiArray , queue_size=10)
        self.vel_pub = rospy.Publisher('/turtlebot/kobuki/commands/wheel_velocities', Float64MultiArray , queue_size=10)
        # self.vel_pub = rospy.Publisher('/mobile_base/commands/wheel_velocities', Float64MultiArray , queue_size=10)
    def callback(self, msg):
        lin_vel = msg.linear.x
        ang_vel = msg.angular.z
        # print("linear and angular ", lin_vel , ang_vel )
        left_linear_vel   = lin_vel  - (ang_vel*self.wheel_base/2)
        right_linear_vel  = lin_vel  +  (ang_vel*self.wheel_base/2)
 
        left_wheel_velocity  = left_linear_vel / self.wheel_radius
        right_wheel_velocity = right_linear_vel / self.wheel_radius
        wheel_vel = Float64MultiArray()
        wheel_vel.data = [left_wheel_velocity, right_wheel_velocity]
        self.vel_pub.publish(wheel_vel)


# Main function
if __name__ == '__main__':
    # Initialize the node
    rospy.init_node('velocity_converter')

    # Create an instance of the DifferentialDrive class
    diff_drive = Velocity_Converter()
    # Spin
    rospy.spin()



