#!/usr/bin/python3

import rclpy
from rclpy.node import Node
import math
from std_msgs.msg import Float64MultiArray
import numpy as np
from geometry_msgs.msg import Twist
from example_interfaces.msg import Float32

class Commander(Node):

    def __init__(self):
        super().__init__('commander')
        self.wheel_vel1 = np.array([0,0,0,0], float)
        self.wheel_vel2 = np.array([0,0,0,0], float)
        self.gpa = np.array([0],float)
        self.publisher_robot1 = self.create_publisher(Float64MultiArray, '/robot_1/forward_velocity_controller/commands', 10)
        self.publisher_robot2 = self.create_publisher(Float64MultiArray, '/robot_2/forward_velocity_controller/commands', 10)
        self.publisher_robot1gimbal = self.create_publisher(Float64MultiArray, '/robot_1/forward_position_controller/commands', 10)

        self.timer_period = 0.005
        self.L = 0.125 # distance from the robot center to the wheel
        self.Rw = 0.03 # Radius ot the wheel

        self.t = 0
        
        self.gp = 0.0
        
        self.twist = Twist()

        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        
        self.subscription = self.create_subscription(Twist, "/cmd_vel", self.teleop_callback, 1)
        
        self.subscription2 = self.create_subscription(Float32, "/gimbal_pos", self.gimbalpos_callback, 1)
        
    def gimbalpos_callback(self, msg):
    	self.gp = msg.data
    	
    def teleop_callback(self, msg):
    	self.twist = msg

    def timer_callback(self):

        # robot1
        vel_x1 = 0
        vel_y1 = 0
        vel_w1 = -0.7
        vel_x1 = self.twist.linear.x
        vel_y1 = self.twist.linear.y
        vel_w1 = self.twist.angular.z
        self.wheel_vel1[0] = (vel_x1*math.sin(math.pi/4            ) + vel_y1*math.cos(math.pi/4            ) + self.L*vel_w1)/self.Rw
        self.wheel_vel1[1] = (vel_x1*math.sin(math.pi/4 + math.pi/2) + vel_y1*math.cos(math.pi/4 + math.pi/2) + self.L*vel_w1)/self.Rw
        self.wheel_vel1[2] = (vel_x1*math.sin(math.pi/4 - math.pi)   + vel_y1*math.cos(math.pi/4 - math.pi)   + self.L*vel_w1)/self.Rw
        self.wheel_vel1[3] = (vel_x1*math.sin(math.pi/4 - math.pi/2) + vel_y1*math.cos(math.pi/4 - math.pi/2) + self.L*vel_w1)/self.Rw
        self.gpa[0] = self.gp

        # robot2
        vel_x2 = -math.sin(math.pi*self.t/5)
        vel_y2 = math.cos(math.pi*self.t/5)
        vel_w2 = 0
        self.wheel_vel2[0] = (vel_x2*math.sin(math.pi/4            ) + vel_y2*math.cos(math.pi/4            ) + self.L*vel_w2)/self.Rw
        self.wheel_vel2[1] = (vel_x2*math.sin(math.pi/4 + math.pi/2) + vel_y2*math.cos(math.pi/4 + math.pi/2) + self.L*vel_w2)/self.Rw
        self.wheel_vel2[2] = (vel_x2*math.sin(math.pi/4 - math.pi)   + vel_y2*math.cos(math.pi/4 - math.pi)   + self.L*vel_w2)/self.Rw
        self.wheel_vel2[3] = (vel_x2*math.sin(math.pi/4 - math.pi/2) + vel_y2*math.cos(math.pi/4 - math.pi/2) + self.L*vel_w2)/self.Rw        

        self.t += self.timer_period

        array_forPublish1 = Float64MultiArray(data=self.wheel_vel1)    
        array_forPublish2 = Float64MultiArray(data=self.wheel_vel2)
        array_forPublish1gimbal = Float64MultiArray(data=self.gpa) 
        #rclpy.logging._root_logger.info(f"wheel vel : {self.wheel_vel}")
        self.publisher_robot1.publish(array_forPublish1)     
        self.publisher_robot2.publish(array_forPublish2) 
        self.publisher_robot1gimbal.publish(array_forPublish1gimbal) 

if __name__ == '__main__':
    rclpy.init(args=None)
    commander = Commander()
    rclpy.spin(commander)
    commander.destroy_node()
    rclpy.shutdown()

