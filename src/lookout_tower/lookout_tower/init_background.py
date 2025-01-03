import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3, Pose, Point
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header

import random
import numpy as np
import cv2
from cv_bridge import CvBridge
import yaml
import camera_commons
from math import radians
import matplotlib.pyplot as plt

bridge = CvBridge()

class InitBackground(Node):
    def __init__(self):
        super().__init__('init_background')

        package_directory = get_package_share_directory('lookout_tower')

        self.image1_sub = Subscriber(self, Image, '/camera1/image_raw')
        self.image2_sub = Subscriber(self, Image, '/camera2/image_raw')
        
        # Synchronize image topics
        self.synchronizer = ApproximateTimeSynchronizer([self.image1_sub, self.image2_sub], queue_size=10, slop=0.1)
        self.synchronizer.registerCallback(self.image_callback)

        self.write_background_flag = False
    

    def image_callback(self, img1, img2):
        # Convert ROS Image to OpenCV image
        self.image1 = bridge.imgmsg_to_cv2(img1, desired_encoding='bgr8')
        self.image2 = bridge.imgmsg_to_cv2(img2, desired_encoding='bgr8')

        if self.write_background_flag == False:
            cv2.imwrite('background1.png', self.image1)
            cv2.imwrite('background2.png', self.image2)
            self.write_background_flag = True
            self.get_logger().info("Background images saved")

        
def main(args=None):
    rclpy.init(args=args)
    init_background_node = InitBackground()
    rclpy.spin(init_background_node)
    init_background_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
