import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header


import numpy as np
import cv2
from cv_bridge import CvBridge
import yaml
import camera_commons
from math import radians
import matplotlib.pyplot as plt

bridge = CvBridge()

class TestingScript(Node):
    def __init__(self):
        super().__init__('floor_projection')

        self.checkerboard_size = (3, 4)  # 4 squares in x, 5 squares in y
        self.square_size = 0.5  # Size of each square in meters

        self.image1_sub = Subscriber(self, Image, '/camera1/image_raw')
        self.image2_sub = Subscriber(self, Image, '/camera2/image_raw')

        # Synchronize image topics
        self.synchronizer = ApproximateTimeSynchronizer([self.image1_sub, self.image2_sub], queue_size=10, slop=0.1)
        self.synchronizer.registerCallback(self.image_callback)

    def image_callback(self, img1, img2):
        # Convert ROS Image to OpenCV image
        self.image1 = bridge.imgmsg_to_cv2(img1, desired_encoding='bgr8')
        self.image2 = bridge.imgmsg_to_cv2(img2, desired_encoding='bgr8')
        self.get_logger().info("Images received", once=True)

        # Generate 3D world points for the checkerboard
        world_points = []
        for i in range(self.checkerboard_size[1]):  # y-axis
            for j in range(self.checkerboard_size[0]):  # x-axis
                world_points.append([j * self.square_size, i * self.square_size, 0])
        world_points = np.array(world_points, dtype=np.float32)

        gray1 = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(self.image2, cv2.COLOR_BGR2GRAY)

        # Detect checkerboard corners
        found1, corners1 = cv2.findChessboardCorners(gray1, self.checkerboard_size, None)
        found2, corners2 = cv2.findChessboardCorners(gray2, self.checkerboard_size, None)

        if found1 and found2:
            # Refine corner detection for higher accuracy
            corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1),
                                        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1),
                                        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

            # Compute homography matrices for both cameras
            homography1, _ = cv2.findHomography(world_points[:, :2], corners1[:, 0, :])
            homography2, _ = cv2.findHomography(world_points[:, :2], corners2[:, 0, :])

            print("Homography Matrix for Camera 1:\n", homography1)
            print("Homography Matrix for Camera 2:\n", homography2)

            # Validate the results by projecting world points into the image
            # for point in world_points:
            #     # Convert world point to homogeneous coordinates
            #     wp_h = np.array([point[0], point[1], 1.0])

            #     # Project into camera 1 image
            #     ip_h1 = homography1 @ wp_h
            #     ip1 = ip_h1[:2] / ip_h1[2]  # Normalize by z

            #     # Project into camera 2 image
            #     ip_h2 = homography2 @ wp_h
            #     ip2 = ip_h2[:2] / ip_h2[2]  # Normalize by z

            #     print(f"World Point {point[:2]} -> Camera 1 Pixel {ip1} -> Camera 2 Pixel {ip2}")

        else:
            print("Checkerboard detection failed. Ensure it is visible in both images.")

def main(args=None):
    rclpy.init(args=args)
    testing_node = TestingScript()
    rclpy.spin(testing_node)
    testing_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
