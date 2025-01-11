import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from message_filters import ApproximateTimeSynchronizer, Subscriber

import numpy as np
import cv2
from cv_bridge import CvBridge
import sys

bridge = CvBridge()

class Calibration(Node):
    def __init__(self):
        super().__init__('calibration')
        self.get_logger().set_level(rclpy.logging.LoggingSeverity.INFO)
        sys.tracebacklimit = 0

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

            self.get_logger().info(f"\n \n \n \n=============================\nHomography matrices computed \n=============================\n\nHomography Matrix for Camera 1:\n{str(homography1)}\n\nHomography Matrix for Camera 2:\n{str(homography2)}\n\n \n \n", once=True)
            # Probably violating all kinds of best practices here, but I'm just trying to get the homographies to print out nicely :)
            
        else:
            self.get_logger().info("Checkerboard not found in one or both images", once=True)

def main(args=None):
    rclpy.init(args=args)
    calibration_node = Calibration()
    rclpy.spin(calibration_node)
    calibration_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
