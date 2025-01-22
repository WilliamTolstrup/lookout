import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image

import numpy as np
import cv2
from cv_bridge import CvBridge

bridge = CvBridge()

class Calibration(Node):
    def __init__(self):
        super().__init__('calibration')

        self.checkerboard_size = (3, 4)  # 4 squares in x, 5 squares in y
        self.square_size = 0.05  # Size of each square in meters

        self.image_sub = self.create_subscription(Image, '/camera/raw_image', self.image_callback, 10)


    def image_callback(self, img):
        # Convert ROS Image to OpenCV image
        self.image = bridge.imgmsg_to_cv2(img, desired_encoding='bgr8')
        self.get_logger().info("Images received", once=True)

        # Generate 3D world points for the checkerboard
        world_points = []
        for i in range(self.checkerboard_size[1]):  # y-axis
            for j in range(self.checkerboard_size[0]):  # x-axis
                world_points.append([j * self.square_size, i * self.square_size, 0])
        world_points = np.array(world_points, dtype=np.float32)

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Detect checkerboard corners
        found, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)

        if found:
            # Refine corner detection for higher accuracy
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

            # Compute homography matrices for both cameras
            homography1, _ = cv2.findHomography(world_points[:, :2], corners[:, 0, :])

            self.get_logger().info(f"\n \n \n \n=============================\nHomography matrix computed \n=============================\n\nHomography Matrix for Camera:\n{str(homography1)}\n\n \n \n", once=True)
            # Probably violating all kinds of best practices here, but I'm just trying to get the homographies to print out nicely :)
            
            cv2.drawChessboardCorners(self.image, self.checkerboard_size, corners, found)
            cv2.imshow("Detected Corners", self.image)
            cv2.waitKey(1)  # Or cv2.waitKey(1) in ROS 2


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


