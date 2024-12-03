import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from message_filters import ApproximateTimeSynchronizer, Subscriber

import numpy as np
import cv2
from cv_bridge import CvBridge

bridge = CvBridge()

class FloorProjection(Node):
    def __init__(self):
        super().__init__('floor_projection')

        # Initialize known camera points
        self.camera_points = np.array([[0, 0], [0, 360], [640, 0], [640, 360]], dtype=np.float32) # Top-left, bottom-left, top-right, bottom-right

        # Initialize known world points in camera coordinates for both cameras
        self.world_points = np.array([[130, 100], [0, 335], [500, 100], [630, 335]], dtype=np.float32) # Camera 1


        self.homography_matrix1 = np.array([[3.35276968e-01,  2.96404276e-02, -1.06713314e+02],
                                            [-2.33900048e-19, -4.91739553e-01,  1.40637512e+02],
                                            [3.63601262e-20,  1.07871720e-01,  1.00000000e+00]])


        self.homography_matrix2 = np.array([[-7.33161066e-01,  5.77134411e-02,  2.35274687e+02],
                                            [7.49145477e-03,  1.95526970e+00, -3.02437521e+02],
                                            [2.36485776e-03,  2.38392444e-01,  1.00000000e+00]])

        # HSV thresholds
        self.rug_hsv_lower = np.array([15, 54, 0])
        self.rug_hsv_upper = np.array([20, 124, 255])

        self.floor_hsv_lower = np.array([19, 14, 0])
        self.floor_hsv_upper = np.array([95, 51, 255])

        self.transformation_matrix = None
        self.transformation_matrix_flag = False

        # Start subscription and publisher
        self.image1_sub = Subscriber(self, Image, '/camera1/image_raw')
        self.image2_sub = Subscriber(self, Image, '/camera2/image_raw')

        # Synchronize image topics
        self.synchronizer = ApproximateTimeSynchronizer([self.image1_sub, self.image2_sub], queue_size=10, slop=0.1)
        self.synchronizer.registerCallback(self.callback)

    def callback(self, img1, img2):
        # Convert ROS Image to OpenCV image
        self.image1 = bridge.imgmsg_to_cv2(img1, desired_encoding='bgr8')
        self.image2 = bridge.imgmsg_to_cv2(img2, desired_encoding='bgr8')
        self.get_logger().info("Images received", once=True)

        if self.transformation_matrix_flag == False:
            self.transformation_matrix = cv2.getPerspectiveTransform(self.world_points, self.camera_points)
            self.transformation_matrix_flag = True

        cleaned_mask1 = self.drivable_space(self.image1)
        cleaned_mask2 = self.drivable_space(self.image2)

        overlay1 = cv2.bitwise_and(self.image1, self.image1, mask=cleaned_mask1)
        overlay2 = cv2.bitwise_and(self.image2, self.image2, mask=cleaned_mask2)

        # Bird's eye view
        transformed_frame1 = cv2.warpPerspective(self.image1, self.transformation_matrix, (640, 360))
        transformed_frame2 = cv2.warpPerspective(self.image2, self.transformation_matrix, (640, 360))

        

        # Display images
        cv2.imshow('Camera 1', self.image1)
        cv2.imshow('Birds Eye 1', transformed_frame1)
        cv2.imshow('Overlay 1', overlay1)
        cv2.imshow('Camera 2', self.image2)
        cv2.imshow('Birds Eye 2', transformed_frame2)
        cv2.imshow('Overlay 2', overlay2)
        cv2.waitKey(1)

    def drivable_space(self, img):
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        rug_mask = cv2.inRange(hsv_img, self.rug_hsv_lower, self.rug_hsv_upper)
        floor_mask = cv2.inRange(hsv_img, self.floor_hsv_lower, self.floor_hsv_upper)

        # Kernel
        kernel = np.ones((5, 5), np.uint8)
        rug_mask = cv2.morphologyEx(rug_mask, cv2.MORPH_CLOSE, kernel)
        floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_CLOSE, kernel)

        drivable_space = cv2.bitwise_or(rug_mask, floor_mask)
        cleaned_mask = cv2.morphologyEx(drivable_space, cv2.MORPH_CLOSE, kernel)
        return cleaned_mask

def main(args=None):
    rclpy.init(args=args)
    floor_projection_node = FloorProjection()
    rclpy.spin(floor_projection_node)
    floor_projection_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
