import cv2
import numpy as np
from rclpy.node import Node
from nav_msgs.msg import Path, OccupancyGrid
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
import camera_commons

class VisualizePath(Node):
    def __init__(self):
        super().__init__('path_visual')
        self.path = []  # Waypoints in world coordinates
        self.current_target_index = 0
        self.bridge = CvBridge()

        # Homography matrices for the cameras
        self.homography_matrix1 = np.array([[7.43359893e+01, -6.05110948e+01,  2.82275964e+02],
                                            [1.86232912e-01,  1.48613961e+00,  2.44246163e+02],
                                            [7.20071467e-04, -1.89563847e-01,  1.00000000e+00]])
        self.homography_matrix2 = np.array([[-5.50815917e+01,  3.78277731e+01,  3.45269719e+02],
                                            [-2.45905106e+00,  3.68606000e+00,  1.63346149e+02],
                                            [-1.31961493e-02,  1.18132825e-01,  1.00000000e+00]])
        
        self.static_occupancy_grid_sub = self.create_subscription(OccupancyGrid, '/static_map', self.update_static_occupancy_grid, 10)
        self.path_sub = self.create_subscription(Path, '/path', self.path_callback, 10)
        self.image1_sub = Subscriber(self, Image, '/camera1/image_raw')
        self.image2_sub = Subscriber(self, Image, '/camera2/image_raw')

        self.image1_pub = self.create_publisher(Image, '/camera1/image_with_path', 10)
        self.image2_pub = self.create_publisher(Image, '/camera2/image_with_path', 10)

        self.synchronizer = ApproximateTimeSynchronizer([self.image1_sub, self.image2_sub], queue_size=10, slop=0.1)
        self.synchronizer.registerCallback(self.image_callback)

        self.static_occupancy_grid = None

        # Define grid resolution and bounds
        self.resolution = 0.15  # Meters per cell
        self.grid_min_x, self.grid_max_x = -6, 6  # 12m x 12m grid
        self.grid_min_y, self.grid_max_y = -6, 6

    def update_static_occupancy_grid(self, msg):
        # Convert the incoming OccupancyGrid to a NumPy array for easier manipulation
        grid = np.array(msg.data).reshape(msg.info.height, msg.info.width)

        # Save the grid for later visualization
        self.static_occupancy_grid = grid

    def path_callback(self, msg):
        # Store path as a list of (x, y) coordinates
        self.path = [(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses]
        self.current_target_index = 0
        self.get_logger().info(f"Received path with {len(self.path)} waypoints.")

    def image_callback(self, img1, img2):
        # Convert ROS Image to OpenCV image
        image1 = self.bridge.imgmsg_to_cv2(img1, desired_encoding='bgr8')
        image2 = self.bridge.imgmsg_to_cv2(img2, desired_encoding='bgr8')

        # Draw waypoints on the images using the respective homography matrices
        image1_with_path = self.draw_path_on_image(image1, self.homography_matrix1, 'Camera 1')
        image2_with_path = self.draw_path_on_image(image2, self.homography_matrix2, 'Camera 2')

        # Display the images
        cv2.imshow('Camera 1 with Path', image1_with_path)
        cv2.imshow('Camera 2 with Path', image2_with_path)
        cv2.waitKey(1)

    def draw_path_on_image(self, image, homography_matrix, label):
        """
        Draws the path waypoints onto the provided image using the homography matrix.
        """
        if not self.path:
            return image

        for i, (x, y) in enumerate(self.path):
            # Transform the world coordinates to image pixel coordinates
            img_point = camera_commons.world_to_image((x, y), homography_matrix)
            self.get_logger().info(f"Waypoint {i}: World: ({x}, {y}), Image: {img_point}")

            # Draw the waypoint as a circle
            color = (0, 255, 0) if i == self.current_target_index else (255, 0, 0)
            cv2.circle(image, (int(img_point[0]), int(img_point[1])), 5, color, -1)

        # Add a label to the image
        cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return image


def main(args=None):
    import rclpy
    rclpy.init(args=args)
    visualizer = VisualizePath()
    rclpy.spin(visualizer)
    visualizer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
