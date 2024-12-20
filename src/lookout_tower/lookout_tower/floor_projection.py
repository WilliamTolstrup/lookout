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

class FloorProjection(Node):
    def __init__(self):
        super().__init__('floor_projection')

        package_directory = get_package_share_directory('lookout_tower')

        self.homography_matrix1 = np.array([[7.43359893e+01, -6.05110948e+01,  2.82275964e+02],
                                            [1.86232912e-01,  1.48613961e+00,  2.44246163e+02],
                                            [7.20071467e-04, -1.89563847e-01,  1.00000000e+00]]) # With checkerboard at 0, 0, 0

        self.homography_matrix2 = np.array([[-5.50815917e+01,  3.78277731e+01,  3.45269719e+02],
                                            [-2.45905106e+00,  3.68606000e+00,  1.63346149e+02],
                                            [-1.31961493e-02,  1.18132825e-01,  1.00000000e+00]])

        self.inverse_homography_matrix1 = np.linalg.inv(self.homography_matrix1)
        self.inverse_homography_matrix2 = np.linalg.inv(self.homography_matrix2)

        # HSV thresholds
        self.rug_hsv_lower = np.array([15, 54, 0])
        self.rug_hsv_upper = np.array([20, 124, 255])

        self.floor_hsv_lower = np.array([19, 14, 0])
        self.floor_hsv_upper = np.array([95, 51, 255])

        self.robot_pose = Vector3()

        self.static_occupancy_grid_flag = False
        self.dynamic_occupancy_grid = None

        self.previous_frame_camera1 = None
        self.previous_frame_camera2 = None

        # Define grid resolution and bounds
        self.grid_resolution = 0.15  # Meters per cell
        self.grid_min_x, self.grid_max_x = -6, 6  # 12m x 12m grid
        self.grid_min_y, self.grid_max_y = -6, 6

        # Start subscription and publisher
        self.robot_pose_sub = self.create_subscription(Vector3, '/robot/pose', self.robot_pose_callback, 10)
        self.image1_sub = Subscriber(self, Image, '/camera1/image_raw')
        self.image2_sub = Subscriber(self, Image, '/camera2/image_raw')

        self.static_occupancy_grid_pub = self.create_publisher(OccupancyGrid, '/static_map', 100)
        self.dynamic_occupancy_grid_pub = self.create_publisher(OccupancyGrid, '/dynamic_map', 100)


        # Synchronize image topics
        self.synchronizer = ApproximateTimeSynchronizer([self.image1_sub, self.image2_sub], queue_size=10, slop=0.1)
        self.synchronizer.registerCallback(self.image_callback)

    def robot_pose_callback(self, msg):
        self.robot_pose = msg # Robot pose in world coordinates

    def image_callback(self, img1, img2):
        # Convert ROS Image to OpenCV image
        self.image1 = bridge.imgmsg_to_cv2(img1, desired_encoding='bgr8')
        self.image2 = bridge.imgmsg_to_cv2(img2, desired_encoding='bgr8')
        self.get_logger().info("Images received", once=True)

        if self.static_occupancy_grid_flag == False:

            cleaned_mask1 = self.drivable_space(self.image1)
            cleaned_mask2 = self.drivable_space(self.image2)

            white_pixels1 = np.argwhere(cleaned_mask1 == 255)[:, ::-1]  # Switch (row, col) to (x, y)
            white_pixels2 = np.argwhere(cleaned_mask2 == 255)[:, ::-1]  # Switch (row, col) to (x, y)

            world_coords1 = camera_commons.pixels_to_world(white_pixels1, np.linalg.inv(self.homography_matrix1))
            world_coords2 = camera_commons.pixels_to_world(white_pixels2, np.linalg.inv(self.homography_matrix2))

            all_world_coords = np.vstack([world_coords1, world_coords2])

            # Initialize the occupancy grid map
            grid_width = int((self.grid_max_x - self.grid_min_x) / self.grid_resolution)
            grid_height = int((self.grid_max_y - self.grid_min_y) / self.grid_resolution)
            occupancy_grid = np.zeros((grid_height, grid_width), dtype=np.uint8)

            offset_x, offset_y = 0.7, 0.6 # Offset to center the map

            # Convert world coordinates to grid indices
            for coord in all_world_coords:
                x_idx = int((coord[0] - offset_x - self.grid_min_x) / self.grid_resolution)
                y_idx = int((coord[1] - offset_y - self.grid_min_y) / self.grid_resolution)
                if 0 <= x_idx < grid_width and 0 <= y_idx < grid_height:
                    occupancy_grid[y_idx, x_idx] = 255  # Mark as drivable

            # Mark the robot's position in the grid
            self.mark_robot_in_grid(occupancy_grid, self.robot_pose, self.grid_resolution, self.grid_min_x, self.grid_min_y)

            # Publish the occupancy grid
            self.publish_occupancy_grid(occupancy_grid.flatten().astype(int).tolist(), self.grid_resolution, grid_width, grid_height, grid_id="static_map")
            self.dynamic_occupancy_grid = occupancy_grid
            self.static_occupancy_grid_flag = True

        # Dynamic occupancy grid based on frame differencing
        # Blurred images
        blurred1 = cv2.GaussianBlur(self.image1, (5, 5), 0)
        blurred2 = cv2.GaussianBlur(self.image2, (5, 5), 0)

        # Convert to grayscale
        gray1 = cv2.cvtColor(blurred1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(blurred2, cv2.COLOR_BGR2GRAY)

        # Absolute difference between current and previous frame for each camera
        diff1 = cv2.absdiff(gray1, self.previous_frame_camera1) if self.previous_frame_camera1 is not None else gray1
        diff2 = cv2.absdiff(gray2, self.previous_frame_camera2) if self.previous_frame_camera2 is not None else gray2

        # Threshold the difference images
        _, thresh1 = cv2.threshold(diff1, 30, 255, cv2.THRESH_BINARY)
        _, thresh2 = cv2.threshold(diff2, 30, 255, cv2.THRESH_BINARY)

        # Find contours of moving objects
        contours1, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2, _ = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find center of contours
        centers1 = []
        for contour in contours1:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if 1 < radius < 20:  # Filter based on radius
                centers1.append((x, y))
        
        centers2 = []
        for contour in contours2:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if 1 < radius < 20:
                centers2.append((x, y))


        # Convert contours to world coordinates
        world_coords1 = camera_commons.image_to_world(centers1, self.inverse_homography_matrix1)
        world_coords2 = camera_commons.image_to_world(centers2, self.inverse_homography_matrix2)

        all_world_coords = np.vstack([world_coords1, world_coords2])

        grid_width = int((self.grid_max_x - self.grid_min_x) / self.grid_resolution)
        grid_height = int((self.grid_max_y - self.grid_min_y) / self.grid_resolution)

        # Convert world coordinates to grid indices
        for coord in all_world_coords:
            x_idx = int((coord[0] - self.grid_min_x) / self.grid_resolution)
            y_idx = int((coord[1] - self.grid_min_y) / self.grid_resolution)
            if 0 <= x_idx < grid_width and 0 <= y_idx < grid_height:
                self.dynamic_occupancy_grid[y_idx, x_idx] = 0  # Mark as occupied by dynamic obstacles

        # Publish the dynamic occupancy grid
        self.publish_occupancy_grid(self.dynamic_occupancy_grid.flatten().astype(int).tolist(), self.grid_resolution, grid_width, grid_height, grid_id="dynamic_map")
        self.previous_frame_camera1 = gray1
        self.previous_frame_camera2 = gray2

    def compute_robot_footprint(self, robot_pose):
        # Define robot footprint in world coordinates
        robot_length = 1.5  # meters
        robot_width = 1.5  # meters

        # Half the length and width
        half_length = robot_length / 2
        half_width = robot_width / 2

        # Define the corners of the robot footprint
        corners = np.array([
            [-half_length, -half_width],
            [half_length, -half_width],
            [half_length, half_width],
            [-half_length, half_width]
        ])

        # Compute the rotation matrix
        rotation_matrix = np.array([
            [np.cos(radians(robot_pose.z)), -np.sin(radians(robot_pose.z))],
            [np.sin(radians(robot_pose.z)),  np.cos(radians(robot_pose.z))]
        ])

        # Rotate the corners
        rotated_corners = rotation_matrix @ corners.T

        # Translate the corners to the robot's position
        translated_corners = rotated_corners.T + np.array([robot_pose.x, robot_pose.y])

        return translated_corners
    

    def mark_robot_in_grid(self, occupancy_grid, robot_pose, grid_resolution, xmin, ymin):
        # Compute the robot's footprint in world coordinates
        robot_footprint = self.compute_robot_footprint(robot_pose)

        # Get bounding box of the robot's footprint
        x_coords = ((robot_footprint[:, 0] - xmin) / grid_resolution).astype(int)
        y_coords = ((-robot_footprint[:, 1] - ymin) / grid_resolution).astype(int)

        # Get min/max grid indices
        min_x, max_x = np.clip([x_coords.min(), x_coords.max()], 0, occupancy_grid.shape[1] - 1)
        min_y, max_y = np.clip([y_coords.min(), y_coords.max()], 0, occupancy_grid.shape[0] - 1)

        # Mark the cells within the bounding box as drivable
        occupancy_grid[min_y:max_y + 1, min_x:max_x + 1] = 255


    def drivable_space(self, img):
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        rug_mask = cv2.inRange(hsv_img, self.rug_hsv_lower, self.rug_hsv_upper)
        floor_mask = cv2.inRange(hsv_img, self.floor_hsv_lower, self.floor_hsv_upper)

        # Clean masks
        kernel = np.ones((7, 7), np.uint8)
        rug_mask = cv2.morphologyEx(rug_mask, cv2.MORPH_CLOSE, kernel)
        floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_CLOSE, kernel)

        # Combine masks
        drivable_space = cv2.bitwise_or(rug_mask, floor_mask)
        cleaned_mask = cv2.morphologyEx(drivable_space, cv2.MORPH_OPEN, kernel)
        return cleaned_mask


    def publish_occupancy_grid(self, map_data, grid_resolution, grid_width, grid_height, grid_id):
        occupancy_grid = OccupancyGrid()

        # Populate the header
        occupancy_grid.header = Header()
        occupancy_grid.header.stamp = self.get_clock().now().to_msg()
        occupancy_grid.header.frame_id = "map"

        # Populate the map metadata
        occupancy_grid.info.resolution = grid_resolution
        occupancy_grid.info.width = grid_width
        occupancy_grid.info.height = grid_height

        map_data = [(value - 128 if value > 127 else value) for value in map_data]

        # Populate the map data
        occupancy_grid.data = map_data

        # Publish the map
        if grid_id == "dynamic_map":
            self.dynamic_occupancy_grid_pub.publish(occupancy_grid)
            self.get_logger().info('Published dynamic occupancy grid', once=True)
        elif grid_id == "static_map":
            self.static_occupancy_grid_pub.publish(occupancy_grid)
            self.get_logger().info('Published static occupancy grid', once=True)
        else:
            self.get_logger().warn('Invalid frame_id for occupancy grid', once=True)

def main(args=None):
    rclpy.init(args=args)
    floor_projection_node = FloorProjection()
    rclpy.spin(floor_projection_node)
    floor_projection_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
