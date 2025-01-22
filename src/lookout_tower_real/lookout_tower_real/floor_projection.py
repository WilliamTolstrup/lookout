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
import os

bridge = CvBridge()

class FloorProjection(Node):
    def __init__(self):
        super().__init__('floor_projection')


        self.homography_matrix = np.array([[ 5.24020805e+02, -4.06412456e+02,  9.43400498e+02],
                                           [ 2.17512805e+01, -1.69587008e+01,  8.76806358e+02],
                                           [-5.76355318e-04, -4.37983444e-01,  1.00000000e+00]]
)




 #       self.homography_matrix = np.array([[ 4.52406580e-02,  3.85609932e-02, -4.16458727e+01],
 #                                               [ 3.84449148e-17, -5.57834326e-02,  6.02461072e+01],
 #                                               [-1.50888263e-03,  2.90202050e-02,  1.00000000e+00]])

        self.inverse_homography_matrix = np.linalg.inv(self.homography_matrix)

        # HSV thresholds
        self.rug_hsv_lower = np.array([15, 54, 0])
        self.rug_hsv_upper = np.array([20, 124, 255])

        self.floor_hsv_lower = np.array([19, 14, 0])
        self.floor_hsv_upper = np.array([95, 51, 255])

        self.robot_pose = Pose() # position(x,y,z) and orientation(x,y,z,w)
        self.weights = Vector3()

        self.static_occupancy_grid_flag = False
        self.static_occupancy_grid = None
        self.dynamic_occupancy_grid = None

        self.previous_frame_camera1 = None
        self.previous_frame_camera2 = None

        self.current_goal = None

        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
        self.persistence_map = None
        self.persistence_map_threshold = 50  # Number of frames to keep a detection in the dynamic grid
        self.world_coords_stationary = None

        # Define grid resolution and bounds
        self.grid_resolution = 0.02  # Meters per cell
        self.grid_min_x, self.grid_max_x = -2.5, 2#-3, 4
        self.grid_min_y, self.grid_max_y = -4, 0.5#-8, 4
        #### Experiment with the grid bounds to get good output
        self.initialize_grids()




        # Start subscription and publisher
        self.robot_pose_sub = self.create_subscription(Pose, '/robot/pose', self.robot_pose_callback, 10)

        self.static_occupancy_grid_pub  = self.create_publisher(OccupancyGrid, '/static_map', 100)
        self.dynamic_occupancy_grid_pub = self.create_publisher(OccupancyGrid, '/dynamic_map', 100)

        self.static_ogm_pub  = self.create_publisher(Image, 'ogm/static_map', 10)
        self.dynamic_ogm_pub = self.create_publisher(Image, 'ogm/dynamic_map', 10)

        self.goal_pub = self.create_publisher(Point, '/goal', 10)

        self.debug_image_pub = self.create_publisher(Image, 'debug_image', 10) # DEBUG

        self.binary_map_sub = self.create_subscription(Image, '/camera/segmentation_output', self.image_callback, 10)

    def robot_pose_callback(self, msg):
        self.robot_pose = msg # Robot pose in world coordinates

    def weights_callback(self, msg):
        self.weights = msg

    def image_callback(self, img1):
        # Convert ROS Image to OpenCV image
        self.image1 = bridge.imgmsg_to_cv2(img1, desired_encoding='bgr8')
        self.image1 = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
        self.get_logger().info("Images received", once=True)

        images = [self.image1]
        homographies = [self.homography_matrix]

        # Update the occupancy grids
        self.update_static_grid(images, homographies)
        self.publish_occupancy_grid(self.static_occupancy_grid.flatten().tolist(), self.grid_resolution, self.grid_width, self.grid_height, "static_map")

    #    self.update_dynamic_grid(images, homographies)
    #    self.publish_occupancy_grid(self.dynamic_occupancy_grid.flatten().tolist(), self.grid_resolution, self.grid_width, self.grid_height, "dynamic_map")

        # Visualize the occupancy grids
        self.visualize_occupancy_grids()

        # Create a random goal for the robot to navigate to, and repeat when the goal is reached
        if self.current_goal is None or self.check_distance_to_goal(self.current_goal.x, self.current_goal.y, self.robot_pose.position.x, self.robot_pose.position.y) < 1:
            goal_x, goal_y = self.get_random_goal(self.static_occupancy_grid)
            self.current_goal = Point(x=float(goal_x), y=float(goal_y), z=0.0)
            self.get_logger().info(f"New goal: ({goal_x}, {goal_y})")
        self.goal_pub.publish(self.current_goal)

########### THIS WORKS ############
    # def image_callback(self, img1, img2):
    #     # Convert ROS Image to OpenCV image
    #     self.image1 = bridge.imgmsg_to_cv2(img1, desired_encoding='bgr8')
    #     self.image2 = bridge.imgmsg_to_cv2(img2, desired_encoding='bgr8')
    #     self.get_logger().info("Images received", once=True)

    #     # Update the occupancy grids
    #     self.update_static_grid([self.image1, self.image2])
    #     self.publish_occupancy_grid(self.static_occupancy_grid.flatten().tolist(), self.grid_resolution, self.grid_width, self.grid_height, "static_map")

    #     self.update_dynamic_grid(self.image1, self.image2)
    #     self.publish_occupancy_grid(self.dynamic_occupancy_grid.flatten().tolist(), self.grid_resolution, self.grid_width, self.grid_height, "dynamic_map")

    #     # Visualize the occupancy grids
    #     self.visualize_occupancy_grids()

    #     # Create a random goal for the robot to navigate to, and repeat when the goal is reached
    #     if self.current_goal is None or self.check_distance_to_goal(self.current_goal.x, self.current_goal.y, self.robot_pose.position.x, self.robot_pose.position.y) < 1:
    #         goal_x, goal_y = self.get_random_goal(self.static_occupancy_grid)
    #         self.current_goal = Point(x=float(goal_x), y=float(goal_y), z=0.0)
    #         self.get_logger().info(f"New goal: ({goal_x}, {goal_y})")
    #     self.goal_pub.publish(self.current_goal)


    def drivable_space(self, img):

        kernel = np.ones((7, 7), np.uint8)
        cleaned_floor_mask = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        cleaned_floor_mask = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        return cleaned_floor_mask



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

    def initialize_grids(self):
        # Calculate grid dimensions
        self.grid_width = int((self.grid_max_x - self.grid_min_x) / self.grid_resolution)
        self.grid_height = int((self.grid_max_y - self.grid_min_y) / self.grid_resolution)

        # Initialize static and dynamic grids
        self.static_occupancy_grid = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)
        self.dynamic_occupancy_grid = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)
        self.persistence_map = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)

    def update_static_grid(self, images, homographies):
        # Process each camera image
        for img, homography in zip(images, homographies):
            cleaned_mask = self.drivable_space(img)
            world_coords = camera_commons.pixels_to_world(
                np.argwhere(cleaned_mask == 255)[:, ::-1], np.linalg.inv(homography)
            )
            self.update_grid(self.static_occupancy_grid, world_coords)

        if self.world_coords_stationary is not None:
            self.update_grid(self.static_occupancy_grid, self.world_coords_stationary+255)

        self.add_robot_to_grid(self.static_occupancy_grid, self.robot_pose)

        if self.current_goal is not None:
            x_idx = int((self.current_goal.x - self.grid_min_x) / self.grid_resolution)
            y_idx = int((self.current_goal.y - self.grid_min_y) / self.grid_resolution)
            self.static_occupancy_grid[int(y_idx), int(x_idx)] = 128  # Mark the goal as occupied

    def update_dynamic_grid(self, images, homographies):
        # Exclude points within a certain radius of the robot
        exclusion_radius = 0.4 # Meters in world coordinates

        # Apply temporal decay to the dynamic grid
        self.dynamic_occupancy_grid = (self.dynamic_occupancy_grid * 0.4).astype(np.uint8)  # Allows the detections to fade out over time

        all_filtered_coords = []
        all_world_coords = []

        for img, homography in zip(images, homographies):
            # Apply background subtraction on both images
            fg_mask = self.bg_subtractor.apply(img)

            # Threshold the foreground masks to isolate moving objects
            _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

            # Get world coordinates of moving objects
            world_coords = camera_commons.pixels_to_world(
                np.argwhere(thresh == 0)[:, ::-1], np.linalg.inv(homography)
            )

            valid_indices = (
                (world_coords[:, 0] >= self.grid_min_x) &
                (world_coords[:, 0] <= self.grid_max_x) &
                (world_coords[:, 1] >= self.grid_min_y) &
                (world_coords[:, 1] <= self.grid_max_y)
            )

            # Filter out points near the robot
            robot_x, robot_y = self.robot_pose.position.x, self.robot_pose.position.y
            distances = np.sqrt((world_coords[:, 0] - robot_x) ** 2 +
                                (world_coords[:, 1] - robot_y) ** 2)
            exclusion_mask = (distances >= exclusion_radius) & valid_indices

            # Apply the exclusion mask
            filtered_coords = world_coords[exclusion_mask]
            all_filtered_coords.append(filtered_coords)
            all_world_coords.append(world_coords)

        # Combine the filtered coordinates from all cameras
        all_filtered_coords = np.vstack(all_filtered_coords)
        all_world_coords = np.vstack(all_world_coords)

        self.world_coords_stationary = self.detect_stationary_obstacles(all_world_coords)

            # Update the grid with the filtered coordinates
        self.update_grid(self.dynamic_occupancy_grid, all_filtered_coords)


######### THIS WORKS #########
    # def update_dynamic_grid(self, img1, img2):
    #     # Exclude points within a certain radius of the robot
    #     exclusion_radius = 0.4 # Meters in world coordinates

    #     # Apply temporal decay to the dynamic grid
    #     self.dynamic_occupancy_grid = (self.dynamic_occupancy_grid * 0.4).astype(np.uint8)  # Allows the detections to fade out over time

    #     # Apply background subtraction on both images
    #     fg_mask1 = self.bg_subtractor.apply(img1)
    #     fg_mask2 = self.bg_subtractor.apply(img2)

    #     # Threshold the foreground masks to isolate moving objects
    #     _, thresh1 = cv2.threshold(fg_mask1, 200, 255, cv2.THRESH_BINARY)
    #     _, thresh2 = cv2.threshold(fg_mask2, 200, 255, cv2.THRESH_BINARY)

    #     # Process each camera image using thresholded masks
    #     for thresh, homography in zip([thresh1, thresh2], [self.homography_matrix1, self.homography_matrix2]):
    #         # Get world coordinates of moving objects
    #         world_coords = camera_commons.pixels_to_world(
    #             np.argwhere(thresh == 0)[:, ::-1], np.linalg.inv(homography)
    #         )

    #         valid_indices = (
    #             (world_coords[:, 0] >= self.grid_min_x) &
    #             (world_coords[:, 0] <= self.grid_max_x) &
    #             (world_coords[:, 1] >= self.grid_min_y) &
    #             (world_coords[:, 1] <= self.grid_max_y)
    #         )

            
    #         # Filter out points near the robot
    #         robot_x, robot_y = self.robot_pose.position.x, self.robot_pose.position.y
    #         distances = np.sqrt((world_coords[:, 0] - robot_x) ** 2 +
    #                             (world_coords[:, 1] - robot_y) ** 2)
    #         exclusion_mask = (distances >= exclusion_radius) & valid_indices

    #         # Apply the exclusion mask
    #         filtered_coords = world_coords[exclusion_mask]

    #     self.world_coords_stationary = self.detect_stationary_obstacles(world_coords)

    #     # Update the grid with the filtered coordinates
    #     self.update_grid(self.dynamic_occupancy_grid, filtered_coords)

    #     self.debug_image(thresh1)


    def update_grid(self, grid, world_coords):
        # Convert world coordinates to grid indices and update the grid
        x_indices, y_indices, _ = self.convert_and_validate_grid_indices(world_coords)

        grid[y_indices, x_indices] = 255  # Mark as occupied
            
    def detect_stationary_obstacles(self, world_coords):
        # Convert world coordinates to grid indices
        x_indices, y_indices, valid_indices = self.convert_and_validate_grid_indices(world_coords)

        # Increment the persistence map at the detected indices
        self.persistence_map[y_indices, x_indices] += 1

        # Find stationary obstacles
        stationary_mask = self.persistence_map[y_indices, x_indices] > self.persistence_map_threshold
        stationary_coords = world_coords[valid_indices][stationary_mask]

        # Reset the persistence map at the detected indices
        self.persistence_map[y_indices[stationary_mask], x_indices[stationary_mask]] = 0

        return stationary_coords


    def convert_and_validate_grid_indices(self, world_coords):
        # Calculate scaling factors
        image_aspect_ratio = self.grid_height / self.grid_width
        grid_aspect_ratio = self.grid_resolution

        # Adjust for the aspect ratio difference
        x_scaling_factor = self.grid_resolution
        y_scaling_factor = self.grid_resolution * image_aspect_ratio

        # Convert world coordinates to grid indices
        x_indices = ((world_coords[:, 0] - self.grid_min_x) / y_scaling_factor).astype(int)
        y_indices = ((world_coords[:, 1] - self.grid_min_y) / x_scaling_factor).astype(int)

        # Ensure the indices are within the grid bounds
        valid_indices = (
            (x_indices >= 0) & (x_indices < self.grid_width) &
            (y_indices >= 0) & (y_indices < self.grid_height)
        )

        return x_indices[valid_indices], y_indices[valid_indices], valid_indices


    def visualize_occupancy_grids(self):
        # Resize the grids for consistent display
        static_grid_resized = cv2.resize(
            self.static_occupancy_grid,
            (self.image1.shape[1], self.image1.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        dynamic_grid_resized = cv2.resize(
            self.dynamic_occupancy_grid,
            (self.image1.shape[1], self.image1.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        # Convert grids to BGR for visualization
        static_grid_color = cv2.cvtColor((static_grid_resized * 1).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        dynamic_grid_color = cv2.cvtColor((dynamic_grid_resized * 1).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        # Add titles to the grids for display
        static_with_text = static_grid_color.copy()
        dynamic_with_text = dynamic_grid_color.copy()
        cv2.putText(static_with_text, "Static Grid", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(dynamic_with_text, "Dynamic Grid", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Concatenate the two grids horizontally
        combined_grid = static_with_text#cv2.hconcat([static_with_text, dynamic_with_text])

        ogm_msg = bridge.cv2_to_imgmsg(combined_grid)
        self.static_ogm_pub.publish(ogm_msg)


    def add_robot_to_grid(self, grid, robot_pose, marker_value=(0, 0, 255)):
        # Convert robot pose (world coordinates) to grid indices
        x_idx = int((robot_pose.position.x - self.grid_min_x) / self.grid_resolution)
        y_idx = int((robot_pose.position.y - self.grid_min_y) / self.grid_resolution)

        # Ensure indices are within grid bounds
        if 0 <= x_idx < self.grid_width and 0 <= y_idx < self.grid_height:
            grid[y_idx, x_idx] = 128 if grid.ndim == 2 else marker_value  # Add robot position marker


    def get_random_goal(self, occupancy_grid):
        # Find a random unoccupied cell in the grid
        while True:
            x_grid = random.randint(0, self.grid_width - 1)
            y_grid = random.randint(0, self.grid_height - 1)
            if occupancy_grid[y_grid, x_grid] == 255:
                # Convert grid indices to world coordinates
                x_world = self.grid_min_x + x_grid * self.grid_resolution
                y_world = self.grid_min_y + y_grid * self.grid_resolution
                break
        return x_world, y_world
    
    def check_distance_to_goal(self, goal_x, goal_y, robot_x, robot_y):
        # Calculate Euclidean distance between robot and goal
        distance = ((goal_x - robot_x) ** 2 + (goal_y - robot_y) ** 2) ** 0.5
        return distance

    def debug_image(self, img):
        img_msg = bridge.cv2_to_imgmsg(img)
        self.debug_image_pub.publish(img_msg)
        

def main(args=None):
    rclpy.init(args=args)
    floor_projection_node = FloorProjection()
    rclpy.spin(floor_projection_node)
    floor_projection_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
