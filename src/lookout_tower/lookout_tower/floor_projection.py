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
        self.static_occupancy_grid = None
        self.dynamic_occupancy_grid = None

        self.previous_frame_camera1 = None
        self.previous_frame_camera2 = None

        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()

        # Define grid resolution and bounds
        self.grid_resolution = 0.15  # Meters per cell
        self.grid_min_x, self.grid_max_x = -6, 6  # 12m x 12m grid
        self.grid_min_y, self.grid_max_y = -6, 6

        self.initialize_grids()




        # Start subscription and publisher
        self.robot_pose_sub = self.create_subscription(Vector3, '/robot/pose', self.robot_pose_callback, 10)
        self.image1_sub = Subscriber(self, Image, '/camera1/image_raw')
        self.image2_sub = Subscriber(self, Image, '/camera2/image_raw')

        self.static_occupancy_grid_pub  = self.create_publisher(OccupancyGrid, '/static_map', 100)
        self.dynamic_occupancy_grid_pub = self.create_publisher(OccupancyGrid, '/dynamic_map', 100)

        self.static_ogm_pub  = self.create_publisher(Image, 'ogm/static_map', 10)
        self.dynamic_ogm_pub = self.create_publisher(Image, 'ogm/dynamic_map', 10)

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

        # Update the occupancy grids
        #if self.static_occupancy_grid_flag is None:
        self.update_static_grid([self.image1, self.image2])
        self.publish_occupancy_grid(self.static_occupancy_grid.flatten().tolist(), self.grid_resolution, self.grid_width, self.grid_height, "static_map")
        self.static_occupancy_grid_flag = True

        self.update_dynamic_grid(self.image1, self.image2)
        self.publish_occupancy_grid(self.dynamic_occupancy_grid.flatten().tolist(), self.grid_resolution, self.grid_width, self.grid_height, "dynamic_map")
       # self.publish_occupancy_grid(self.static_occupancy_grid.flatten().tolist(), self.grid_resolution, self.grid_width, self.grid_height, "static_map")

        # Visualize the occupancy grids
        self.visualize_occupancy_grids()


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

    def initialize_grids(self):
        # Calculate grid dimensions
        self.grid_width = int((self.grid_max_x - self.grid_min_x) / self.grid_resolution)
        self.grid_height = int((self.grid_max_y - self.grid_min_y) / self.grid_resolution)

        # Initialize static and dynamic grids
        self.static_occupancy_grid = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)
        self.dynamic_occupancy_grid = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)

    def update_static_grid(self, images):
        # Process each camera image
        for img, homography in zip(images, [self.homography_matrix1, self.homography_matrix2]):
            cleaned_mask = self.drivable_space(img)
            world_coords = camera_commons.pixels_to_world(
                np.argwhere(cleaned_mask == 255)[:, ::-1], np.linalg.inv(homography)
            )
            self.update_grid(self.static_occupancy_grid, world_coords)

    def update_dynamic_grid(self, img1, img2):
        # Apply temporal decay to the dynamic grid
        self.dynamic_occupancy_grid = (self.dynamic_occupancy_grid * 0.4).astype(np.uint8) # Allows the detections to fade out over time

        # Apply background subtraction on both images
        fg_mask1 = self.bg_subtractor.apply(img1)
        fg_mask2 = self.bg_subtractor.apply(img2)

        # Threshold the foreground masks to isolate moving objects
        _, thresh1 = cv2.threshold(fg_mask1, 200, 255, cv2.THRESH_BINARY)
        _, thresh2 = cv2.threshold(fg_mask2, 200, 255, cv2.THRESH_BINARY)

        # Process each camera image using thresholded masks
        for thresh, homography in zip([thresh1, thresh2], [self.homography_matrix1, self.homography_matrix2]):
            world_coords = camera_commons.pixels_to_world(
                np.argwhere(thresh == 255)[:, ::-1], np.linalg.inv(homography)
            )
            self.update_grid(self.dynamic_occupancy_grid, world_coords)

    def update_grid(self, grid, world_coords): ### NOTE: This function doesn't quite work as intended YET.
        exclusion_zone = 0.5  # meters in world coordinates

        # Convert world coordinates to grid indices and update the grid
        for coord in world_coords:
            x_idx = int((coord[0] - self.grid_min_x) / self.grid_resolution)
            y_idx = int((coord[1] - self.grid_min_y) / self.grid_resolution)

            # Check if the coordinate is within the exclusion zone
            if hasattr(self, 'robot_pose'):
                robot_x = self.robot_pose.x
                robot_y = self.robot_pose.y

                # Distance check in world coordinates
                if ((coord[0] - robot_x) ** 2 + (coord[1] - robot_y) ** 2) < exclusion_zone ** 2:
                    continue

            # Ensure the indices are within the grid bounds
            if 0 <= x_idx < self.grid_width and 0 <= y_idx < self.grid_height:
                grid[y_idx, x_idx] = 255  # Mark as occupied
            
       # grid[:] = cv2.morphologyEx(grid, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))


    # def update_grid(self, grid, world_coords):
    #     # Convert world coordinates to grid indices and update the grid
    #     for coord in world_coords:
    #         x_idx = int((coord[0] - self.grid_min_x) / self.grid_resolution)
    #         y_idx = int((coord[1] - self.grid_min_y) / self.grid_resolution)

    #         # Ensure the indices are within the grid bounds
    #         if 0 <= x_idx < self.grid_width and 0 <= y_idx < self.grid_height:
    #             grid[y_idx, x_idx] = 255  # Mark as occupied
            
    #    # grid[:] = cv2.morphologyEx(grid, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    def calculate_frame_difference(self, img1, img2):
        # Blur and convert to grayscale
        blurred1 = cv2.GaussianBlur(img1, (5, 5), 0)
        blurred2 = cv2.GaussianBlur(img2, (5, 5), 0)
        gray1 = cv2.cvtColor(blurred1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(blurred2, cv2.COLOR_BGR2GRAY)

        # Calculate absolute difference
        diff1 = cv2.absdiff(gray1, self.previous_frame_camera1) if self.previous_frame_camera1 is not None else gray1
        diff2 = cv2.absdiff(gray2, self.previous_frame_camera2) if self.previous_frame_camera2 is not None else gray2

        # Store current frames for next calculation
        self.previous_frame_camera1 = gray1
        self.previous_frame_camera2 = gray2

        return diff1, diff2

    def apply_threshold(self, diff1, diff2):
        _, thresh1 = cv2.threshold(diff1, 30, 255, cv2.THRESH_BINARY)
        _, thresh2 = cv2.threshold(diff2, 30, 255, cv2.THRESH_BINARY)
        return thresh1, thresh2

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
        combined_grid = cv2.hconcat([static_with_text, dynamic_with_text])

        ogm_msg = bridge.cv2_to_imgmsg(combined_grid)
        self.static_ogm_pub.publish(ogm_msg)

        # Display the combined image
      #  cv2.imshow("Occupancy Grids", combined_grid)
      #  cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    floor_projection_node = FloorProjection()
    rclpy.spin(floor_projection_node)
    floor_projection_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
