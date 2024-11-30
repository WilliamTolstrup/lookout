import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3
from message_filters import ApproximateTimeSynchronizer, Subscriber

import numpy as np
import cv2
from cv_bridge import CvBridge
import math
from collections import deque

bridge = CvBridge()

class DetectRobot(Node):
    def __init__(self):
        super().__init__('detect_robot')

        self.orientation_filter = OrientationFilter(window_size=25)

        # Initialize homography matrices
        self.homography_matrix1 = np.array([[3.35276968e-01,  2.96404276e-02, -1.06713314e+02],
                                            [-2.33900048e-19, -4.91739553e-01,  1.40637512e+02],
                                            [3.63601262e-20,  1.07871720e-01,  1.00000000e+00]])

        self.homography_matrix2 = np.array([[-4.03860312e-01,  3.60729010e-02,  1.29154583e+02],
                                             [2.52468441e-18,  1.00867758e+00, -1.65423123e+02],
                                             [4.25339981e-04,  1.30096820e-01,  1.00000000e+00]])
        
        self.camera1_world_position = np.array([0.0, -2.9, 3.0])
        self.camera2_world_position = np.array([0.0, 6.1, 3.0])

        # Initialize variables
        self.previous_orientation = {1: None, 2: None}

        # Start subscription and publisher
        self.image1_sub = Subscriber(self, Image, '/camera1/image_raw')
        self.image2_sub = Subscriber(self, Image, '/camera2/image_raw')
        self.pub_image1_processed = self.create_publisher(Image, 'camera1/image_processed', 10)
        self.pub_image2_processed = self.create_publisher(Image, 'camera2/image_processed', 10)
        self.pub_pose = self.create_publisher(Vector3, '/robot/pose', 10)

        # Synchronize image topics
        self.synchronizer = ApproximateTimeSynchronizer([self.image1_sub, self.image2_sub], queue_size=10, slop=0.1)
        self.synchronizer.registerCallback(self.callback)

    def callback(self, img1, img2):
        # Convert ROS Image to OpenCV image
        self.image1 = bridge.imgmsg_to_cv2(img1, desired_encoding='bgr8')
        self.image2 = bridge.imgmsg_to_cv2(img2, desired_encoding='bgr8')
        self.get_logger().info("Images received", once=True)

        world_robot_position1, camera_robot_orientation1, weight1, front_midpoint1, rear_midpoint1 = self.loop(self.image1, 1)
        world_robot_position2, camera_robot_orientation2, weight2, front_midpoint2, rear_midpoint2 = self.loop(self.image2, 2)

        # Normalise weights for logging
        weight_total = weight1 + weight2
        weight1 = weight1 / weight_total if weight_total != 0 else 0
        weight2 = weight2 / weight_total if weight_total != 0 else 0
       # self.get_logger().info(f"Weight1: {weight1}, Weight2: {weight2}")
        self.get_logger().info(f"Orientation1: {camera_robot_orientation1}, Orientation2: {camera_robot_orientation2}")


        # Fuse the data
        if weight1 + weight2 != 0:
            fused_position, fused_orientation = self.weighted_data_fusion(world_robot_position1, world_robot_position2, camera_robot_orientation1, camera_robot_orientation2, weight1, weight2)
            self.orientation_filter.add_orientation(fused_orientation)
            fused_orientation = self.orientation_filter.get_filtered_orientation()
        else:
            fused_position, fused_orientation = (0.0, 0.0), 0.0
        self.get_logger().info(f"Fused position: {fused_position}, Fused orientation: {fused_orientation}")
        # Annotate image
        self.image1 = self.annotate_image(self.image1, fused_position, fused_orientation, front_midpoint1, rear_midpoint1)
        self.image2 = self.annotate_image(self.image2, fused_position, fused_orientation, front_midpoint2, rear_midpoint2)

        # Publish the messages
        self.publish_msgs(self.image1, self.image2, (fused_position, fused_orientation))

    def loop(self, img, camera_id):

        if camera_id == 1:
            homography_matrix = self.homography_matrix1
            camera_world_position = self.camera1_world_position
        elif camera_id == 2:
            homography_matrix = self.homography_matrix2
            camera_world_position = self.camera2_world_position
        else:
            self.get_logger().warn("Invalid camera ID. Skipping processing.", once=True)
            return

        if img is not None:
            front_wheels, rear_wheels = self.find_wheels(img)

            front_center = self.detect_wheel_centers(front_wheels)
            rear_center = self.detect_wheel_centers(rear_wheels)

            camera_robot_position, camera_robot_orientation, front_midpoint, rear_midpoint = self.calculate_robot_pose(front_center, rear_center, camera_id)

            if camera_robot_orientation is None:
                self.get_logger().warn(f"Cannot calculate robot orientation from camera {camera_id}.", once=True)
                weight = 0
                camera_robot_orientation = None
            if camera_robot_position is None:
                self.get_logger().warn(f"Cannot calculate robot position from camera {camera_id}.", once=True)
                weight = 0
                world_robot_position = None
            if camera_robot_position and camera_robot_orientation is None:
                self.get_logger().warn(f"Cannot calculate robot position and orientation from camera {camera_id}.", once=True)
                weight = 0
                world_robot_position = None
                camera_robot_orientation = None


            elif camera_robot_position and camera_robot_orientation is not None:
                world_robot_position = self.homography_transform(homography_matrix, camera_robot_position)
                weight = self.calculate_weights(world_robot_position, camera_world_position)

            return world_robot_position, camera_robot_orientation, weight, front_midpoint, rear_midpoint

    def find_wheels(self, img):
        """
        Detect wheels in the image and calculate the robot's pose.
        """
        # Initialize pose variables
        robot_position = None
        robot_orientation = None

        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Red and blue thresholds
        red_lower_1, red_upper_1 = (0, 50, 50), (10, 255, 255)
        red_lower_2, red_upper_2 = (170, 50, 50), (180, 255, 255)
        blue_lower, blue_upper = (100, 50, 50), (130, 255, 255)

        # Create masks
        red_mask_1 = cv2.inRange(hsv, red_lower_1, red_upper_1)
        red_mask_2 = cv2.inRange(hsv, red_lower_2, red_upper_2)
        red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

        return red_mask, blue_mask

    def detect_wheel_centers(self, mask):
        """
        Detect wheel centers from a mask.
        :param mask: Binary mask for the color.
        :param img: Image for debugging.
        :param debug_image: Whether to draw debug circles.
        :param color_name: Name of the color ("red" or "blue").
        :return: List of (x, y) positions for detected wheels.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        centers = []
        for contour in contours:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if 1 < radius < 20:  # Filter based on radius
                centers.append((x, y))
        return centers

    def normalize_angle(self, angle):
        """
        Normalize an angle to the range [-180, 180] degrees.
        :param angle: The angle to normalize.
        :return: The normalized angle.
        """
        angle = angle % 360
        if angle < 0:
            angle += 360
        return angle

    def calculate_robot_pose(self, front_centers, rear_centers, camera_id):
        """
        Calculate the robot's position and orientation based on wheel centerpoints.
        :param front_wheels: List of two (x, y) positions for the front wheels.
        :param rear_wheels: List of two (x, y) positions for the rear wheels.
        :return: Robot position (midpoint of rear wheels) and orientation (in degrees).
        """

        if len(front_centers) == 2:
            # Midpoint of front wheels
            front_midpoint = (
                (front_centers[0][0] + front_centers[1][0]) / 2,
                (front_centers[0][1] + front_centers[1][1]) / 2,
            )
        elif len(front_centers) == 1:
            front_midpoint = front_centers[0]
        else:
            front_midpoint = None


        if len(rear_centers) == 2:
            # Midpoint of rear wheels (robot's position)
            rear_midpoint = (
                (rear_centers[0][0] + rear_centers[1][0]) / 2,
                (rear_centers[0][1] + rear_centers[1][1]) / 2,
            )
        elif len(rear_centers) == 1:
            rear_midpoint = rear_centers[0]
        else:
            rear_midpoint = None

        # Calculate robot position by taking the midpoint of the two midpoints from front and rear wheels (if both are present)
        if front_midpoint is not None and rear_midpoint is not None:
            robot_position = (
                (front_midpoint[0] + rear_midpoint[0]) / 2,
                (front_midpoint[1] + rear_midpoint[1]) / 2,
            )
            if camera_id == 2:
                dx = -(front_midpoint[0] - rear_midpoint[0])
                dy = -(front_midpoint[1] - rear_midpoint[1])
            else:
                dx = front_midpoint[0] - rear_midpoint[0]
                dy = front_midpoint[1] - rear_midpoint[1]
            robot_orientation = math.atan2(dy, dx) * 180 / math.pi  # Angle in degrees
            robot_orientation = self.normalize_angle(robot_orientation)
            robot_orientation = self.validate_orientation(self.previous_orientation[camera_id], robot_orientation)
            self.previous_orientation[camera_id] = robot_orientation

        # Fallbacks if one of the midpoints is missing
        elif front_midpoint is not None:
            robot_position = front_midpoint
            robot_orientation = None

        elif rear_midpoint is not None:
            robot_position = rear_midpoint
            robot_orientation = None

        else:
            robot_position = None
            robot_orientation = None

        # Round the robot position to two decimal places
        if robot_position is not None:
            robot_position = round(robot_position[0], 2), round(robot_position[1], 2)
        
        return robot_position, robot_orientation, front_midpoint, rear_midpoint

    def validate_orientation(self, previous_orientation, current_orientation, max_change=10):
        """
        Validate the current orientation based on the previous orientation.
        :param previous_orientation: The previous orientation.
        :param current_orientation: The current orientation.
        :return: The validated orientation.
        """

        if current_orientation is None:
            return previous_orientation

        if previous_orientation is None:
            # No reference orientation available, accept the new orientation
            return current_orientation


        change = abs(current_orientation - previous_orientation)
        # Normalize change for wrapping cases (e.g 359 to 1 deg)
        change = min(change, 360 - change)

        if change > max_change:
            # Reject the new orientation
            return previous_orientation

        # Else accept the new orientation
        return current_orientation

    def homography_transform(self, homography_matrix, pixel_point):
        """
        Transform a pixel point to world coordinates using a homography matrix.
        :param homography_matrix: The homography matrix.
        :param pixel_point: The pixel point to transform.
        :return: The world coordinates.
        """
        # Convert the pixel point to homogeneous coordinates
        pixel_point_homogeneous = np.append(pixel_point, 1)

        # Transform the pixel point to world coordinates using the homography matrix
        world_point_homogeneous = np.dot(homography_matrix, pixel_point_homogeneous)

        # Convert back to Cartesian coordinates
        world_point = world_point_homogeneous[:2] / world_point_homogeneous[2]

        return world_point

    def calculate_weights(self, world_robot_pos, camera_world_position):
        """
        Calculate weights for two robot positions based on distance to the cameras.
        :param world_robot_pos: The robot's world position
        :param camera_world_position: The camera's world position.
        :return: The weight for the camera.
        """
        robot_z = 0  # Assume robot is on the ground
        # Calculate distances from the robot to each camera
        distance = math.sqrt((world_robot_pos[0] - camera_world_position[0])**2 + (world_robot_pos[1] - camera_world_position[1])**2 + (robot_z - camera_world_position[2])**2)

        weight_camera = 1 / distance if distance != 0 else 0

        return weight_camera


    def weighted_data_fusion(self, world_robot_pos1, world_robot_pos2, orientation1, orientation2, weight1, weight2):
        """
        Fuse two sets of position and orientation data using a weighted average.
        :param world_robot_pos1: The robot's world position from the first source.
        :param world_robot_pos2: The robot's world position from the second source.
        :param orientation1: The robot's orientation from the first source.
        :param orientation2: The robot's orientation from the second source.
        :param weight1: The camera weight for camera1.
        :param weight2: The camera weight for camera2.
        :return: Fused position and orientation.
        """

        # Normalize weights
        total_weight = weight1 + weight2
        norm_weight1 = weight1 / total_weight
        norm_weight2 = weight2 / total_weight

        if weight1 > 0 and weight2 > 0:

            position = (
                norm_weight1 * world_robot_pos1[0] + norm_weight2 * world_robot_pos2[0],
                norm_weight1 * world_robot_pos1[1] + norm_weight2 * world_robot_pos2[1],
            )
            
            orientation1_rad = math.radians(orientation1)
            orientation2_rad = math.radians(orientation2)

            sin_sum = norm_weight1 * math.sin(orientation1_rad) + norm_weight2 * math.sin(orientation2_rad)
            cos_sum = norm_weight1 * math.cos(orientation1_rad) + norm_weight2 * math.cos(orientation2_rad)

            fused_orientation_rad = math.atan2(sin_sum, cos_sum)

            orientation = math.degrees(fused_orientation_rad)
            orientation = self.normalize_angle(orientation)

            #orientation = (norm_weight1 * orientation1 + norm_weight2 * orientation2)/total_weight # Try / total_weight later

        elif weight1 == 0 or orientation1 is None or world_robot_pos1 is None:
            position = world_robot_pos2
            orientation = orientation2

        elif weight2 == 0 or orientation2 is None or world_robot_pos2 is None:
            position = world_robot_pos1
            orientation = orientation1

        return position, orientation

    def annotate_image(self, img, position, orientation, front_midpoint, rear_midpoint):
        """
        Annotate the image with the robot's position and orientation.
        :param img: The image to annotate.
        :param position: The robot's position.
        :param orientation: The robot's orientation.
        :return: The annotated image.
        """
        if position is not None:
            position = (round(position[0], 2), round(position[1], 2))
            cv2.putText(img, f"Position: {position}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if orientation is not None:
            cv2.putText(img, f"Orientation: {orientation:.2f} degrees", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if front_midpoint is not None:
            cv2.circle(img, (int(front_midpoint[0]), int(front_midpoint[1])), 5, (0, 255, 0), -1)
        if rear_midpoint is not None:
            cv2.circle(img, (int(rear_midpoint[0]), int(rear_midpoint[1])), 5, (0, 255, 0), -1)
        
        # Draw line between front and rear wheels
        if front_midpoint is not None and rear_midpoint is not None:
            cv2.line(img, (int(front_midpoint[0]), int(front_midpoint[1])), (int(rear_midpoint[0]), int(rear_midpoint[1])), (0, 255, 0), 2)

        return img


    def publish_msgs(self, img1, img2, pose):
        """
        Publish processed image and robot position + orientation.
        :param img1: image1 to publish.
        :param img2: image2 to publish.
        :param pose: Tuple containing (position, orientation).
        """
        # Ensure pos_ang_msg contains three values
        if pose is None or len(pose) < 2:
            self.get_logger().warn("Invalid pose data. Skipping publish.", once=True)
            return

        # Fill in default values if data is incomplete
      #  robot_position = pos_ang_msg[0] if pos_ang_msg[0] else (0.0, 0.0)
        if pose[0] is None or not isinstance(pose[0], (tuple, list, np.ndarray)) or np.all(pose[0] == 0):
            robot_position = (0.0, 0.0)
        else:
            robot_position = pose[0]
            
        robot_orientation = pose[1] if len(pose) > 1 else 0.0

        # Prepare and publish the position message
        pose_msg = Vector3()
        pose_msg.x = robot_position[0]
        pose_msg.y = robot_position[1]
        pose_msg.z = robot_orientation
        self.pub_pose.publish(pose_msg)

        # Publish processed images
        image1_processed_msg = bridge.cv2_to_imgmsg(img1)
        if image1_processed_msg is not None:
            self.pub_image1_processed.publish(image1_processed_msg)
        image2_processed_msg = bridge.cv2_to_imgmsg(img2)
        if image2_processed_msg is not None:
            self.pub_image2_processed.publish(image2_processed_msg)


class OrientationFilter:
    def __init__(self, window_size=5):
        self.sin_window = deque(maxlen=window_size)
        self.cos_window = deque(maxlen=window_size)

    def add_orientation(self, orientation):
        # Convert to radians
        orientation = math.radians(orientation)

        # Append sin and cos values to the window
        self.sin_window.append(math.sin(orientation))
        self.cos_window.append(math.cos(orientation))
    
    def get_filtered_orientation(self):
        if not self.sin_window or not self.cos_window:
            return None

        # Calculate the average sin and cos values
        avg_sin = sum(self.sin_window) / len(self.sin_window)
        avg_cos = sum(self.cos_window) / len(self.cos_window)

        # Calculate the average orientation
        orientation = math.atan2(avg_sin, avg_cos)

        # Convert to degrees
        orientation = math.degrees(orientation)
        orientation = orientation % 360

        return orientation

def main(args=None):
    rclpy.init(args=args)
    master_camera_node = DetectRobot()
    rclpy.spin(master_camera_node)
    master_camera_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
