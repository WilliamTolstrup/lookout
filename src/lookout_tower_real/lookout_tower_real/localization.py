import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose

import numpy as np
import cv2
from cv_bridge import CvBridge
import math
from scipy.spatial.transform import Rotation as R
import camera_commons
from filterpy.kalman import KalmanFilter

bridge = CvBridge()

class DetectRobot(Node):
    def __init__(self):
        super().__init__('localization')

        # Initialize homography matrix
        self.homography_matrix = np.array([[ 5.24020805e+02, -4.06412456e+02,  9.43400498e+02],
                                           [ 2.17512805e+01, -1.69587008e+01,  8.76806358e+02],
                                           [-5.76355318e-04, -4.37983444e-01,  1.00000000e+00]])
        
        self.camera_matrix = np.array([[1071.517664190868, 0, 972.2054550900091],
                          [0, 1071.770866282639, 506.1378258844148],
                          [0, 0, 1]])

        self.dist_coeffs = np.array([0.04869923290209156, -0.2997938546736623,
                        -0.0005899377699054353, -0.002503515897917072, 0.6394053014838494])

        # Aruco dictionary and parameters
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters_create()

        # Initialize variables
        self.previous_orientation = None
        self.previous_position = None

        # Kalman filter
        self.kalman_filter = KalmanFilter(dim_x=6, dim_z=3)
        self.kalman_filter.x = np.zeros((6, 1))  # [x, y, theta, v] state vector
        self.kalman_filter.P = np.eye(6) * 1000  # Large initial uncertainty
        self.kalman_filter.F = np.array([
            [1, 0, 0, 1, 0, 0],  # x = x + vx
            [0, 1, 0, 0, 1, 0],  # y = y + vy
            [0, 0, 1, 0, 0, 1],  # theta = theta + omega
            [0, 0, 0, 1, 0, 0],  # vx = vx
            [0, 0, 0, 0, 1, 0],  # vy = vy
            [0, 0, 0, 0, 0, 1],  # omega = omega
        ])  # State transition matrix
        self.kalman_filter.H = np.array([
            [1, 0, 0, 0, 0, 0],  # Map x
            [0, 1, 0, 0, 0, 0],  # Map y
            [0, 0, 1, 0, 0, 0],  # Map theta
        ])  # Measurement function
        self.kalman_filter.R = np.eye(3) * 0.1  # Measurement noise
        self.kalman_filter.Q = np.eye(6) * 0.1  # Process noise



        # Start subscription and publisher
        self.image_sub = self.create_subscription(Image, '/camera/raw_image', self.image_callback, 10)
        self.pub_pose = self.create_publisher(Pose, '/robot/pose', 10)
        self.pub_localization_processed = self.create_publisher(Image, '/camera/image_processed', 10)
        self.pub_localization_debug = self.create_publisher(Image, '/camera/localization_debug', 10)

    def image_callback(self, img):
        # Convert ROS Image to OpenCV image
        self.image = bridge.imgmsg_to_cv2(img, desired_encoding='bgr8')
        self.get_logger().info("Images received", once=True)

        # Detect and localize the robot
        position, orientation = self.detect_and_localize(self.image)

        # Validate the orientation
        orientation = self.validate_orientation(self.previous_orientation, orientation)

        # Validate the position
        position = self.validate_position(self.previous_position, position)

        # Store the orientation and position for the next iteration
        self.previous_orientation = orientation
        self.previous_position = position

        # Publish the robot pose
        if position is not None and orientation is not None:
            self.publish_pose(position, orientation)

            # Draw the robot pose on the image
#            img_debug = self.draw_robot_pose(self.image, (position[0], position[1]), orientation)
            img = self.annotate_image(self.image, (position[0], position[1]), orientation)
#            self.debug_image(img_debug)

            # Publish the processed image
            img_msg = bridge.cv2_to_imgmsg(img)
            self.pub_localization_processed.publish(img_msg)


    def detect_and_localize(self, img):
        # Aruco detection as primary, wheel detection as backup
        pose = self.detect_aruco(img)

        if pose is None:
            # Use backup localization
            self.get_logger().info("No Aruco markers detected, using backup localization")
            pose = self.detect_wheels(img)

        # If no position or orientation is detected from either method, rely on kalman filter prediction
        if pose is None:
            self.get_logger().info("No localization detected, using Kalman filter prediction")
            self.kalman_filter.predict()
            pose = np.array([self.kalman_filter.x[0], self.kalman_filter.x[2]]).reshape(2, 1)
        
        # Check if pose is still None after prediction and initialization
        if pose is None:
            self.get_logger().error("Pose is None, unable to update Kalman filter")
            return None, None  # Early return if pose is still invalid

        if pose is not None:
            # Update kalman filter with the new measurements
            position, orientation = pose[0], pose[1]

            # Faltten position to extract x and y
            if isinstance(position, (tuple, np.ndarray)) and len(position) == 2:
                x, y = position[0], position[1]
            else:
                raise ValueError(f"Unexpected position format: {position}")
            
            # Handle orientation as yaw angle (Because the kalman filter doesn't accept quaternions)
            if isinstance(orientation, (list, tuple, np.ndarray)) and len(orientation) == 4:
                orientation = self.quaternion_to_yaw(orientation)
            elif isinstance(orientation, (float, int)):
                pass
            else:
                raise ValueError(f"Unexpected orientation format: {orientation}")
            
            pose = np.array([x, y, orientation], dtype=float).reshape(3, 1)
            self.kalman_filter.update(pose)

        return (pose[0][0], pose[1][0]), pose[2][0] # Position, orientation

    def detect_aruco(self, img):
        # Detect Aruco markers
        corners, ids, _ = cv2.aruco.detectMarkers(img, self.aruco_dict, parameters=self.aruco_params,
                                                  cameraMatrix=self.camera_matrix, distCoeff=self.dist_coeffs)
        
        if ids is not None:
            for i, corner in enumerate(corners):
                # Get the 2D image coordinates of the marker corners
                corners_2d = corner[0] # Shape: (4, 2)

                center_2d = np.mean(corners_2d, axis=0)

                # Map the center and corners to the world plane
                center_world = camera_commons.point_to_world(center_2d, self.homography_matrix)
                corners_world = [camera_commons.point_to_world(corners_2d, self.homography_matrix) for corners_2d in corners_2d]

                # Compute orientation using the first two corners
                dx = corners_world[1][0] - corners_world[0][0]
                dy = corners_world[1][1] - corners_world[0][1]
                angle = math.atan2(dy, dx) # Angle in radians

                quaternion = self.angle_to_quaternion(angle)

                return center_world, quaternion
        
        else:
            return None

    def quaternion_multiply(self, quaternion1, quaternion0):
        w0, x0, y0, z0 = quaternion0
        w1, x1, y1, z1 = quaternion1
        return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                        x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                        -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                        x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)

    def normalize_angle_degrees(self, angle):
        """
        Normalize an angle to the range [-180, 180] degrees.
        :param angle: The angle to normalize.
        :return: The normalized angle.
        """
        angle = angle % 360
        if angle < 0:
            angle += 360
        return angle
    
    def normalize_angle_rad(self, angle):
        """
        Normalize an angle to the range [-π, π] radians.
        :param angle: The angle in radians.
        :return: The normalized angle.
        """
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def angle_to_quaternion(self, angle):
        # Quaternion representing a rotation about the z-axis
        q_w = math.cos(angle / 2)  # Scalar part
        q_z = math.sin(angle / 2)  # Z-axis rotation part
        return np.array([q_w, 0, 0, q_z])

    def normalize_quaternion(self, q):
        """
        Normalize a quaternion.
        :param q: The quaternion to normalize.
        :return: The normalized quaternion.
        """
        norm = np.linalg.norm(q)
        if norm == 0:
            return q
        return q / norm

    def quaternion_to_yaw(self, q):
        """
        Calculate the yaw angle from a quaternion.
        :param q: The quaternion.
        :return: The yaw angle in degrees.
        """
        x, y, z, w = q
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y ** 2 + z ** 2))

        # Convert to degrees
        return math.degrees(yaw)

    def validate_orientation(self, previous_orientation, current_orientation, max_change=math.radians(10)):
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
        
        previous_yaw = previous_orientation#self.quaternion_to_yaw(previous_orientation)
        current_yaw = current_orientation#self.quaternion_to_yaw(current_orientation)


        change = abs(current_yaw - previous_yaw)
        change = min(change, 2 * math.pi - change) 

        if change > max_change:
            # Reject the new orientation
            return previous_orientation

        # Else accept the new orientation
        return current_orientation

    def validate_position(self, previous_position, current_position, max_change=0.1):
        """
        Validate the current position based on the previous position.
        :param previous_position: The previous position.
        :param current_position: The current position.
        :return: The validated position.
        """

        if current_position is None:
            return previous_position

        if previous_position is None:
            # No reference position available, accept the new position
            return current_position

        # convert to numpy arrays
        previous_position = np.array(previous_position)
        current_position = np.array(current_position)

        change = np.linalg.norm(current_position - previous_position)

        if change > max_change:
            # Reject the new position
            return previous_position

        # Else accept the new position
        return current_position

    def draw_aruco_markers(self, img, corners, ids):
        img = cv2.aruco.drawDetectedMarkers(img, corners, ids)
            # Iterate through each detected marker to calculate and draw orientation
        if ids is not None:
            for i, corner in enumerate(corners):
                # Get the top-left, top-right, bottom-right, and bottom-left corners
                top_left = corner[0][0]
                top_right = corner[0][1]

                # Calculate the center of the marker
                center_x = int((top_left[0] + top_right[0]) / 2)
                center_y = int((top_left[1] + top_right[1]) / 2)
                center = (center_x, center_y)

                # Calculate the forward direction (approximate orientation)
                forward_x = int(center_x + (top_right[0] - top_left[0]) * 0.5)
                forward_y = int(center_y + (top_right[1] - top_left[1]) * 0.5)
                forward_point = (forward_x, forward_y)

                # Draw the arrow to indicate orientation
                cv2.arrowedLine(img, center, forward_point, (0, 255, 0), 2, tipLength=0.3)

        return img
    
    def draw_robot_pose(self, img, position, orientation):
        position = (int(position[0]), int(position[1]))
        cv2.circle(img, position, 5, (0, 255, 0), -1)

        if isinstance(orientation, (list, tuple, np.ndarray)) and len(orientation) == 4:
            orientation = self.quaternion_to_yaw(orientation)

        # Draw the orientation of the robot
        dx = 50 * math.cos(orientation)
        dy = 50 * math.sin(orientation)
        end_point = (position[0] + int(dx), position[1] + int(dy))
        cv2.arrowedLine(img, position, end_point, (0, 0, 255), 2)

        return img
    
    def publish_pose(self, position, orientation):
        pose = Pose()

        # Convert orientation to quaternion
        orientation = self.angle_to_quaternion(orientation)

        pose.position.x = position[0]
        pose.position.y = position[1]
        pose.position.z = 0.0
        pose.orientation.x = orientation[0]
        pose.orientation.y = orientation[1]
        pose.orientation.z = orientation[2]
        pose.orientation.w = orientation[3]
        self.pub_pose.publish(pose)

    def debug_image(self, img):
        img_msg = bridge.cv2_to_imgmsg(img)
        self.pub_localization_debug.publish(img_msg)

###############################################################################
##                        Backup pose estimation code                        ##
###############################################################################

    def find_wheels(self, img):
        """
        Detect wheels in the image
        """
        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Blue wheel thresholds
        blue_lower_1, blue_upper_1 = (100, 120, 36), (118, 233, 48)
        blue_lower_2, blue_upper_2 = (100, 144, 57), (115, 255, 255)
        blue_lower_3, blue_upper_3 = (106, 175, 24), (179, 255, 255)

        # Masks
        blue_mask_1 = cv2.inRange(hsv, blue_lower_1, blue_upper_1)
        blue_mask_2 = cv2.inRange(hsv, blue_lower_2, blue_upper_2)
        blue_mask_3 = cv2.inRange(hsv, blue_lower_3, blue_upper_3)
        blue_mask = cv2.bitwise_or(blue_mask_1, blue_mask_2)
        blue_mask = cv2.bitwise_or(blue_mask, blue_mask_3)

        # Erode and dilate masks to remove noise
        kernel = np.ones((5, 5), np.uint8)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)
        blue_mask = cv2.erode(blue_mask, kernel, iterations=1)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)

        self.debug_image(blue_mask)

        return blue_mask

    def detect_wheel_centers(self, mask):
        """
        Detect wheel centers from a mask.
        :param mask: Binary mask for the color.
        :return: List of (x, y) positions for detected wheels.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        centers = []
        for contour in contours:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            if 1 < radius < 200:  # Filter based on radius
                centers.append((x, y))
        

        return centers

    def estimate_robot_position(self, centers):
        # Calculate the average of all wheel centers, changing depending on how many centers are detected; represents the robot position
        if len(centers) == 4:
            # Calculate the center of the robot
            x = (centers[0][0] + centers[1][0] + centers[2][0] + centers[3][0]) / 4
            y = (centers[0][1] + centers[1][1] + centers[2][1] + centers[3][1]) / 4
            return x, y
        elif len(centers) == 3:
            # Calculate the center of the robot
            x = (centers[0][0] + centers[1][0] + centers[2][0]) / 3
            y = (centers[0][1] + centers[1][1] + centers[2][1]) / 3
            return x, y
        elif len(centers) == 2:
            # Calculate the center of the robot
            x = (centers[0][0] + centers[1][0]) / 2
            y = (centers[0][1] + centers[1][1]) / 2
            return x, y
        elif len(centers) == 1:
            return centers[0]
        else:
            self.get_logger().info("No wheels detected")
            return self.previous_position

    def estimate_robot_orientation(self, position):
        # Estimate orientation based on displacement between frames, to estimate orientation based on direction of movement
        if position is None or self.previous_position is None:
            return self.previous_orientation

        # Displacement vector
        displacement = np.array(position) - np.array(self.previous_position)
        magnitude = np.linalg.norm(displacement)

        if magnitude < 0.1: # If movement is negligible, keep the previous orientation
            return self.previous_orientation
        
        # Calculate the angle of the displacement vector
        orientation = math.atan2(displacement[1], displacement[0])

        return orientation

    def detect_wheels(self, img):
        # Detect the wheels
        mask = self.find_wheels(img)
        centers = self.detect_wheel_centers(mask)

        # Estimate the robot position
        position = self.estimate_robot_position(centers)

        # Estimate the robot orientation
        orientation = self.estimate_robot_orientation(position)

        # Update the previous position and orientation
        self.previous_position = position
        self.previous_orientation = orientation

        return position, orientation

    def annotate_image(self, img, position, orientation):
        # Draw the position of the robot
        position = (round(int(position[0]), 2), round(int(position[1]), 2))
        cv2.circle(img, position, 5, (0, 255, 0), -1)

        if isinstance(orientation, (list, tuple, np.ndarray)):
            orientation = self.quaternion_to_yaw(orientation)

        # Draw the orientation of the robot
        dx = 50 * math.cos(orientation)
        dy = 50 * math.sin(orientation)
        end_point = (position[0] + int(dx), position[1] + int(dy))
        cv2.arrowedLine(img, position, end_point, (0, 0, 255), 2)

        return img

def main(args=None):
    rclpy.init(args=args)
    master_camera_node = DetectRobot()
    rclpy.spin(master_camera_node)
    master_camera_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
