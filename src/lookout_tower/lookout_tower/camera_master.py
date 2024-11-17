import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3

import numpy as np
import cv2
from cv_bridge import CvBridge
import math

bridge = CvBridge()

class DetectRobot(Node):
    def __init__(self):
        super().__init__('detect_robot')
        # Init tracked wheels
        self.tracked_wheels = {}  # Format: {id: (x, y)}
        self.next_id = 0

        # Start subscription and publisher
        self.get_image_raw = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.pub_image_processed = self.create_publisher(Image, 'camera/image_processed', 10)
        self.pub_pos_angle = self.create_publisher(Vector3, '/robot/position_angle', 10)

    def convert_between_ros2_and_opencv(self, img, parameter="ROS2_to_CV2"):
        if parameter == "ROS2_to_CV2":
            return bridge.imgmsg_to_cv2(img, desired_encoding='bgr8')
        elif parameter == "CV2_to_ROS2":
            return bridge.cv2_to_imgmsg(img)
        else:
            self.get_logger().warn("Parameter not set! Use either: ROS2_to_CV2 or CV2_to_ROS2")

    def image_callback(self, msg):
        # Convert ROS Image to OpenCV image
        image_raw = self.convert_between_ros2_and_opencv(msg, parameter="ROS2_to_CV2")

        if image_raw is not None:
            image_processed, pos_angle = self.find_wheels(image_raw, debug_image=True)
            if pos_angle and pos_angle[0] is not None:
                self.publish_msgs(image_processed, pos_angle)
            else:
                self.get_logger().warn("Pose could not be calculated. Not publishing.")


    def find_wheels(self, img, debug_image=False):
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

        # Detect wheel centers
        red_centers = self.detect_wheel_centers(red_mask, img, debug_image, "red")
        blue_centers = self.detect_wheel_centers(blue_mask, img, debug_image, "blue")

        # Calculate pose
        robot_position, robot_orientation = self.calculate_robot_pose(red_centers, blue_centers)

        # Debugging output on image
        if robot_position is not None:
            cv2.putText(img, f"Position: {robot_position}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(img, f"Orientation: {robot_orientation:.2f} degrees", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return img, (robot_position, robot_orientation)




    def detect_wheel_centers(self, mask, img, debug_image, color_name):
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
                if debug_image:
                    color = (0, 0, 255) if color_name == "red" else (255, 0, 0)
                    cv2.circle(img, (int(x), int(y)), 5, color, -1)
        return centers


    def calculate_robot_pose(self, front_centers, rear_centers):
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
            self.get_logger().warn("Front wheels not detected. Skipping pose calculation.")


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
            self.get_logger().warn("Rear wheels not detected. Skipping pose calculation.")

        if front_midpoint is None or rear_midpoint is None:
            return None, None
        
        else:
            robot_position = round(rear_midpoint[0], 2), round(rear_midpoint[1], 2)

            # Calculate orientation as the angle between the front and rear midpoints
            dx = front_midpoint[0] - rear_midpoint[0]
            dy = front_midpoint[1] - rear_midpoint[1]
            robot_orientation = math.atan2(dy, dx) * 180 / math.pi  # Angle in degrees

            self.get_logger().info(f"Robot Position: {robot_position}, Orientation: {robot_orientation:.2f}")
            return robot_position, robot_orientation



    def publish_msgs(self, img_msg, pos_ang_msg):
        """
        Publish processed image and robot position + orientation.
        :param img_msg: The processed image to publish.
        :param pos_ang_msg: Tuple containing (position, orientation).
        """
        # Ensure pos_ang_msg contains three values
        if pos_ang_msg is None or len(pos_ang_msg) < 2:
            self.get_logger().warn("Invalid pose data. Skipping publish.")
            return

        # Fill in default values if data is incomplete
        robot_position = pos_ang_msg[0] if pos_ang_msg[0] else (0.0, 0.0)
        robot_orientation = pos_ang_msg[1] if len(pos_ang_msg) > 1 else 0.0

        # Prepare and publish the position message
        position_angle_msg = Vector3()
        position_angle_msg.x = robot_position[0]
        position_angle_msg.y = robot_position[1]
        position_angle_msg.z = robot_orientation
        self.pub_pos_angle.publish(position_angle_msg)

        # Publish processed image
        image_processed_msg = self.convert_between_ros2_and_opencv(img_msg, parameter="CV2_to_ROS2")
        if image_processed_msg is not None:
            self.pub_image_processed.publish(image_processed_msg)



def main(args=None):
    rclpy.init(args=args)
    master_camera_node = DetectRobot()
    rclpy.spin(master_camera_node)
    master_camera_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
