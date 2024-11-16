import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3

import numpy as np
import cv2
from cv_bridge import CvBridge

bridge = CvBridge()

class DetectRobot(Node):
    def __init__(self):
        super().__init__('detect_robot')
        # Start subscription and publisher right away
        self.get_image_raw = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.pub_image_processed = self.create_publisher(Image, 'camera/image_processed', 10)
        self.pub_pos_angle = self.create_publisher(Vector3, '/robot/position_angle', 10)

    def convert_between_ros2_and_opencv(self, img, parameter="ROS2_to_CV2"):
        if parameter == "ROS2_to_CV2":
            return bridge.imgmsg_to_cv2(img, desired_encoding='bgr8')

        if parameter == "CV2_to_ROS2":
            return bridge.cv2_to_imgmsg(img)

        else:
            self.get_logger().warn("Parameter not set! Use either: ROS2_to_CV2 or CV2_to_ROS2")



    def image_callback(self, msg):
        # Convert ROS Image to OpenCV image
        image_raw = self.convert_between_ros2_and_opencv(msg, parameter="ROS2_to_CV2")

        if image_raw is not None:
            self.find_wheels(image_raw, debug_image=True)

    def fixHSVRange(self, h, s, v):
        # Normal H,S,V: (0-360,0-100%,0-100%)
        # OpenCV H,S,V: (0-180,0-255 ,0-255)
        return (180 * h / 360, 255 * s / 100, 255 * v / 100)


    def find_wheels(self, img, debug_image=False):
        # Messages
        position_and_angle_msg = Vector3()
        midpoint = (0,0)
        angle = 0.0

        # Img shape
        H, W = img.shape[:2]

        # Perform HSV color thresholding to find the distinct blue wheels
        blue_lower_limit = (100, 50, 50)
        blue_upper_limit = (130, 255, 255)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv, blue_lower_limit, blue_upper_limit)

        # Find contours
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        centers = []
        for contour in contours:
        #    if cv2.contourArea(contour) > 50: # Filter out noise
            (x, y), radius = cv2.minEnclosingCircle(contour)

            if 5 < radius < 20: # Assuming wheel radius is between 5 and 20 pixels, maybe test this
                centers.append((x, y))

                if debug_image == True:
                    cv2.circle(img, (int(x), int(y)), 5, (0, 255, 255), -1)
                        

        # Ensure we detect 4 wheels
        if len(centers) == 4:
            # sort by y-coordinate, to separate front and rear wheels (hopefully)
            centers.sort(key=lambda c: c[1])
            rear_wheels = centers[:2] # Assuming lower y-coordinate are the rear wheels
            rear_wheel_1 = (int(rear_wheels[0][0]), int(rear_wheels[0][1]))
            rear_wheel_2 = (int(rear_wheels[1][0]), int(rear_wheels[1][1]))
            front_wheels = centers[2:]

            # Find midpoint and orientation based on the rear wheels
            midpoint = ((rear_wheels[0][0] + rear_wheels[1][0]) // 2,
                        (rear_wheels[0][1] + rear_wheels[1][1]) // 2)
            print(midpoint)
            dx = rear_wheels[1][0] - rear_wheels[0][0]
            dy = rear_wheels[1][1] - rear_wheels[0][1]
            angle = np.arctan2(dy, dx) * 180 / np.pi
        
            if debug_image == True:
                for (x, y) in centers:
                    cv2.circle(img, (int(x),int(y)), 5, (0, 255, 255), -1)
                cv2.circle(img, (int(midpoint[0]), int(midpoint[1])), 5, (255, 0, 0), -1)
                cv2.line(img, rear_wheel_1, rear_wheel_2, (0, 0, 255), 2)
                cv2.putText(img, f"Angle: {angle:.2f}", (int(midpoint[0]) + 10, int(midpoint[1]) + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                self.get_logger().info(f"Midpoint: {midpoint}, Angle: {angle:.2f}")

            else:
                self.get_logger().warn("Did not detect 4 wheels!", once=True)

        # Populate msgs
        image_processed_msg = self.convert_between_ros2_and_opencv(img, parameter="CV2_to_ROS2")
        if image_processed_msg is not None:
            self.pub_image_processed.publish(image_processed_msg)

        position_and_angle_msg.x = float(midpoint[0])
        position_and_angle_msg.y = float(midpoint[1])
        position_and_angle_msg.z = angle
        self.pub_pos_angle.publish(position_and_angle_msg)

            
def main(args=None):
    rclpy.init(args=args)
    master_camera_node = DetectRobot()
    rclpy.spin(master_camera_node)
    master_camera_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()