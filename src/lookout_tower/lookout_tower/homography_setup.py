import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

# Script finds the pixel coordinates for the yellow markers in the gazebo world, used in another script to calculate the homography matrix

class ImageSubscriber(Node):

    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/camera2/image_raw',
            self.listener_callback,
            10)
        self.publisher_ = self.create_publisher(Image, '/camera/image_processed', 10)
        self.bridge = CvBridge()

    def listener_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # Convert BGR image to HSV
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Define range for yellow color in HSV
        lower_yellow = np.array([20, 100, 100])
        upper_yellow = np.array([30, 255, 255])

        # Threshold the HSV image to get only yellow colors
        mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Iterate through contours and find the center of each yellow object
        for contour in contours:
            M = cv2.moments(contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                self.get_logger().info(f'Yellow object center: ({cx}, {cy})')
                # Draw the center on the image
                cv2.circle(cv_image, (cx, cy), 5, (0, 255, 0), -1)
                cv2.putText(cv_image, f'({cx}, {cy})', (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert the processed image back to ROS Image message
        processed_image_msg = self.bridge.cv2_to_imgmsg(cv_image, 'bgr8')
        self.publisher_.publish(processed_image_msg)

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()