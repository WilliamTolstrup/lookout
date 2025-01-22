import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        self.publisher_ = self.create_publisher(Image, '/camera/raw_image', 10)
        self.FPS = 1 / 30
        self.timer = self.create_timer(self.FPS, self.timer_callback)
        self.cap = cv2.VideoCapture(4)
        self.bridge = CvBridge()

        # Define four points in the image
        self.points = [
            (395, 395), # Top left
            (1470, 430), # Top right
            (0, 1080), # Bottom left
            (1920, 1080), # Bottom right
        ]

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            # Draw points and their coordinates on the frame
            # for point in self.points:
            #     cv2.circle(frame, point, 5, (0, 0, 255), -1)  # Draw red point
            #     cv2.putText(frame, f"{point}", (point[0] + 10, point[1] - 10),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

            # Convert to ROS Image message
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            self.publisher_.publish(msg)

            # Visualize image with points (optional)
            # cv2.imshow('image_with_points', frame)
            # cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    camera_publisher = CameraPublisher()
    rclpy.spin(camera_publisher)
    camera_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
