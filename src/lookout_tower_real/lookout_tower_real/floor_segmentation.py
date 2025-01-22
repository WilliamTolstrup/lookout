import rclpy
from rclpy.node import Node
import torch
import torchvision
import torchvision.transforms as T
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import numpy as np
import cv2

class SegmentationNode(Node):
    def __init__(self):
        super().__init__('segmentation_node')

        # ROS 2 image subscription and publication
        self.image_sub = self.create_subscription(Image, '/camera/raw_image', self.image_callback, 10)
        self.image_pub = self.create_publisher(Image, '/camera/segmentation_output', 10)

        # Timer for periodic processing
        self.timer_period = 10.0  # Adjust the interval as needed (in seconds)
        self.timer = self.create_timer(self.timer_period, self.process_image_timer_callback)

        # Variable to store the latest image message
        self.current_image = None

    #    self.truth_image = cv2.imread('/home/william/Datasets/floorspace/floor1_master.png', cv2.IMREAD_GRAYSCALE)
    #    self.truth_image = cv2.imread('/home/william/repos/lookout/src/lookout_tower_sim/images/drivable_space_camera1.png', cv2.IMREAD_GRAYSCALE)

        # Load the trained segmentation model
        self.model_path = "/home/william/repos/lookout/src/lookout_tower_real/segmentation_model.pth"
        self.model = self.load_model()

        # Transformations for input images
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.bridge = CvBridge()
        self.get_logger().info('Segmentation node initialized!')

    def load_model(self):
        # Load the DeepLabV3 model
        model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
        num_classes = 5
        model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
        model.aux_classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)

        # Load state_dict, ignoring mismatched keys
        state_dict = torch.load(self.model_path, map_location=torch.device('cpu'))

        # Filter out aux_classifier if mismatched
        filtered_state_dict = {
            k: v for k, v in state_dict.items() if "aux_classifier" not in k
        }

        model.load_state_dict(filtered_state_dict, strict=False)
        model.eval()  # Set model to evaluation mode
        self.get_logger().info('Model loaded successfully with adjusted aux_classifier!')
        return model

    def image_callback(self, msg):
        self.current_image = msg

    def process_image_timer_callback(self):
        if self.current_image is None:
            self.get_logger().info('No image received yet!')
            return

        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(self.current_image, desired_encoding='bgr8')

            print("Shape pre segmentation: ", cv_image.shape)

            # Preprocess the image for the model
            input_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            input_tensor = self.transforms(input_image).unsqueeze(0)

            # Perform inference
            with torch.no_grad():
                output = self.model(input_tensor)['out'][0]  # Get model output
                prediction = torch.argmax(output, dim=0).byte().cpu().numpy()

            # Remap classes: Keep green and yellow only, set others to background
            remapped_prediction = np.zeros_like(prediction)
            remapped_prediction[prediction == 2] = 1  # Green → Class 1
            remapped_prediction[prediction == 3] = 2  # Yellow → Class 2

            # Apply morphological opening to the segmentation output
            kernel = np.ones((5, 5), np.uint8)  # Adjust kernel size as needed
            morphed_segmentation = cv2.erode(remapped_prediction, kernel, iterations=3)
            morphed_segmentation = cv2.morphologyEx(morphed_segmentation, cv2.MORPH_CLOSE, kernel)

            # Apply blob analysis: Keep the 2 largest blobs for each class
            filtered_segmentation = self.keep_largest_blobs(segmentation=morphed_segmentation, num_blobs=4, distance_threshold=50)

            # Apply morphological closing to the filtered segmentation
            filtered_segmentation = cv2.morphologyEx(filtered_segmentation, cv2.MORPH_OPEN, kernel, iterations=3)

            # Create a binary map with predictions as white (255) and background as black (0)
            binary_map = (filtered_segmentation > 0).astype(np.uint8) * 255

            # Convert binary map to 3-channel image for visualization
            binary_map_colored = cv2.cvtColor(binary_map, cv2.COLOR_GRAY2BGR)
    #        truth_image = cv2.cvtColor(self.truth_image, cv2.COLOR_BGR2RGB)
            # Convert back to ROS Image message and publish
            segmentation_msg = self.bridge.cv2_to_imgmsg(binary_map_colored, encoding='bgr8')
    #        segmentation_msg = self.bridge.cv2_to_imgmsg(truth_image, encoding='bgr8')
            self.image_pub.publish(segmentation_msg)
            self.get_logger().info('Segmentation result with blob analysis published!')
            print("Shape post segmentation: ", binary_map_colored.shape)

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")



    def keep_largest_blobs(self, segmentation, num_blobs=2, distance_threshold=50):
        """
        Keeps the `num_blobs` largest connected components for each class in the segmentation map.
        Additionally, ensures the blobs are either directly connected or within a specified distance.
        """
        binary_map = np.zeros_like(segmentation, dtype=np.uint8)

        for class_id in np.unique(segmentation):
            if class_id == 0:  # Skip background
                continue

            # Create a binary mask for the current class
            class_mask = (segmentation == class_id).astype(np.uint8)

            # Perform connected components analysis
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(class_mask, connectivity=8)

            # Sort blobs by area (column 4 of stats is the area)
            sorted_indices = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1] + 1  # Exclude background (index 0)
            largest_blobs = sorted_indices[:num_blobs]  # Get indices of the largest blobs

            # Calculate the centroid of the largest blob
            largest_blob_idx = largest_blobs[0]
            largest_blob_centroid = centroids[largest_blob_idx]

            # Keep blobs that are either connected or within the distance threshold
            for blob_idx in largest_blobs:
                blob_centroid = centroids[blob_idx]

                # Calculate the Euclidean distance between centroids
                distance = np.linalg.norm(largest_blob_centroid - blob_centroid)

                # Check if the blob is within the distance threshold
                if distance <= distance_threshold or blob_idx == largest_blob_idx:
                    binary_map[labels == blob_idx] = class_id

        return binary_map



    def visualize_segmentation(self, prediction):
        # Define colors for each class (floor, rug, background)
        colors = {
            0: (0, 0, 0),       # Background (black)
            1: (0, 255, 0),     # Floor (green)
            2: (0, 0, 255),     # Rug (red)
#            3: (255, 255, 0),   # Table (yellow)
#            4: (255, 0, 255)    # Chair (magenta)
        }

        # Create a color map
        height, width = prediction.shape
        color_segmentation = np.zeros((height, width, 3), dtype=np.uint8)

        for class_id, color in colors.items():
            color_segmentation[prediction == class_id] = color

        return color_segmentation


def main(args=None):
    rclpy.init(args=args)
    node = SegmentationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
