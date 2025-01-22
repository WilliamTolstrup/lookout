import cv2
import numpy as np

# Image points in pixels
image_points = np.array([
    [0, 1080],       # Bottom-left
    [1920, 1080],    # Bottom-right
    [395, 395],      # Top-left
    [1470, 430]      # Top-right
], dtype=np.float32)

# Real-world points in meters
real_world_points = np.array([
    [0, 0],          # Bottom-left
    [2.95, 0],       # Bottom-right
    [-0.72, 3.22],   # Top-left
    [3.68, 3.22]     # Top-right
], dtype=np.float32)

# Calculate the homography matrix
H, status = cv2.findHomography(image_points, real_world_points)

print("Homography matrix:\n", H)

# Example image point
image_point1 = np.array([0, 1080, 1])  # Homogeneous coordinates
image_point2 = np.array([1920, 1080, 1])  # Homogeneous coordinates
image_point3 = np.array([395, 395, 1])  # Homogeneous coordinates
image_point4 = np.array([1470, 430, 1])  # Homogeneous coordinates
combined_image_points = np.array([image_point1, image_point2, image_point3, image_point4])

# Map to real-world coordinates
for points in (combined_image_points):
   real_world_point = np.dot(H, points)
   real_world_point /= real_world_point[2]  # Normalize by dividing by the third element

   print("Real-world coordinates:", real_world_point[:2])
