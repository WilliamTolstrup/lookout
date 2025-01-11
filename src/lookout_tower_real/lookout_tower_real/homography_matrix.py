import cv2
import numpy as np

# Define the physical positions of the markers
world_points = np.array([
    [-2, 4],
    [3, 4],
    [-2, 0],
    [3, 0]
], dtype=np.float32)

# Define the pixel positions of the markers
# Camera 1
# image_points = np.array([
#     [204, 148],
#     [457, 148],
#     [103, 286],
#     [578, 286]
# ], dtype=np.float32)

# Camera 2
image_points = np.array([
    [551, 307],
    [42, 306],
    [438, 153],
    [177, 154]
], dtype=np.float32)

# Compute the Homography matrix
homography_matrix, status = cv2.findHomography(image_points, world_points)

print("Homography Matrix:")
print(homography_matrix)


def pixel_to_world(homography_matrix, pixel_point):
    # Convert the pixel point to homogeneous coordinates
    pixel_point_homogeneous = np.append(pixel_point, 1)
    
    # Transform the pixel point to world coordinates using the homography matrix
    world_point_homogeneous = np.dot(homography_matrix, pixel_point_homogeneous)
    
    # Convert back to Cartesian coordinates
    world_point = world_point_homogeneous[:2] / world_point_homogeneous[2]
    
    return world_point

# Example usage
pixel_point = np.array([551, 307], dtype=np.float32)
world_point = pixel_to_world(homography_matrix, pixel_point)
print("World Point:")
print(world_point)