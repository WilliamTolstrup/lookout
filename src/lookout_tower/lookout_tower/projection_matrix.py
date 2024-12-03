import math
import numpy as np
import cv2

intrinsic_matrix = np.array([[343.50, 0.0, 320.0],
                             [0.0, 343.50, 180.0],
                             [0.0, 0.0, 1.0]])

def rotation_matrix(roll, pitch, yaw):
    """
    Create a rotation matrix from roll, pitch, and yaw angles.
    :param roll: The roll angle.
    :param pitch: The pitch angle.
    :param yaw: The yaw angle.
    :return: The rotation matrix.
    """
    # Roll
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(roll), -math.sin(roll)],
                    [0, math.sin(roll), math.cos(roll)]])

    # Pitch
    R_y = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                    [0, 1, 0],
                    [-math.sin(pitch), 0, math.cos(pitch)]])

    # Yaw
    R_z = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                    [math.sin(yaw), math.cos(yaw), 0],
                    [0, 0, 1]])

    # Combine the rotation matrices
    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


def translation_vector(x, y, z):
    """
    Create a translation vector from x, y, and z coordinates.
    :param x: The x coordinate.
    :param y: The y coordinate.
    :param z: The z coordinate.
    :return: The translation vector.
    """
    t = np.array([x, y, z])

    return t

def projection_matrix(intrinsic_matrix, extrinsic_matrix):
    """
    Create a projection matrix from the intrinsic and extrinsic matrices.
    :param intrinsic_matrix: The intrinsic matrix.
    :param extrinsic_matrix: The extrinsic matrix.
    :return: The projection matrix.
    """
    # Get the rotation matrix
    R = rotation_matrix(extrinsic_matrix[3], extrinsic_matrix[4], extrinsic_matrix[5])

    # Get the translation vector
    t = translation_vector(extrinsic_matrix[0], extrinsic_matrix[1], extrinsic_matrix[2])

    # Combine the rotation matrix and translation vector
    Rt = np.column_stack((R, t))

    # Compute the projection matrix
    P = np.dot(intrinsic_matrix, Rt)

    return P


# Camera poses (extrinsic matrices)
camera1_pose = np.array([0.0, -2.9, 3.0, 0.0, 0.50, 1.57]) # x, y, z, roll, pitch, yaw
camera2_pose = np.array([0.0, 6.6, 3.0, 0.0, 0.50, -1.57]) # x, y, z, roll, pitch, yaw

# Compute the projection matrices
P1 = projection_matrix(intrinsic_matrix, camera1_pose)
P2 = projection_matrix(intrinsic_matrix, camera2_pose)

# World points
world_points = np.array([
    [-2, 4, 0, 1],
    [3, 4, 0, 1],
    [-2, 0, 0, 1],
    [3, 0, 0, 1]
], dtype=np.float32).T
# Third row is Z=0 and fourth row is W=1 which is the homogeneous coordinate

# Project the world points to camera planes
image_points1 = np.dot(P1, world_points)
image_points2 = np.dot(P2, world_points)

# Normalize the image points
image_points1 = image_points1 / image_points1[2]
image_points2 = image_points2 / image_points2[2]

# Extract the 2D points
image_points1_2d = image_points1[:2].T
image_points2_2d = image_points2[:2].T

print("Third row of image_points1: ", image_points1[2])
print("Third row of image_points2: ", image_points2[2])

# Compute the homography matrix
homography_matrix, status = cv2.findHomography(image_points1_2d, image_points2_2d) # camera1 to camera2 homography

# Print results
print("Image Points in Camera1:\n", image_points1_2d)
print("Image Points in Camera2:\n", image_points2_2d)
print("Homography Matrix:\n", homography_matrix)
