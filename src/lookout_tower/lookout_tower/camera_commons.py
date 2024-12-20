import numpy as np
from math import sin, cos, radians
import cv2


def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    Converts Euler angles (roll, pitch, yaw) to a rotation matrix.
    """
    Rx = np.array([
        [1, 0, 0],
        [0, cos(roll), -sin(roll)],
        [0, sin(roll), cos(roll)]
    ])
    Ry = np.array([
        [cos(pitch), 0, sin(pitch)],
        [0, 1, 0],
        [-sin(pitch), 0, cos(pitch)]
    ])
    Rz = np.array([
        [cos(yaw), -sin(yaw), 0],
        [sin(yaw), cos(yaw), 0],
        [0, 0, 1]
    ])
    return Rz @ Ry @ Rx


def compute_homography(intrinsic_matrix, rotation_matrix, translation_vector, z_plane=0):
    """
    Computes the homography matrix for the planar world (z = z_plane).
    """
    # Compute the extrinsic matrix [R|T]
    extrinsic_matrix = np.hstack((rotation_matrix, translation_vector))

    # Simplify for the z=0 plane: H = K * [R|T]
    homography_matrix = intrinsic_matrix @ extrinsic_matrix

    # Normalize to ensure the matrix remains homogeneous (3x3)
    homography_matrix = homography_matrix[:, :3]

    # Apply 180 degree rotation
    rotation_180 = np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ])
    homography_matrix = rotation_180 @ homography_matrix

    return homography_matrix

def validate_homographies(homography_matrix, world_point=(0, 0, 1)):
    # Test point in world coordinates
    x, y, z = world_point
    world_point_array = np.array([x, y, z])  # Assume planar world, z = 0

    # Project into camera 1's and camera 2's frames
    camera_point = homography_matrix @ world_point_array

    # Normalize homogeneous coordinates
    camera_point /= camera_point[-1]

    return camera_point


def image_to_world(image_points, homography_matrix):
    """
    Vectorized transformation of image points to world coordinates for a planar world.
    
    :param image_points: Array of shape (N, 2) with pixel coordinates.
    :param homography_matrix: 3x3 homography matrix.
    :return: Array of shape (N, 2) with world coordinates.
    """
    # Convert to numpy array if input is a list
    image_points = np.array(image_points)

    # Handle empty input
    if image_points.shape[0] == 0:
        return np.empty((0, 2))

    # Add homogeneous coordinate (z=1) for all image points
    image_points_homogeneous = np.hstack([image_points, np.ones((image_points.shape[0], 1))])

    # Apply homography
    world_points_homogeneous = image_points_homogeneous @ np.linalg.inv(homography_matrix).T

    # Normalize to get (x, y)
    world_points = world_points_homogeneous[:, :2] / world_points_homogeneous[:, 2:3]
    return world_points

def pixels_to_world(white_pixels, homography):
    """
    Maps white pixels (drivable space) from image coordinates to world coordinates.
    Args:
        white_pixels: Nx2 array of (x, y) pixel coordinates
        homography: 3x3 homography matrix for the camera
    Returns:
        Nx2 array of world coordinates
    """
    # Convert to homogeneous coordinates
    pixel_homogeneous = np.hstack([white_pixels, np.ones((white_pixels.shape[0], 1))])  # Nx3
    world_homogeneous = homography @ pixel_homogeneous.T  # 3xN
    world_homogeneous /= world_homogeneous[2, :]  # Normalize
    return world_homogeneous[:2, :].T  # Nx2 (x, y world coordinates)


def point_to_world(pixel_point, homography_matrix):
    """
    Transform a pixel point to world coordinates using a homography matrix.
    :param pixel_point: The pixel point to transform.
    :param homography_matrix: The homography matrix.
    :return: The world coordinates.
    """
    # Convert the pixel point to homogeneous coordinates
    pixel_point_homogeneous = np.append(pixel_point, 1)

    # Transform the pixel point to world coordinates using the homography matrix
    world_point_homogeneous = np.dot(homography_matrix, pixel_point_homogeneous)

    # Convert back to Cartesian coordinates
    world_point = world_point_homogeneous[:2] / world_point_homogeneous[2]

    return world_point

def world_to_image(world_point, homography_matrix):
    """
    Transform a world point to image pixel coordinates using a homography matrix.
    :param world_point: The world point to transform.
    :param homography_matrix: The homography matrix.
    :return: The pixel coordinates.
    """
    # Convert the world point to homogeneous coordinates
    world_point_homogeneous = np.append(world_point, 1)

    # Transform the world point to pixel coordinates using the homography matrix
    pixel_point_homogeneous = np.dot(np.linalg.inv(homography_matrix), world_point_homogeneous)

    # Convert back to Cartesian coordinates
    pixel_point = pixel_point_homogeneous[:2] / pixel_point_homogeneous[2]

    return pixel_point