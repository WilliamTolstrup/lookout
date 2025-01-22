import numpy as np
import cv2
import camera_commons

# Assuming your helper functions (image_to_world, world_to_image) are already defined.

def create_occupancy_grid(image, homography_matrix, grid_min_x=-1, grid_max_x=4, grid_min_y=0, grid_max_y=4, grid_resolution=640):
    """
    Create an occupancy grid map from a binary image and a homography matrix.
    
    :param image: The binary input image (obstacle=black, free space=white).
    :param homography_matrix: The homography matrix mapping image points to world coordinates.
    :param grid_min_x: The minimum x-value in world coordinates.
    :param grid_max_x: The maximum x-value in world coordinates.
    :param grid_min_y: The minimum y-value in world coordinates.
    :param grid_max_y: The maximum y-value in world coordinates.
    :param grid_resolution: The resolution of the output grid image (both width and height).
    
    :return: The occupancy grid map as a binary image (black=obstacle, white=free space).
    """
    # Define the world grid resolution (how many grid cells along each axis)
    grid_width = grid_resolution
    grid_height = grid_resolution

    # Create an empty grid (all free space initially)
    occupancy_grid = np.ones((grid_height, grid_width), dtype=np.uint8) * 255  # White for free space

    # Generate world coordinates for each pixel in the output grid
    x_vals = np.linspace(grid_min_x, grid_max_x, grid_width)
    y_vals = np.linspace(grid_min_y, grid_max_y, grid_height)
    
    # For each pixel in the grid, check if it is occupied or free space
    for i in range(grid_width):
        for j in range(grid_height):
            # Convert world coordinates to image coordinates
            world_point = np.array([x_vals[i], y_vals[j]])
            pixel_coords = camera_commons.world_to_image(world_point, homography_matrix)
            
            # Check if the pixel is within the bounds of the input image
            x_pixel, y_pixel = pixel_coords
            if 0 <= x_pixel < image.shape[1] and 0 <= y_pixel < image.shape[0]:
                # Map the pixel from the image: If the pixel is black (obstacle), mark it as occupied
                if image[int(y_pixel), int(x_pixel)] == 0:
                    occupancy_grid[j, i] = 0  # Mark as occupied (black)
    
    return occupancy_grid

# Load the binary image (already loaded as a grayscale image)
image = cv2.imread('/home/william/Datasets/floorspace/floor1_master.png', cv2.IMREAD_GRAYSCALE)

H = np.array([[ 4.52406580e-02,  3.85609932e-02, -4.16458727e+01],
                                           [ 3.84449148e-17, -5.57834326e-02,  6.02461072e+01],
                                           [-1.50888263e-03,  2.90202050e-02,  1.00000000e+00]])

# Example usage with the provided homography matrix
occupancy_grid = create_occupancy_grid(image, H)

# Show the resulting occupancy grid
cv2.imshow('Occupancy Grid', occupancy_grid)
cv2.waitKey(0)
cv2.destroyAllWindows()
