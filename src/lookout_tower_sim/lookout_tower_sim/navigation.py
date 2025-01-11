import heapq
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Vector3, Point, PoseStamped, Pose
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import Bool

import cv2
import numpy as np
import math
from scipy.ndimage import distance_transform_edt

class AStarPlanner(Node):
    def __init__(self):
        super().__init__('astar_planner')
        self.grid = None  # Occupancy grid
        self.resolution = None
        self.origin_x = None
        self.origin_y = None
        self.grid_width = None
        self.grid_height = None
        self.grid_min_x, self.grid_max_x = -6, 6  # 12m x 12m grid
        self.grid_min_y, self.grid_max_y = -6, 6

        self.static_grid = None
        self.dynamic_grid = None

        self.robot_pose = None  # Robot pose in world coordinates
        self.goal = None  # Goal in world coordinates

        self.robot_pose_sub = self.create_subscription(Pose, '/robot/pose', self.robot_pose_callback, 10)
        self.static_occupancy_grid_sub = self.create_subscription(OccupancyGrid, '/static_map', self.update_static_occupancy_grid, 10)
       # self.dynamic_occupancy_grid_sub = self.create_subscription(OccupancyGrid, '/dynamic_map', self.update_dynamic_occupancy_grid, 10)
        self.goal_sub = self.create_subscription(Point, '/goal', self.goal_callback, 10)

        self.path_pub = self.create_publisher(Path, '/path', 10)

        #self.plan_path() # Initial path planning
        self.timer_period = 20.0  # seconds
        self.timer = self.create_timer(self.timer_period, self.plan_path)

    def goal_reached_callback(self, msg):
        if msg.data:
            self.get_logger().info("Goal reached, re-planning path")
            self.plan_path()

    def robot_pose_callback(self, msg):
        self.robot_pose = msg # Robot pose in world coordinates

    def goal_callback(self, msg):
        self.goal = (msg.x, msg.y)

    def plan_path(self):
        if self.robot_pose is None or self.goal is None or self.grid is None:
            self.get_logger().warn("Waiting for necessary data: robot pose, goal, and/or occupancy grid")
            return
        
        start = (self.robot_pose.position.x, self.robot_pose.position.y)
        goal = (self.goal[0], self.goal[1])

        try:
            path = self.a_star(start, goal)
            if path is not None:
                self.publish_path(path)
                self.get_logger().info("Path planning successful", once=True)

        except ValueError as e:
            self.get_logger().warn(f"Path planning failed: {e}")

    def apply_cost_gradient(self):
        if self.grid is None:
            self.get_logger().warn("No occupancy grid data available")
            return

        original_grid = self.grid.copy()

        free_space = self.grid == 127

        # Compute the distance transform
        distance_transform = distance_transform_edt(free_space)

        # Normalize the distance values to range [0, 255]
        max_distance = np.max(distance_transform)


        if max_distance > 0:
            gradient_cost = (np.log1p(distance_transform) / np.log1p(max_distance)) * 127
        else:
            gradient_cost = distance_transform

        # Ensure obstacles stay black (0)
        costmap = np.where(free_space, gradient_cost, 0).astype(np.float32)
#        print("Costmap values:", np.unique(costmap))  # Debug final costmap values

        # Update the grid with the costmap
        self.grid = costmap

        # Display for visualization
#        print("Costmap values:", np.unique(costmap))  # Debug final costmap values
        self.display_grid(self.grid, title="Cost Gradient Grid")
#        self.display_grid(original_grid, title="Original Grid")



    def inflate_obstacles(self, inflation_radius):
        # Kernel size based on inflation radius
        kernel_size = int(2 * inflation_radius + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Inflate obstacles
        binary_grid = np.where(self.grid == 0, 1, 0).astype(np.uint8)
        inflated_grid = cv2.dilate(binary_grid, kernel, iterations=1)
        
        # Update costmap: Inflated cells become obstacles (cost = 127)
        self.grid = np.where(inflated_grid > 0, 0, self.grid)

#        self.display_grid(self.grid, title="Inflated Grid")

    def display_grid(self, grid, title="Grid"):
        # Normalize grid for display (values between 0 and 255 for visualization)
        displayable_grid = (grid.astype(np.float32) / 127 * 255).astype(np.uint8)
        
        # Resize the grid for better visibility
        height, width = displayable_grid.shape
        resized_grid = cv2.resize(displayable_grid, (width * 10, height * 10), interpolation=cv2.INTER_NEAREST)
        
        cv2.imshow(title, resized_grid)
        cv2.waitKey(1)  # Small delay to update the OpenCV window



    def convert_to_numpy(self, occupancy_grid_msg):
        self.resolution = occupancy_grid_msg.info.resolution
        self.origin_x = occupancy_grid_msg.info.origin.position.x
        self.origin_y = occupancy_grid_msg.info.origin.position.y
        self.grid_width = occupancy_grid_msg.info.width
        self.grid_height = occupancy_grid_msg.info.height

        data = np.array(occupancy_grid_msg.data, dtype=np.int8)
        grid = data.reshape((occupancy_grid_msg.info.height, occupancy_grid_msg.info.width))
        return grid

    def update_static_occupancy_grid(self, occupancy_grid_msg):
        self.static_grid = self.convert_to_numpy(occupancy_grid_msg)
        self.update_combined_occupancy_grid()

    def update_dynamic_occupancy_grid(self, occupancy_grid_msg):
        self.dynamic_grid = self.convert_to_numpy(occupancy_grid_msg)
        self.update_combined_occupancy_grid()

    def update_combined_occupancy_grid(self):
        if self.static_grid is not None and self.dynamic_grid is not None:
            self.grid = np.maximum(self.static_grid, self.dynamic_grid)
        
        elif self.static_grid is not None and self.dynamic_grid is None:
            self.grid = self.static_grid
            self.get_logger().warn("No dynamic map data available", once=True)

        elif self.static_grid is None and self.dynamic_grid is not None:
            self.grid = self.dynamic_grid
            self.get_logger().warn("No static map data available", once=True)

        else:
            self.grid = None
            self.get_logger().warn("No occupancy grid data available")

        self.inflate_obstacles(inflation_radius=1.15)
        self.apply_cost_gradient()

    def a_star(self, start, goal):
        start = self.world_to_grid(*start)
        goal = self.world_to_grid(*goal)
        # Priority queue
        open_list = []
        heapq.heappush(open_list, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_list:
            _, current = heapq.heappop(open_list)
            if current == goal:
                return self.reconstruct_path(came_from, current)

            x, y = current
            for neighbor in self.get_neighbors(x, y):
                tentative_g_score = g_score[current] + self.gradient_cost(neighbor)
                if tentative_g_score == float('inf'):
                    continue # Skip obstacles

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + 0.5 * self.heuristic(neighbor, goal)
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))

        return None

    def heuristic(self, a, b):
        # Euclidean distance
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def gradient_cost(self, node):
        x, y = node

        if self.grid[y][x] == 0:
            return float('inf')

        return np.max(self.grid) - self.grid[y][x]

        # max_cost = np.max(self.grid)
        # inverted_cost = max_cost - self.grid[y][x]
        # return inverted_cost / max_cost

    def get_neighbors(self, x, y):
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),  # Cardinal
                    (-1, -1), (-1, 1), (1, -1), (1, 1)]  # Diagonal
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                if self.grid[ny][nx] > 0: # Exclude obstacles
                    neighbors.append((nx, ny))
        return neighbors

    def reconstruct_path(self, came_from, current):
        path = []
        while current in came_from:
            path.append(self.grid_to_world(*current))
            current = came_from[current]
        return path[::-1]



    def world_to_grid(self, x, y):
        grid_x = int((x - self.grid_min_x) / self.resolution)
        grid_y = int((y - self.grid_min_y) / self.resolution)
        return grid_x, grid_y

    def grid_to_world(self, grid_x, grid_y):
        x = grid_x * self.resolution + self.grid_min_x
        y = grid_y * self.resolution + self.grid_min_y
        return x, y

    # # def is_valid_cell(self, x, y):
    # #     return (0 <= x < self.grid.shape[0] and
    # #             0 <= y < self.grid.shape[1] and
    # #             self.grid[x, y] > 0)  # Ensure >0 for valid cells


    # def find_nearest_valid_cell(self, grid_cell):
    #     x, y = grid_cell
    #     directions = [(-1, 0), (1, 0), (0, -1), (0, 1),  # Cardinal
    #                 (-1, -1), (-1, 1), (1, -1), (1, 1)]  # Diagonal
    #     queue = [(x, y)]
    #     visited = set()
        
    #     while queue:
    #         cx, cy = queue.pop(0)
    #         if (cx, cy) in visited:
    #             continue
    #         visited.add((cx, cy))
            
    #         # Check if cell is valid
    #         if self.is_valid_cell(cx, cy) and self.grid[cx, cy] > 0:
    #             return cx, cy
            
    #         # Add neighbors to the queue
    #         for dx, dy in directions:
    #             queue.append((cx + dx, cy + dy))
        
    #     raise ValueError("No valid cells found near the given point.")


    # def is_valid_cell(self, x, y):
    #     if not (0 <= x < self.grid_width and 0 <= y < self.grid_height):
    #         return False
    #     return self.grid[y, x] > 100 # == 127


    # def heuristic(self, x1, y1, x2, y2):
    #     return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  # Euclidean distance

    # def plan(self, start, goal):
    #     start_grid = self.world_to_grid(*start)
    #     goal_grid = self.world_to_grid(*goal)

    #     if self.grid[goal_grid[0], goal_grid[1]] == 0:
    #         goal_grid = self.find_nearest_valid_cell(goal_grid)

    #     open_list = []
    #     heapq.heappush(open_list, (0, start_grid))
    #     came_from = {}
    #     cost_so_far = {start_grid: 0}

    #     directions = [(-1, 0), (1, 0), (0, -1), (0, 1),  # Cardinal directions
    #                   (-1, -1), (-1, 1), (1, -1), (1, 1)]  # Diagonals

    #     while open_list:
    #         _, current = heapq.heappop(open_list)

    #         if current == goal_grid:
    #             break

    #         for dx, dy in directions:
    #             neighbor = (current[0] + dx, current[1] + dy)
    #             if not self.is_valid_cell(*neighbor):
    #                 continue # Skip invalid cells
                
    #             gradient_cost = self.grid[neighbor[0], neighbor[1]]
    #             if gradient_cost == 0:
    #                 continue # Skip obstacles

    #             new_cost = cost_so_far[current] + 0.1 * self.heuristic(*current, *neighbor) + gradient_cost

    #             if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
    #                 cost_so_far[neighbor] = new_cost
    #                 priority = new_cost + self.heuristic(*neighbor, *goal_grid)
    #                 heapq.heappush(open_list, (priority, neighbor))
    #                 came_from[neighbor] = current
    #            # print(f"Current: {current}, Neighbor: {neighbor}, Cost: {new_cost}")


    #     # Backtrack to form the path
    #     path = []
    #     current = goal_grid
    #     while current != start_grid:
    #         path.append(self.grid_to_world(*current))
    #         current = came_from.get(current)
    #         if current is None:
    #             raise ValueError("No path found")
            
    #         if current is None:
    #             # Attempt fallback to lowest-cost neighbors of the goal
    #             neighbors = [(goal_grid[0] + dx, goal_grid[1] + dy) for dx, dy in directions]
    #             valid_neighbors = [n for n in neighbors if self.is_valid_cell(*n) and self.grid[n[0], n[1]] > 0]
    #             if valid_neighbors:
    #                 goal_grid = min(valid_neighbors, key=lambda n: cost_so_far.get(n, float('inf')))
    #                 continue
    #             else:
    #                 raise ValueError("No path found")





    #     path.reverse()
    #     return path

    def publish_path(self, path):
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"

        for x, y in path:
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0  # No rotation
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)

def main(args=None):
    rclpy.init(args=args)
    astar_node = AStarPlanner()
    rclpy.spin(astar_node)
    astar_node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
