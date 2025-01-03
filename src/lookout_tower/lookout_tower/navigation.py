import heapq
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Vector3, Point, PoseStamped
from nav_msgs.msg import OccupancyGrid, Path

import numpy as np
import math

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

        self.robot_pose_sub = self.create_subscription(Vector3, '/robot/pose', self.robot_pose_callback, 10)
        self.static_occupancy_grid_sub = self.create_subscription(OccupancyGrid, '/static_map', self.update_static_occupancy_grid, 10)
       # self.dynamic_occupancy_grid_sub = self.create_subscription(OccupancyGrid, '/dynamic_map', self.update_dynamic_occupancy_grid, 10)
        self.goal_sub = self.create_subscription(Point, '/goal', self.goal_callback, 10)

        self.path_pub = self.create_publisher(Path, '/path', 10)

        self.timer_period = 1.0  # seconds
        self.timer = self.create_timer(self.timer_period, self.plan_path)

    def robot_pose_callback(self, msg):
        self.robot_pose = msg # Robot pose in world coordinates

    def goal_callback(self, msg):
        self.goal = (msg.x, msg.y)

    def plan_path(self):
        if self.robot_pose is None or self.goal is None or self.grid is None:
            self.get_logger().warn("Waiting for necessary data: robot pose, goal, and/or occupancy grid")
            return
        
        start = (self.robot_pose.x, self.robot_pose.y)
        goal = (self.goal[0], self.goal[1])

        try:
            path = self.plan(start, goal)
            self.publish_path(path)
            self.get_logger().info("Path planning successful", once=True)
        except ValueError as e:
            self.get_logger().warn(f"Path planning failed: {e}")


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


    def world_to_grid(self, x, y):
        grid_x = int((x - self.grid_min_x) / self.resolution)
        grid_y = int((y - self.grid_min_y) / self.resolution)
        return grid_x, grid_y

    def grid_to_world(self, grid_x, grid_y):
        x = grid_x * self.resolution + self.grid_min_x
        y = grid_y * self.resolution + self.grid_min_y
        return x, y

    def is_valid_cell(self, x, y):
        return 0 <= x < self.grid_width and 0 <= y < self.grid_height and self.grid[y, x] == 127

    def heuristic(self, x1, y1, x2, y2):
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  # Euclidean distance

    def plan(self, start, goal):
        start_grid = self.world_to_grid(*start)
        goal_grid = self.world_to_grid(*goal)

        open_list = []
        heapq.heappush(open_list, (0, start_grid))
        came_from = {}
        cost_so_far = {start_grid: 0}

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),  # Cardinal directions
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]  # Diagonals

        while open_list:
            _, current = heapq.heappop(open_list)

            if current == goal_grid:
                break

            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                if not self.is_valid_cell(*neighbor):
                    continue

                new_cost = cost_so_far[current] + self.heuristic(*current, *neighbor)
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self.heuristic(*neighbor, *goal_grid)
                    heapq.heappush(open_list, (priority, neighbor))
                    came_from[neighbor] = current

        # Backtrack to form the path
        path = []
        current = goal_grid
        while current != start_grid:
            path.append(self.grid_to_world(*current))
            current = came_from.get(current)
            if current is None:
                raise ValueError("No path found")
        path.reverse()
        return path

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

if __name__ == '__main__':
    main()
