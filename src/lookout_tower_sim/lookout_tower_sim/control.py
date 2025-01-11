import math
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Path
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Bool

class SimpleController(Node):
    def __init__(self):
        super().__init__('simple_controller')
        self.path = []
        self.current_target_index = 0
        self.robot_pose = None  # Robot pose in world coordinates

        self.path_sub = self.create_subscription(Path, '/path', self.path_callback, 10)
        self.robot_pose_sub = self.create_subscription(Pose, '/robot/pose', self.robot_pose_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.timer_period = 0.1  # 10 Hz
        self.timer = self.create_timer(self.timer_period, self.execute_path)

    def path_callback(self, msg):
        self.path = [(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses]
        if self.robot_pose is not None:
            self.current_target_index = self.find_closest_waypoint()
        else:
            self.current_target_index = 0
#        self.get_logger().info(f"Received path with {len(self.path)} waypoints.")

    def robot_pose_callback(self, msg):
        self.robot_pose = msg  # Robot pose in world coordinates

    def execute_path(self):
        if not self.path or self.current_target_index >= len(self.path):
            self.stop_robot()
            return

        if self.robot_pose is None:
            self.get_logger().warn("Robot pose not yet received.")
            return

        #self.current_target_index = self.find_closest_waypoint()

        current_target = self.path[self.current_target_index]

        robot_x = self.robot_pose.position.x
        robot_y = self.robot_pose.position.y

        # Extract yaw from quaternion
        robot_orientation_quat = (self.robot_pose.orientation.x, self.robot_pose.orientation.y,
                                  self.robot_pose.orientation.z, self.robot_pose.orientation.w)
        robot_yaw = R.from_quat(robot_orientation_quat).as_euler('xyz')[2]

        # Compute errors
        dx = current_target[0] - robot_x
        dy = current_target[1] - robot_y
        distance_to_target = math.sqrt(dx**2 + dy**2)
        angle_to_target = math.atan2(dy, dx)
        angular_error = -math.atan2(math.sin(angle_to_target - robot_yaw), math.cos(angle_to_target - robot_yaw))

        # DEBUG
#        self.get_logger().info(f"Robot pose: x={robot_x:.2f}, y={robot_y:.2f}, yaw={math.degrees(robot_yaw):.2f}°")
#        self.get_logger().info(f"Target: x={current_target[0]:.2f}, y={current_target[1]:.2f}")
#        self.get_logger().info(f"dx={dx:.2f}, dy={dy:.2f}, distance_to_target={distance_to_target:.2f}")
#        self.get_logger().info(f"angle_to_target={math.degrees(angle_to_target):.2f}°, angular_error={math.degrees(angular_error):.2f}°")


        # Check if target is reached
        if distance_to_target < 0.2:  # Threshold for reaching a waypoint
            #self.get_logger().info("Reached waypoint.")
            #return
            self.get_logger().info(f"Reached waypoint {self.current_target_index}.")
            self.current_target_index += 1
        else:
            self.move_towards_target(angular_error, distance_to_target)

    def move_towards_target(self, angular_error, distance_to_target):
        cmd = Twist()

        # Control parameters
        max_linear_speed = 0.2
        max_angular_speed = 0.4
        linear_gain = 0.8
        angular_gain = 2.0

        # Compute control inputs
        cmd.linear.x = max(0, min(max_linear_speed, linear_gain * distance_to_target))
        cmd.angular.z = max(-max_angular_speed, min(max_angular_speed, angular_gain * angular_error))

        self.cmd_vel_pub.publish(cmd)
       # self.get_logger().info(f"Publishing cmd: Linear: {cmd.linear.x:.2f}, Angular: {cmd.angular.z:.2f}")

    def find_closest_waypoint(self):
        if not self.path or self.robot_pose is None:
            return None

        robot_x = self.robot_pose.position.x
        robot_y = self.robot_pose.position.y

        min_distance = float('inf')
        closest_index = self.current_target_index  # Default to the current target index

        for i, waypoint in enumerate(self.path):
            dx = waypoint[0] - robot_x
            dy = waypoint[1] - robot_y
            distance = math.sqrt(dx**2 + dy**2)
            if distance < min_distance:
                min_distance = distance
                closest_index = i

        self.get_logger().info(f"Closest waypoint is {closest_index} with distance {min_distance:.2f}, and coordinates {self.path[closest_index]}")
        return closest_index


    def stop_robot(self):
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)
#        self.get_logger().info("Robot stopped.")

def main(args=None):
    import rclpy
    rclpy.init(args=args)
    controller = SimpleController()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
