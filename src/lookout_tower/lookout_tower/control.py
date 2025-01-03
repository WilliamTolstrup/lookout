import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, String

class SimpleMotorControlNode(Node):
    def __init__(self):
        super().__init__('simple_motor_control')
        
        # Publisher to motor control topics
        self.motor_a_pub = self.create_publisher(Float64, 'motor_a_speed', 10)
        self.motor_b_pub = self.create_publisher(Float64, 'motor_b_speed', 10)
        
        # Subscriber to commands (e.g., "move_forward", "turn_left")
        self.create_subscription(String, 'cmd_vel', self.cmd_callback, 10)

    def cmd_callback(self, msg):
        command = msg.data
        
        if command == 'move_forward':
            self.publish_motor_speed(5.0, 5.0)  # Both motors go forward
        elif command == 'move_backward':
            self.publish_motor_speed(-5.0, -5.0)  # Both motors reverse
        elif command == 'turn_left':
            self.publish_motor_speed(-5.0, 5.0)  # Turn left
        elif command == 'turn_right':
            self.publish_motor_speed(5.0, -5.0)  # Turn right
        elif command == 'stop':
            self.publish_motor_speed(0.0, 0.0)  # Stop
        else:
            self.get_logger().info(f"Unknown command: {command}")

    def publish_motor_speed(self, motor_a_voltage, motor_b_voltage):
        # Publish to the motor control topics
        self.motor_a_pub.publish(Float64(data=motor_a_voltage))
        self.motor_b_pub.publish(Float64(data=motor_b_voltage))


def main(args=None):
    rclpy.init(args=args)
    
    node = SimpleMotorControlNode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
