import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import RPi.GPIO as GPIO

# Pin definitions
IN1 = 17  # MotorA (4WD motors)
IN2 = 27
IN3 = 22  # MotorB (Ackermann steering)
IN4 = 23
ENA = 12  # PWM for MotorA
ENB = 12  # (Tied with ENA for simplicity)

# GPIO setup
GPIO.setmode(GPIO.BCM)
GPIO.setup([IN1, IN2, IN3, IN4], GPIO.OUT)
GPIO.setup(ENA, GPIO.OUT)

# Initialize PWM
pwm_a = GPIO.PWM(ENA, 1000)  # 1kHz frequency
pwm_a.start(0)  # Start with 0% duty cycle (stopped)

# Helper functions
def stop():
    GPIO.output([IN1, IN2, IN3, IN4], GPIO.LOW)
    pwm_a.ChangeDutyCycle(0)

def move_forward(speed=50):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    pwm_a.ChangeDutyCycle(speed)

def move_backward(speed=50):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    pwm_a.ChangeDutyCycle(speed)

def turn_left():
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)

def turn_right():
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)

def ackermann_left(speed=10):
    move_forward(speed)
    turn_left()

def ackermann_right(speed=10):
    move_forward(speed)
    turn_right()

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        self.subscription = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.get_logger().info("Robot Controller Initialized")

    def cmd_vel_callback(self, msg):
        linear_speed = msg.linear.x
        angular_speed = msg.angular.z

        self.get_logger().info(f"Received: linear_x={linear_speed}, angular_z={angular_speed}")

        # Thresholds for control logic
        linear_threshold = 0.1
        angular_threshold = 0.1

        # Movement logic
        if abs(linear_speed) > linear_threshold:
            if linear_speed > 0:
                move_forward(speed=50)
            else:
                move_backward(speed=50)
        else:
            stop()

        if abs(angular_speed) > angular_threshold:
            if angular_speed > 0:
                ackermann_left(speed=10)
            else:
                ackermann_right(speed=10)

def main(args=None):
    rclpy.init(args=args)
    robot_controller = RobotController()

    try:
        rclpy.spin(robot_controller)
    except KeyboardInterrupt:
        pass
    finally:
        robot_controller.destroy_node()
        rclpy.shutdown()
        stop()
        GPIO.cleanup()

if __name__ == '__main__':
    main()
