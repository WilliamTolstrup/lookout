#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <control_toolbox/pid.hpp>

class ControllerNode : public rclcpp::Node
{
public:
    ControllerNode() : Node("controller_node"), linear_pid_(), angular_pid_()
    {
        // Initialize PID parameters (adjust these as needed)
        linear_pid_.initPid(1.0, 0.0, 0.1, 0.0, 0.0);
        angular_pid_.initPid(1.0, 0.0, 0.1, 0.0, 0.0);

        // Initialize subscribers and publishers
        cmd_vel_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "/cmd_vel", 10,
            std::bind(&ControllerNode::cmdVelCallback, this, std::placeholders::_1));

        motor_pub_ = this->create_publisher<geometry_msgs::msg::Twist>(
            "/robot/motor_commands", 10);

        // Initialize last time
        last_time_ = this->get_clock()->now();
    }

private:
    void cmdVelCallback(const geometry_msgs::msg::Twist::SharedPtr msg)
    {
        // Get the current time
        rclcpp::Time current_time = this->get_clock()->now();
        uint64_t dt = (current_time - last_time_).nanoseconds(); // dt in nanoseconds
        last_time_ = current_time;

        // Use the PID controllers to compute motor commands
        double linear_output = linear_pid_.computeCommand(msg->linear.x, 0.0, dt);
        double angular_output = angular_pid_.computeCommand(msg->angular.z, 0.0, dt);

        // Publish the motor commands
        auto motor_msg = geometry_msgs::msg::Twist();
        motor_msg.linear.x = linear_output;
        motor_msg.angular.z = angular_output;
        motor_pub_->publish(motor_msg);
    }

    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr motor_pub_;
    control_toolbox::Pid linear_pid_;
    control_toolbox::Pid angular_pid_;

    rclcpp::Time last_time_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ControllerNode>());
    rclcpp::shutdown();
    return 0;
}
