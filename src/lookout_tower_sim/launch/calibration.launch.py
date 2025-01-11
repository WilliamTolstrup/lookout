
"""
Calibration launch file to find homographies for two cameras
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction, Shutdown
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    bringup_dir = get_package_share_directory('lookout_tower_sim')

    # Simulation settings
    world = LaunchConfiguration('world')
    simulator = LaunchConfiguration('simulator')
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    # Declare the launch arguments
    declare_world_cmd = DeclareLaunchArgument(
        'world',
        default_value=os.path.join(bringup_dir, 'worlds', 'calibration.world'),
        description='Full path to world file to load')

    declare_simulator_cmd = DeclareLaunchArgument(
        'simulator',
        default_value='gazebo',
        description='The simulator to use (gazebo or gzserver)')
    
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use sim time if true')

    # Start Gazebo with plugin providing the robot spawing service
    start_gazebo_cmd = ExecuteProcess(
        cmd=[simulator, '--verbose', '-s', 'libgazebo_ros_init.so',
                                     '-s', 'libgazebo_ros_factory.so', world,
                                     '--log-level', 'fatal'],
        output='log',
        additional_env={"IGN_PARTITION": "quiet"})

    # Run calibration node
    calibration_node = Node(
        package="lookout_tower_sim",
        executable="find_homographies",
        name="calibration",
        output="screen",
        arguments=["--ros-args", "--log-level", "info"],
    )

    shutdown_timer = TimerAction(
        period=2.0,
        actions=[Shutdown(reason="Calibration finished")],
    )

    
    # Create the launch description and populate
    ld = LaunchDescription()

    # Declare the launch options
    ld.add_action(declare_simulator_cmd)
    ld.add_action(declare_world_cmd)
    ld.add_action(declare_use_sim_time)

    # Add the actions to start gazebo, robots and simulations
    ld.add_action(start_gazebo_cmd)
    ld.add_action(calibration_node)
    ld.add_action(shutdown_timer)

    return ld