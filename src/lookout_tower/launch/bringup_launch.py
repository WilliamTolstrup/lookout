
"""
Bare bones gazebo bringup with custom world and custom robot
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    lookout_dir = get_package_share_directory('lookout_tower')
    launch_dir = os.path.join(lookout_dir, 'launch')

    # Robot model
    robot_model_name = 'exort.urdf'
    robot_description_path = os.path.join(lookout_dir, 'urdf', robot_model_name)

    # Simulation settings
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    world = LaunchConfiguration('world')
    simulator = LaunchConfiguration('simulator')
    robot = LaunchConfiguration('robot')

    # Declare the launch arguments
    declare_world_cmd = DeclareLaunchArgument(
        'world',
        default_value=os.path.join(lookout_dir, 'worlds', 'maze.world'),
        description='Full path to world file to load')
    
    declare_robot_cmd = DeclareLaunchArgument(
        'robot',
        default_value=os.path.join(lookout_dir, 'urdf', 'exort.urdf'),
        description='Full path to robot model to load')

    declare_simulator_cmd = DeclareLaunchArgument(
        'simulator',
        default_value='gazebo',
        description='The simulator to use (gazebo or gzserver)')

    declare_use_robot_state_pub_cmd = DeclareLaunchArgument(
        'use_robot_state_pub',
        default_value='True',
        description='Whether to start the robot state publisher')

    # Start Gazebo with plugin providing the robot spawing service
    start_gazebo_cmd = ExecuteProcess(
        cmd=[simulator, '--verbose', '-s', 'libgazebo_ros_init.so',
                                     '-s', 'libgazebo_ros_factory.so', world],
        output='screen')
    
    # Spawn robot into Gazebo
    spawn_robot_cmd = Node(
        package='gazebo_ros', 
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'exort',  # Name of the robot in Gazebo
            '-file', robot_description_path,  # Path to URDF or SDF file
            '-x', '0', '-y', '0', '-z', '0.1'  # Initial spawn position
        ],
        output='screen'
    )
   
    
    # Create the launch description and populate
    ld = LaunchDescription()

    # Declare the launch options
    ld.add_action(declare_simulator_cmd)
    ld.add_action(declare_world_cmd)
    ld.add_action(declare_robot_cmd)
    ld.add_action(declare_use_robot_state_pub_cmd)

    # Add the actions to start gazebo, robots and simulations
    ld.add_action(start_gazebo_cmd)
    ld.add_action(spawn_robot_cmd)

    return ld