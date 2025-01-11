
"""
Bare bones gazebo bringup with custom world and custom robot
"""

import os
import xacro

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    bringup_dir = get_package_share_directory('lookout_tower_sim')
    launch_dir = os.path.join(bringup_dir, 'launch')
    description_dir = os.path.join(bringup_dir, 'description')

    # Robot model, process the URDF file
    pkg_path = os.path.join(get_package_share_directory('lookout_tower_sim'))
    xacro_file = os.path.join(pkg_path,'description','robot.urdf.xacro')
    robot_description_config = xacro.process_file(xacro_file)

    # Simulation settings
    world = LaunchConfiguration('world')
    simulator = LaunchConfiguration('simulator')
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    rviz_config = LaunchConfiguration("rviz_config")

    # Declare the launch arguments
    declare_world_cmd = DeclareLaunchArgument(
        'world',
        default_value=os.path.join(bringup_dir, 'worlds', 'maze.world'),
        description='Full path to world file to load')

    declare_simulator_cmd = DeclareLaunchArgument(
        'simulator',
        default_value='gazebo',
        description='The simulator to use (gazebo or gzserver)')
    
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use sim time if true')
    
    declare_rviz_config_arg = DeclareLaunchArgument(
        'rviz_config',
        default_value=os.path.join(bringup_dir, 'config', 'view_robot.rviz'),
        description="Absolute path to rviz config"
    )

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
            '-topic', 'robot_description', # Subscribe to the robot_description topic
            '-entity', 'robot',  # Name of the robot in Gazebo
        ],
        output='screen'
    )

    # Create a robot_state_publisher node
    rsp_params = {'robot_description': robot_description_config.toxml(), 'use_sim_time': use_sim_time}
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[rsp_params]
    )

    # Create joint_state_publisher node
    joint_state_publisher = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # Enables the map to be visualized in rviz
    tf2_map = Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='map_to_odom',
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom']
        )
    tf2_static_map = Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_map_to_map',
            arguments=['0', '0', '0', '0', '0', '0', 'static_map', 'map']
        )
    tf2_dynamic_map = Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='dynamic_map_to_map',
            arguments=['0', '0', '0', '0', '0', '0', 'dynamic_map', 'map']
        )

    # Open Rviz2 with config file
    rviz2 = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        parameters=[{'use_sim_time': use_sim_time}],
        arguments=[
            "-d", rviz_config
        ]
    )

    # Launch custom nodes
    # floor_projection = Node(
    #     package='lookout_tower',
    #     executable='floor_projection',
    #     name='floor_projection',
    #     output='screen',
    #     parameters=[{'use_sim_time': use_sim_time}]
    # )

    # localization = Node(
    #     package='lookout_tower',
    #     executable='localization',
    #     name='localization',
    #     output='screen',
    #     parameters=[{'use_sim_time': use_sim_time}]
    # )

   
    
    # Create the launch description and populate
    ld = LaunchDescription()

    # Declare the launch options
    ld.add_action(declare_simulator_cmd)
    ld.add_action(declare_world_cmd)
    ld.add_action(declare_use_sim_time)
    ld.add_action(declare_rviz_config_arg)

    # Add the actions to start gazebo, robots and simulations
    ld.add_action(robot_state_publisher)
    ld.add_action(joint_state_publisher)
    ld.add_action(start_gazebo_cmd)
    ld.add_action(spawn_robot_cmd)
    ld.add_action(rviz2)
    ld.add_action(tf2_map)
    ld.add_action(tf2_static_map)
    ld.add_action(tf2_dynamic_map)
    # ld.add_action(floor_projection)
    # ld.add_action(localization)

    return ld