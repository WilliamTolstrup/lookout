import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    # Set paths to packages
    pkg_gazebo_ros = FindPackageShare(package='gazebo_ros').find('gazebo_ros')
    pkg_lookout = FindPackageShare(package='lookout_tower').find('lookout_tower')

    # World
    world_file_name = 'living_room_world'
    world_path = os.path.join(pkg_lookout, 'worlds', world_file_name)

    # Robot model
    robot_model_name = 'exort.urdf'
    robot_description = os.path.join(pkg_lookout, 'urdf', robot_model_name)

    # Sim param
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    return LaunchDescription([

            DeclareLaunchArgument(
                'use_sim_time',
                default_value='false',
                description='Use simulation (Gazebo) clock if true'),
            
            DeclareLaunchArgument(
                name='world',
                default_value=world_path,
                description='Use custom world'),

            ExecuteProcess(
                cmd=['gazebo', '--verbose', '-s', 'libgazebo_ros_factory.so'],
                output='screen'),

            Node(
                package='robot_state_publisher',
                executable='robot_state_publisher',
                name='robot_state_publisher',
                output='screen',
                parameters=[{'use_sim_time': use_sim_time}],
                arguments=[robot_description]),

            Node(
                package='joint_state_publisher',
                executable='joint_state_publisher',
                name='joint_state_publisher',
                output='screen',
                parameters=[{'use_sim_time': use_sim_time}]
                ),

            Node(
                package='gazebo_ros',
                executable='spawn_entity.py',
                name='urdf_spawner',
                output='screen',
                arguments=["-topic", "/robot_description", "-entity", "cam_bot"])
    ])