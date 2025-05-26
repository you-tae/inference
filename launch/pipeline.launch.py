# launch/seld_pipeline.launch.py
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    pkg_share      = get_package_share_directory('inference')
    tb3_gazebo     = get_package_share_directory('turtlebot3_gazebo')
    nav2_bringup   = get_package_share_directory('nav2_bringup')

    map_yaml    = os.path.join(pkg_share, 'map', 'map.yaml')
    nav2_params = os.path.join(pkg_share, 'config', 'nav2_params.yaml')

    return LaunchDescription([
        # 1) Gazebo 에서 TurtleBot3 World 띄우기
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(tb3_gazebo, 'launch', 'turtlebot3_world.launch.py')
            )
        ),

        # 2) Nav2 Localization (map_server + amcl)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(nav2_bringup, 'launch', 'localization_launch.py')
            ),
            launch_arguments={
                'use_sim_time': 'true',
                'autostart':    'true',
                'map':          map_yaml,
                'params_file':  nav2_params,
            }.items()
        ),

        # 3) Nav2 Navigation (costmaps + planner + controllers)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(nav2_bringup, 'launch', 'navigation_launch.py')
            ),
            launch_arguments={
                'use_sim_time': 'true',
                'autostart':    'true',
                'map':          map_yaml,
                'params_file':  nav2_params,
            }.items()
        ),

        # 4) 테스트용 static transform (map → odom)
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_map_to_odom',
            arguments=['0','0','0','0','0','0','map','odom'],
            output='screen'
        ),

        # 5) SELD 파이프라인 노드들
        Node(package='inference',
             executable='audio_buffer_node',
             name='audio_buffer',
             output='screen'),
        Node(package='inference',
             executable='inference_node',
             name='inference',
             output='screen'),
        Node(package='inference',
             executable='aggregator_node',
             name='aggregator',
             output='screen'),
        Node(package='inference',
             executable='probability_mapper',
             name='probability_mapper',
             output='screen'),
        Node(package='inference',
             executable='dynamic_costmap_node',
             name='dynamic_costmap',
             output='screen'),
    ])
