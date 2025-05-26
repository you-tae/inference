import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 1) SELD 파이프라인 노드들
        Node(
            package='inference',
            executable='audio_buffer_node',
            name='audio_buffer',
            output='screen'
        ),
        Node(
            package='inference',
            executable='inference_node',
            name='inference',
            output='screen',
            parameters=[{'model_weights': 'dummy_weights_path'}]  # 더미 모델 weight 경로 설정
        ),
        Node(
            package='inference',
            executable='aggregator_node',
            name='aggregator',
            output='screen'
        ),
        Node(
            package='inference',
            executable='probability_mapper',
            name='probability_mapper',
            output='screen'
        ),
        Node(
            package='inference',
            executable='dynamic_costmap_node',
            name='dynamic_costmap',
            output='screen'
        ),
    ])
