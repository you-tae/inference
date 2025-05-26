import launch
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # audio_buffer_node 실행
        Node(
            package='inference',  # 패키지 이름
            executable='audio_buffer_node.py',  # 실행할 노드 이름
            name='audio_buffer_node',  # 노드 이름 (선택 사항)
            output='screen',  # 화면에 출력
        ),
        # inference_node 실행
        Node(
            package='inference',  # 패키지 이름
            executable='inference_node.py',  # 실행할 노드 이름
            name='inference_node',  # 노드 이름 (선택 사항)
            output='screen',  # 화면에 출력
        ),
        # 추가적으로 다른 노드들도 여기에 추가 가능
    ])
