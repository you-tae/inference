import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    # 첫 번째 'turtlebot3_gazebo' 패키지의 launch 파일 경로
    turtlebot3_gazebo_launch_file_dir = os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'launch')
    # 두 번째 'turtlebot3_navigation2' 패키지의 launch 파일 경로
    turtlebot3_navigation2_launch_file_dir = os.path.join(get_package_share_directory('turtlebot3_navigation2'), 'launch')

    # 파라미터를 받는 부분 (기본값 설정)
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    use_rviz = LaunchConfiguration('use_rviz', default='true')

    map_dir = LaunchConfiguration(
        'map',
        default=os.path.join(
            get_package_share_directory('turtlebot3_navigation2'),
            'map',
            'map2.yaml')
    )

    param_file_name = os.environ.get('TURTLEBOT3_MODEL', 'burger') + '.yaml'  # 환경 변수 TURTLEBOT3_MODEL로부터 파일명 설정
    param_dir = LaunchConfiguration(
        'params_file',
        default=os.path.join(
            get_package_share_directory('turtlebot3_navigation2'),
            'param',
            param_file_name)
    )

    nav2_launch_file_dir = os.path.join(get_package_share_directory('nav2_bringup'), 'launch')
    rviz_config_dir = os.path.join(
        get_package_share_directory('nav2_bringup'),
        'rviz',
        'nav2_default_view.rviz')  

    # DeclareLaunchArgument로 파라미터를 선언
    ld = LaunchDescription([
        DeclareLaunchArgument(
            'map',
            default_value=map_dir,
            description='Full path to map file to load'
        ),
        DeclareLaunchArgument(
            'params_file',
            default_value=param_dir,
            description='Full path to param file to load'
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation (Gazebo) clock if true'
        ),
        DeclareLaunchArgument(
            'use_rviz',
            default_value='true',
            description='Launch rviz for visualization'
        ),
    ])

    # 첫 번째 Gazebo launch 파일
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(turtlebot3_gazebo_launch_file_dir, 'turtlebot3_world.launch.py')
        ),
        launch_arguments={'use_sim_time': use_sim_time}.items()  # 사용 시 인자 전달
    )

    # 두 번째 Navigation launch 파일
    navigation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(turtlebot3_navigation2_launch_file_dir, 'navigation2.launch.py')
        ),
        launch_arguments={'use_sim_time': use_sim_time, 'params_file': param_dir, 'map': map_dir}.items()  # 파라미터 전달
    )

    # Spawn 로봇을 한 번만 하도록 하는 로직 추가
    spawn_turtlebot_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(turtlebot3_gazebo_launch_file_dir, 'spawn_turtlebot3.launch.py')
        ),
        launch_arguments={
            'x_pose': '0.0',  # 기본 위치 X
            'y_pose': '0.0',  # 기본 위치 Y
            'z_pose': '0.0',  # 기본 위치 Z
        }.items()
    )

    # 두 개의 launch 파일과 spawn 한 번만 호출하기
    ld.add_action(gazebo_launch)
    ld.add_action(navigation_launch)
    ld.add_action(spawn_turtlebot_launch)

    return ld
