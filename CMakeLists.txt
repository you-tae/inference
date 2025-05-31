cmake_minimum_required(VERSION 3.8)
project(inference)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(visualization_msgs REQUIRED)


# message generation
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/SeldTimedResult.msg"
  DEPENDENCIES std_msgs
)

# python node
install(PROGRAMS
  src/tf_buffer_node.py
  src/audio_buffer_node.py
  src/dynamic_costmap_node.py
  src/extract_features.py
  src/inference_node.py
  src/loss.py
  src/main_model.py
  src/metrics.py
  src/node_kill.py
  src/parameters.py
  src/probability_mapper.py
  src/utils.py
  src/visualization_node.py

  DESTINATION lib/${PROJECT_NAME}
)

# launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)



if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
