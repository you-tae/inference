#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from inference.msg import SeldTimedResult
from geometry_msgs.msg import PoseStamped
import tf2_ros
import math
from collections import deque, defaultdict
from visualization_msgs.msg import Marker
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_pose
import numpy as np
import time
from nav_msgs.msg import OccupancyGrid


class TFBufferNode(Node):
    def __init__(self):
        super().__init__('tf_buffer_node')

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.transformed_buffer = deque(maxlen=50)  # 최근 50개 추론 저장
        self.occupancy_grid = None

        self.subscription = self.create_subscription(
            SeldTimedResult,
            'seld_timed_result',
            self.seld_callback,
            10
        )

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        self.marker_pub = self.create_publisher(Marker, 'transformed_marker', 10)

        # 히트맵 마커를 1초마다 주기적으로 갱신
        self.timer = self.create_timer(1.0, self.publish_heatmap_markers)

        self.get_logger().info("TFBufferNode started.")

    def map_callback(self, msg):
        self.occupancy_grid = msg

    def seld_callback(self, msg):
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        try:
            tf_map_to_base = self.tf_buffer.lookup_transform(
                target_frame='map',
                source_frame='base_link',
                time=rclpy.time.Time()
            )
        except Exception as e:
            return

        for az, dist, cls in zip(msg.azimuth, msg.distance, msg.classes):
            x_bl = dist * 5 * math.cos(math.radians(az))
            y_bl = dist * 5 * math.sin(math.radians(az))

            pose_stamped = PoseStamped()
            pose_stamped.header.frame_id = 'base_link'
            pose_stamped.header.stamp = tf_map_to_base.header.stamp
            pose_stamped.pose.position.x = x_bl
            pose_stamped.pose.position.y = y_bl
            pose_stamped.pose.position.z = 0.0
            pose_stamped.pose.orientation.w = 1.0

            try:
                pose_in_map = do_transform_pose(pose_stamped.pose, tf_map_to_base)
                transformed = {
                    'timestamp': timestamp,
                    'x': pose_in_map.position.x,
                    'y': pose_in_map.position.y,
                    'class': cls
                }
                self.transformed_buffer.append(transformed)

            except Exception as e:
                self.get_logger().error(f"Error transforming pose: {e}")

    def count_grid_occurrences(self, buffer, grid_size=0.2):
        heatmap = defaultdict(int)
        for item in buffer:
            x, y = item['x'], item['y']
            gx = round(x / grid_size) * grid_size
            gy = round(y / grid_size) * grid_size
            heatmap[(gx, gy)] += 1
        return heatmap

    def is_free_space(self, x, y):
        if self.occupancy_grid is None:
            return True  # 맵이 없을 때는 우선 허용

        map_msg = self.occupancy_grid
        resolution = map_msg.info.resolution
        origin = map_msg.info.origin.position
        width = map_msg.info.width
        height = map_msg.info.height

        mx = int((x - origin.x) / resolution)
        my = int((y - origin.y) / resolution)

        if 0 <= mx < width and 0 <= my < height:
            idx = my * width + mx
            value = map_msg.data[idx]
            return value == 0  # 0: free, 100: occupied, -1: unknown
        return False

    def publish_heatmap_markers(self):
        heatmap = self.count_grid_occurrences(self.transformed_buffer)

        for i, ((x, y), count) in enumerate(heatmap.items()):
            if not self.is_free_space(x, y):
                continue

            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.2 + 0.1 * min(count, 5)
            marker.scale.y = 0.2 + 0.1 * min(count, 5)
            marker.scale.z = 0.2

            marker.color.a = 1.0
            marker.color.r = min(1.0, 0.2 + 0.15 * count)
            marker.color.g = max(0.0, 1.0 - 0.2 * count)
            marker.color.b = 0.0

            self.marker_pub.publish(marker)

        self.get_logger().info(f"Published {len(heatmap)} heatmap markers.")

    def get_buffer(self):
        return list(self.transformed_buffer)


def main():
    rclpy.init()
    node = TFBufferNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
