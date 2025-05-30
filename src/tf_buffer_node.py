#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from inference.msg import SeldTimedResult
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
import tf2_ros
import math
from collections import deque, defaultdict
from visualization_msgs.msg import Marker
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_pose
from nav_msgs.msg import OccupancyGrid

class TFBufferNode(Node):
    def __init__(self):
        super().__init__('tf_buffer_node')

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.transformed_buffer = deque(maxlen=50)
        self.occupancy_grid = None

        self.create_subscription(
            SeldTimedResult,
            'seld_timed_result',
            self.seld_callback,
            10
        )

        self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )

        self.marker_pub = self.create_publisher(Marker, '/transformed_marker', 10)
        self.pc_pub = self.create_publisher(PointCloud2, '/virtual_obstacles', 10)

        self.create_timer(0.5, self.publish_heatmap_markers)
        self.get_logger().info("TFBufferNode started with PointCloud2 and marker publishers.")

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
        except Exception:
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
                self.transformed_buffer.append({
                    'timestamp': timestamp,
                    'x': pose_in_map.position.x,
                    'y': pose_in_map.position.y,
                    'class': cls
                })
            except Exception as e:
                self.get_logger().error(f"Error transforming pose: {e}")

    def count_grid_occurrences(self, buffer, grid_size=0.2):
        heatmap = defaultdict(int)
        for item in buffer:
            gx = round(item['x'] / grid_size) * grid_size
            gy = round(item['y'] / grid_size) * grid_size
            heatmap[(gx, gy)] += 1
        return heatmap

    def is_free_space(self, x, y):
        if not self.occupancy_grid:
            return True
        info = self.occupancy_grid.info
        mx = int((x - info.origin.position.x) / info.resolution)
        my = int((y - info.origin.position.y) / info.resolution)
        if 0 <= mx < info.width and 0 <= my < info.height:
            val = self.occupancy_grid.data[my * info.width + mx]
            return val == 0
        return False

    # def publish_heatmap_markers(self):
    #     # 1) 기존 마커 전부 삭제
    #     clear_marker = Marker()
    #     clear_marker.header.frame_id = 'map'
    #     clear_marker.header.stamp = self.get_clock().now().to_msg()
    #     clear_marker.action = Marker.DELETEALL
    #     self.marker_pub.publish(clear_marker)

    #     # 2) heatmap 계산
    #     heatmap = self.count_grid_occurrences(self.transformed_buffer)

    #     # 3) 로봇 위치 lookup
    #     try:
    #         tf_map_to_base = self.tf_buffer.lookup_transform(
    #             target_frame='map',
    #             source_frame='base_link',
    #             time=rclpy.time.Time()
    #         )
    #         rx = tf_map_to_base.transform.translation.x
    #         ry = tf_map_to_base.transform.translation.y
    #     except Exception as e:
    #         self.get_logger().warn(f"TF lookup failed in publish: {e}")
    #         rx = ry = None

    #     # 4) 새로운 마커 퍼블리시 (1m 이내는 스킵)
    #     for i, ((x, y), count) in enumerate(heatmap.items()):
    #         # free space 검사
    #         if not self.is_free_space(x, y):
    #             continue

    #         # 로봇과의 거리 계산 후 1m 미만이면 스킵
    #         if rx is not None and ry is not None:
    #             dist = math.hypot(x - rx, y - ry)
    #             if dist < 1.0:
    #                 continue

    #         marker = Marker()
    #         marker.header.frame_id = 'map'
    #         marker.header.stamp = self.get_clock().now().to_msg()
    #         marker.id = i
    #         marker.type = Marker.CUBE
    #         marker.action = Marker.ADD
    #         marker.pose.position.x = x
    #         marker.pose.position.y = y
    #         marker.pose.position.z = 0.0
    #         marker.pose.orientation.w = 1.0
    #         scale = 0.2 + 0.1 * min(count, 5)
    #         marker.scale.x = scale
    #         marker.scale.y = scale
    #         marker.scale.z = 0.2
    #         marker.color.a = 1.0
    #         marker.color.r = min(1.0, 0.2 + 0.15 * count)
    #         marker.color.g = max(0.0, 1.0 - 0.2 * count)
    #         marker.color.b = 0.0
    #         self.marker_pub.publish(marker)

    #     # 5) PointCloud2 퍼블리시 (heatmap 중 1m 이상만)
    #     header = Header()
    #     header.frame_id = 'map'
    #     header.stamp = self.get_clock().now().to_msg()
    #     points = []
    #     for (x, y), cnt in heatmap.items():
    #         if not self.is_free_space(x, y):
    #             continue
    #         if rx is not None and ry is not None and math.hypot(x - rx, y - ry) < 1.0:
    #             continue
    #         points.append((x, y, 0.0))

    #     # PointField 정의
    #     fields = []
    #     for name, offset in [('x', 0), ('y', 4), ('z', 8)]:
    #         pf = PointField()
    #         pf.name = name
    #         pf.offset = offset
    #         pf.datatype = PointField.FLOAT32
    #         pf.count = 1
    #         fields.append(pf)

    #     pc2_msg = point_cloud2.create_cloud(header, fields, points)
    #     self.pc_pub.publish(pc2_msg)

    #     self.get_logger().info(
    #         f"Cleared old markers and published {len(heatmap)} new markers, "
    #         f"{len(points)} points (distance ≥ 1m)."
    #     )

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
