#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from inference.msg import SeldTimedResult
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32
from std_msgs.msg import Header
import numpy as np
from rclpy.qos import QoSProfile, QoSDurabilityPolicy

class ObstaclePublisher(Node):
    def __init__(self):
        super().__init__('obstacle_publisher')

        # # QoS 설정: TRANSIENT_LOCAL → 퍼블리시 후에도 메시지 유지됨
        # qos_profile = QoSProfile(
        #     depth=1,
        #     durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        # )

        self.sub = self.create_subscription(
            SeldTimedResult,
            '/seld_timed_result',
            self.seld_callback,
            10)
        self.pub = self.create_publisher(PointCloud2, '/virtual_obstacles', 10)

        # 왼/오 위치 (x, y, z)
        self.RIGHT_POS = np.array([2.0, -4.0, 0.0])
        self.LEFT_POS  = np.array([2.0,  4.0, 0.0])

        # 클러스터 생성 설정
        self.cluster_radius    = 1   # 중심에서 ±1 m
        self.cluster_density   = 10     # 10×10 격자

    def seld_callback(self, msg: SeldTimedResult):
        # 클래스=1인 azimuth 모으기
        azs = [az for cls, az in zip(msg.classes, msg.azimuth) if cls == 1]
        if not azs:
            return

        avg_az = sum(azs) / len(azs)
        center = self.LEFT_POS if avg_az > 0 else self.RIGHT_POS
        side   = 'LEFT' if avg_az > 0 else 'RIGHT'

        # 클러스터 포인트 생성 (격자)
        pts = []
        lin = np.linspace(-self.cluster_radius,
                          self.cluster_radius,
                          self.cluster_density)
        for dx in lin:
            for dy in lin:
                x = float(center[0] + dx)
                y = float(center[1] + dy)
                z = float(center[2])
                pts.append([x, y, z])

        # 1) 새 Header 만들기
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'map'   # RViz fixed frame과 일치시킵니다.

        # PointCloud2 생성 및 퍼블리시
        cloud = create_cloud_xyz32(header, pts)
        self.pub.publish(cloud)
        self.get_logger().info(f'[{side}] published {len(pts)}-point cluster around {center.tolist()}')

def main():
    rclpy.init()
    node = ObstaclePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()