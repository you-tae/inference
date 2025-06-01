#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from inference.msg import SeldTimedResult
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32
from std_msgs.msg import Header
import numpy as np

class SeldObstacleNode(Node):
    def __init__(self):
        super().__init__('seld_obstacle_node')

        # 1) /seld_timed_result 구독자 설정
        self.sub = self.create_subscription(
            SeldTimedResult,
            '/seld_timed_result',
            self.seld_callback,
            10)

        # 2) /virtual_obstacles 퍼블리셔 설정 (PointCloud2 타입)
        self.pub = self.create_publisher(PointCloud2, '/virtual_obstacles', 10)

        # 3) 왼쪽/오른쪽 장애물 중심 좌표 정의 (예시)
        #    필요하다면 적절한 좌표로 수정하세요.
        self.LEFT_POS  = np.array([3.0,  2.2, 0.0])
        self.RIGHT_POS = np.array([3.0, -2.2, 0.0])

        # 4) 클러스터 생성 파라미터
        self.cluster_radius  = 1.0   # 중심으로부터 ±1m
        self.cluster_density = 10    # 10×10 격자

        self.get_logger().info('SELD Obstacle Node가 시작되었습니다.')

    def seld_callback(self, msg: SeldTimedResult):
        """
        SeldTimedResult 메시지의 l_r 필드를 보고,
        좌/우에 해당하는 장애물을 생성해서 퍼블리시합니다.
        """
        # 1) msg.l_r은 uint8[] 타입 (예: [0,0], [0,1], [1,0], [1,1])
        #    left_flag  = msg.l_r[0], right_flag = msg.l_r[1]
        if len(msg.l_r) != 2:
            self.get_logger().warn(f"예상치 못한 l_r 크기: {len(msg.l_r)}. 이 메시지는 무시합니다.")
            return

        left_flag  = bool(msg.l_r[0])  # 0->False, 1->True
        right_flag = bool(msg.l_r[1])

        # 2) l_r 값에 따라 장애물 위치 결정
        centers = []
        if left_flag:
            centers.append(self.LEFT_POS)
        if right_flag:
            centers.append(self.RIGHT_POS)

        # 3) 만약 두 플래그가 모두 False라면 장애물 없음
        if not centers:
            # (원한다면, 장애물이 없을 때 이전 장애물을 지우거나 특별히 처리)
            self.get_logger().info('l_r=[0,0] → 장애물 없음')
            return

        # 4) 각 중심점마다 PointCloud2 생성/퍼블리시
        for center in centers:
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

            header = Header()
            header.stamp    = self.get_clock().now().to_msg()
            header.frame_id = 'map'  # RViz에서 볼 때 고정 프레임 이름과 일치시키세요

            cloud = create_cloud_xyz32(header, pts)
            self.pub.publish(cloud)

        side_str = []
        if left_flag:
            side_str.append('LEFT')
        if right_flag:
            side_str.append('RIGHT')
        self.get_logger().info(f"l_r={list(msg.l_r)} → {' & '.join(side_str)} 장애물 퍼블리시 완료")

def main():
    rclpy.init()
    node = SeldObstacleNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()