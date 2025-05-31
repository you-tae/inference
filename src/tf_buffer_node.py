#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from inference.msg import SeldTimedResult
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32
from std_msgs.msg import Header

import numpy as np
import random
import math

# 1) 모양 상수 정의
SHAPE_SQUARE = "square"
SHAPE_CIRCLE = "circle"

class SeldObstacleNode(Node):
    def __init__(self):
        super().__init__('seld_obstacle_node')

        # 2) /seld_timed_result 구독자 설정
        self.sub = self.create_subscription(
            SeldTimedResult,
            '/seld_timed_result',
            self.seld_callback,
            10  # 큐 사이즈 = 10
        )

        # 3) /virtual_obstacles 퍼블리셔 설정 (PointCloud2 타입)
        self.pub = self.create_publisher(PointCloud2, '/virtual_obstacles', 10)

        # 4) 왼쪽/오른쪽 장애물 좌표 후보 리스트 정의
        #    필요에 따라 이 리스트에 더 많은 후보를 추가할 수 있음
        self.LEFT_CANDIDATES = [
            np.array([3.0,  2.2, 0.0]),
            np.array([3.2,  2.5, 0.0]),
            np.array([2.8,  1.9, 0.0]),
            np.array([3.1,  2.3, 0.0]),
            np.array([2.9,  2.1, 0.0]),
            np.array([3.0,  2.7, 0.0]),
            np.array([2.9,  3.0, 0.0])
        ]
        self.RIGHT_CANDIDATES = [
            np.array([3.0, -2.2, 0.0]),
            np.array([3.2, -2.5, 0.0]),
            np.array([2.8, -1.9, 0.0]),
            np.array([3.1, -2.3, 0.0]),
            np.array([2.9, -2.1, 0.0]),
            np.array([3.0, -2.7, 0.0]),
            np.array([2.9, -3.0, 0.0])
        ]

        # 5) 클러스터 생성 파라미터
        self.cluster_radius  = 1.0   # 중심으로부터 ±1m 범위
        self.cluster_density = 10    # 10×10 격자 생성

        # 6) 마지막으로 받은 l_r ([l, r])를 저장
        #    초기에 [0, 0]으로 두어 “양쪽에 소리 없음” 상태로 설정
        self.last_l_r = [0, 0]

        # 7) 타이머 설정: 0.5초마다 timer_callback 호출
        publish_interval = 0.5  # 단위: 초
        self.timer = self.create_timer(publish_interval, self.timer_callback)

        self.get_logger().info('SELD Obstacle Node가 초기화되었습니다.')

    def seld_callback(self, msg: SeldTimedResult):
        """
        /seld_timed_result 토픽으로부터 l_r을 수신하여,
        즉시 장애물을 퍼블리시하고 last_l_r를 갱신합니다.
        """
        # 1) msg.l_r은 uint8[] 타입 (크기가 2여야 함)
        if len(msg.l_r) != 2:
            self.get_logger().warn(f"예상치 못한 l_r 크기: {len(msg.l_r)}. 메시지를 무시합니다.")
            return

        # 2) 마지막 받은 l_r 저장
        self.last_l_r = [int(msg.l_r[0]), int(msg.l_r[1])]

        # 3) 즉시 장애물 퍼블리시 (랜덤 후보 중 선택된 centers에 대해)
        centers = self._choose_random_centers(self.last_l_r)
        if centers:
            for center in centers:
                self._choose_shape_and_publish(center)

    def timer_callback(self):
        """
        0.5초마다 호출됨.
        만약 last_l_r이 [0,0]이라면 아무 것도 하지 않음.
        그렇지 않으면 last_l_r 기준으로 랜덤 좌표를 뽑아 장애물을 퍼블리시.
        """
        # 1) 마지막에 받은 l_r이 [0,0] 인 상태라면 장애물 없음
        if self.last_l_r == [0, 0]:
            return

        # 2) last_l_r 값에 따라 랜덤 후보 좌표 리스트 생성
        centers = self._choose_random_centers(self.last_l_r)

        # 3) 후보가 비어 있지 않으면 장애물 퍼블리시
        if centers:
            for center in centers:
                self._choose_shape_and_publish(center)

    def _choose_random_centers(self, lr_list):
        """
        lr_list: [l_flag, r_flag], 각 값은 0 또는 1
        왼쪽이 활성화(l_flag=1)되었으면 LEFT_CANDIDATES에서 랜덤 한 개 선택
        오른쪽이 활성화(r_flag=1)되었으면 RIGHT_CANDIDATES에서 랜덤 한 개 선택
        """
        left_flag, right_flag = lr_list

        centers = []
        if left_flag:
            center_left = random.choice(self.LEFT_CANDIDATES)
            centers.append(center_left)
        if right_flag:
            center_right = random.choice(self.RIGHT_CANDIDATES)
            centers.append(center_right)

        return centers

    def _choose_shape_and_publish(self, center: np.ndarray):
        """
        center: np.array([cx, cy, cz])
        사각형 또는 원형 모양을 랜덤으로 선택하여 퍼블리시
        """
        # 1) 사각형 vs 원형 중 랜덤으로 선택
        shape = random.choice([SHAPE_SQUARE, SHAPE_CIRCLE])
        if shape == SHAPE_SQUARE:
            pts = self._generate_square_points(center)
        else:
            pts = self._generate_circle_points(center)

        # 2) PointCloud2 생성 및 퍼블리시
        self._publish_pointcloud(pts)

        # 3) 로깅: 모양, 중심좌표, 점 개수
        self.get_logger().info(
            f"퍼블리시: 모양={shape}, 중심={center.tolist()}, 점 개수={len(pts)}"
        )

    def _generate_square_points(self, center: np.ndarray):
        """
        center: np.array([cx, cy, cz])
        return: 사각형(격자) 형태의 pts 리스트
        """
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
        return pts

    def _generate_circle_points(self, center: np.ndarray):
        """
        center: np.array([cx, cy, cz])
        return: 원형(디스크) 형태의 pts 리스트
        """
        pts = []
        r = self.cluster_radius
        lin = np.linspace(-r, r, self.cluster_density)
        for dx in lin:
            for dy in lin:
                # 반경 내에 있으면 원 내부 점으로 간주
                if dx*dx + dy*dy <= r*r:
                    x = float(center[0] + dx)
                    y = float(center[1] + dy)
                    z = float(center[2])
                    pts.append([x, y, z])
        return pts

    def _publish_pointcloud(self, points):
        """
        points: [[x, y, z], ...] 리스트
        헤더 생성 → create_cloud_xyz32 → 퍼블리시
        """
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'map'

        cloud = create_cloud_xyz32(header, points)
        self.pub.publish(cloud)

def main():
    rclpy.init()
    node = SeldObstacleNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()