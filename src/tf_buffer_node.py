#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from inference.msg import SeldTimedResult
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32
from std_msgs.msg import Header

import numpy as np
import random  # Python의 랜덤 모듈 :contentReference[oaicite:4]{index=4}

class SeldObstacleNode(Node):
    def __init__(self):
        super().__init__('seld_obstacle_node')

        # 1) /seld_timed_result 구독자 설정
        self.sub = self.create_subscription(
            SeldTimedResult,
            '/seld_timed_result',
            self.seld_callback,
            10)  # 큐 사이즈 = 10

        # 2) /virtual_obstacles 퍼블리셔 설정 (PointCloud2 타입)
        self.pub = self.create_publisher(PointCloud2, '/virtual_obstacles', 10)

        # 3) 왼쪽/오른쪽 장애물 좌표 후보 리스트 정의
        #    필요에 따라 이 리스트에 더 많은 후보를 추가할 수 있음
        self.LEFT_CANDIDATES = [
            np.array([3.0,  2.2, 0.0]),
            np.array([3.2,  2.5, 0.0]),
            np.array([2.8,  1.9, 0.0])
        ]
        self.RIGHT_CANDIDATES = [
            np.array([3.0, -2.2, 0.0]),
            np.array([3.2, -2.5, 0.0]),
            np.array([2.8, -1.9, 0.0])
        ]

        # 4) 클러스터 생성 파라미터
        self.cluster_radius  = 1.0   # 중심으로부터 ±1m 범위
        self.cluster_density = 10    # 10×10 격자 생성

        # 5) 마지막으로 받은 l_r ([l, r])를 저장
        #    초기에 [0, 0]으로 두어 “양쪽에 소리 없음” 상태로 설정
        self.last_l_r = [0, 0]

        # 6) 타이머 설정: 1초마다 timer_callback 호출 :contentReference[oaicite:5]{index=5}
        publish_interval = 1.0  # 단위: 초
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

        # 3) 즉시 장애물 퍼블리시 (랜덤 후보 중 선택)
        #    centers 리스트를 받아서 publish_obstacles를 호출
        centers = self._choose_random_centers(self.last_l_r)
        if centers:
            self.publish_obstacles(centers)

    def timer_callback(self):
        """
        1초마다 호출됨.
        만약 last_l_r이 [0,0]이라면 아무 것도 하지 않음.
        그렇지 않으면 last_l_r 기준으로 랜덤 좌표를 뽑아 장애물 퍼블리시.
        """
        # 1) 마지막에 받은 l_r이 [0,0] 인 상태라면 장애물 없음
        if self.last_l_r == [0, 0]:
            # (optional) “장애물 없음” 로그를 주기적으로 남기고 싶다면 여기에 추가
            return

        # 2) last_l_r 값에 따라 랜덤 후보 좌표 리스트 생성
        centers = self._choose_random_centers(self.last_l_r)

        # 3) 후보가 비어있지 않으면 장애물 퍼블리시
        if centers:
            self.publish_obstacles(centers)

    def _choose_random_centers(self, lr_list):
        """
        lr_list: [l_flag, r_flag], 각 값은 0 또는 1
        왼쪽이 활성화(l_flag=1)되었으면 LEFT_CANDIDATES에서 랜덤 한 개 선택
        오른쪽이 활성화(r_flag=1)되었으면 RIGHT_CANDIDATES에서 랜덤 한 개 선택
        """
        left_flag, right_flag = lr_list

        centers = []
        # ★ 랜덤 후보 선택: random.choice(리스트) 사용 :contentReference[oaicite:6]{index=6}
        if left_flag:
            center_left = random.choice(self.LEFT_CANDIDATES)
            centers.append(center_left)
        if right_flag:
            center_right = random.choice(self.RIGHT_CANDIDATES)
            centers.append(center_right)

        return centers

    def publish_obstacles(self, centers):
        """
        centers: [np.array([x, y, z]), ...] 형태(1개 또는 2개 좌표)
        각 중심점에 대해 PointCloud2 클러스터를 생성/퍼블리시
        """
        for center in centers:
            pts = []
            # ±cluster_radius 범위 안에서 cluster_density x cluster_density 점 생성
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
            header.frame_id = 'map'  # RViz Fixed Frame과 일치시킬 것

            cloud = create_cloud_xyz32(header, pts)
            self.pub.publish(cloud)

        # 로그 출력
        side_str = []
        if centers and np.array_equal(centers[0], centers[0] if len(centers)==1 else centers[0]) and list(self.last_l_r)[0] == 1:
            # 단일 중심일 때 첫 요소로 왼쪽인지 오른쪽인지 구분 가능
            l_flag, r_flag = self.last_l_r
            if l_flag:
                side_str.append('LEFT')
            if r_flag:
                side_str.append('RIGHT')
        else:
            # 복수 센터일 때 [1,1] 상태
            if self.last_l_r[0] == 1:
                side_str.append('LEFT')
            if self.last_l_r[1] == 1:
                side_str.append('RIGHT')

        self.get_logger().info(f"타이머/콜백 → l_r={self.last_l_r}, 위치={side_str} 장애물 퍼블리시 완료")

def main():
    rclpy.init()
    node = SeldObstacleNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()