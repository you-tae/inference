#!/usr/bin/env python3
import math, struct, numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Header
from sklearn.cluster import DBSCAN
import sensor_msgs_py.point_cloud2 as pc2
import tf2_ros

# ─── 간단한 칼만 필터 구현 ─────────────────────────────────────
class KalmanTracker:
    def __init__(self, x0, y0, dt):
        # 상태벡터 [x, y, vx, vy]
        self.x = np.array([x0, y0, 0.0, 0.0], dtype=float)
        # 상태 공분산
        self.P = np.eye(4) * 1.0
        # 상태전이행렬
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=float)
        # 관측행렬: 위치만 관측
        self.H = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
        # 프로세스·관측 잡음
        self.Q = np.eye(4) * 0.01
        self.R = np.eye(2) * 0.1
        self.age = 0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1

    def update(self, zx, zy):
        z = np.array([zx, zy], dtype=float)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.age = 0

    def state(self):
        return self.x.copy()

# ─── Dynamic Costmap Node ────────────────────────────────────────
class DynamicCostmapNode(Node):
    def __init__(self):
        super().__init__('dynamic_costmap_node')
        # DBSCAN 파라미터
        self.declare_parameter('dbscan_eps', 0.5)
        self.declare_parameter('dbscan_min_samples', 5)
        # 칼만 dt
        self.declare_parameter('track_dt', 0.1)
        # 예측 궤적 길이 (초)
        self.declare_parameter('predict_horizon', 1.0)
        # 예측 점 개수
        self.declare_parameter('predict_steps', 10)

        self.eps = self.get_parameter('dbscan_eps').value
        self.min_pts = self.get_parameter('dbscan_min_samples').value
        self.dt = self.get_parameter('track_dt').value
        self.horizon = self.get_parameter('predict_horizon').value
        self.steps = self.get_parameter('predict_steps').value

        # 트래커 관리
        self.trackers = {}
        self.next_id = 0

        # TF
        self.tfbuf = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.tfbuf, self)

        # 구독: 확률 맵핑된 포인트클라우드
        self.sub = self.create_subscription(
            PointCloud2, 'seld_prob_cloud', self.cloud_cb, 10)
        # 퍼블리시: 예측 장애물 포인트클라우드
        self.pub = self.create_publisher(
            PointCloud2, 'predicted_obstacles', 10)

        # 주기: dt 간격으로 예측 퍼블리시
        self.create_timer(self.dt, self.publish_predictions)
        self.get_logger().info('DynamicCostmapNode started.')

    def cloud_cb(self, cloud: PointCloud2):
        # 1) 포인트클라우드 → numpy (N×3)
        pts = np.array([
            (x,y,z) for x,y,z in pc2.read_points(
                cloud, field_names=('x','y','z'), skip_nans=True)
        ])
        if pts.shape[0] == 0:
            return

        # 2) DBSCAN 군집화 → 각 클러스터 중심
        labels = DBSCAN(eps=self.eps, min_samples=self.min_pts).fit_predict(pts)
        unique_labels = set(labels) - {-1}
        centers = [pts[labels==lab].mean(axis=0) for lab in unique_labels]

        used = set()
        # 3) 클러스터 중심 ↔ existing tracker 매칭
        for cx, cy, cz in centers:
            # 최장치 tracker 찾기
            dists = {tid: np.linalg.norm(tr.x[:2] - np.array([cx,cy]))
                     for tid,tr in self.trackers.items()}
            if dists and min(dists.values()) < self.eps:
                tid = min(dists, key=dists.get)
                self.trackers[tid].update(cx, cy)
                used.add(tid)
            else:
                # 신규 tracker 생성
                tid = self.next_id
                self.trackers[tid] = KalmanTracker(cx, cy, self.dt)
                self.next_id += 1
                used.add(tid)

        # 4) 사용 안 된 tracker age 증가, 일정 초과 시 삭제
        for tid, tr in list(self.trackers.items()):
            if tid not in used:
                tr.age += 1
            if tr.age * self.dt > 2.0:  # 2초 동안 미갱신 시 제거
                del self.trackers[tid]

    def publish_predictions(self):
        # 5) tracker 별 예측 궤적 생성
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'map'

        points = []
        for tr in self.trackers.values():
            # 현 상태 시간 동기화
            state = tr.state()           # [x, y, vx, vy]
            x0, y0, vx, vy = state
            # 미래 예측
            for i in range(1, self.steps+1):
                t = (i/self.steps) * self.horizon
                # 예측: x0+vx*t, y0+vy*t
                xp = x0 + vx * t
                yp = y0 + vy * t
                # map 좌표계로 변환
                pt = PointStamped()
                pt.header.frame_id = 'base_link'
                pt.header.stamp = header.stamp
                pt.point.x, pt.point.y, pt.point.z = xp, yp, 0.0
                try:
                    pt_map = self.tfbuf.transform(pt, 'map',
                        timeout=rclpy.duration.Duration(seconds=0.1))
                except Exception:
                    continue
                points.append((pt_map.point.x,
                               pt_map.point.y,
                               pt_map.point.z,
                               1.0))  # intensity

        # 6) PointCloud2 생성·퍼블리시
        pc2_msg = self.create_pointcloud2(header, points)
        self.pub.publish(pc2_msg)

    def create_pointcloud2(self, header, points):
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('intensity', 12, PointField.FLOAT32, 1),
        ]
        data = b''.join(struct.pack('ffff', x, y, z, i) 
                        for (x, y, z, i) in points)
        return PointCloud2(
            header=header,
            height=1, width=len(points),
            is_dense=False, is_bigendian=False,
            fields=fields,
            point_step=16, row_step=16*len(points),
            data=data
        )

def main():
    rclpy.init()
    node = DynamicCostmapNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__=='__main__':
    main()
