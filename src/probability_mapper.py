#!/usr/bin/env python3
import math, rclpy
import numpy as np
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from geometry_msgs.msg import PointStamped
from collections import defaultdict
import tf2_ros, struct

class ProbabilityMapper(Node):
    def __init__(self):
        super().__init__('probability_mapper')
        # 파라미터
        self.declare_parameter('prob_threshold', 0.6)
        self.declare_parameter('angle_acc', [0.8]*13)
        self.declare_parameter('dist_acc',  [0.9]*13)
        self.declare_parameter('sigma_angle', [10.0]*13)  # deg 단위
        self.declare_parameter('sigma_dist',  [0.2]*13)   # m 단위

        # 맵 구독
        self.map = None
        self.create_subscription(OccupancyGrid, 'map', self.cb_map, 1)

        # Aggregator에서 만든 이벤트 테이블을 외부에서 가져왔다고 가정
        self.events = defaultdict(list)

        # TF
        self.tfbuf = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.tfbuf, self)

        # PointCloud2 퍼블리셔
        self.pub_pc = self.create_publisher(PointCloud2, 'seld_prob_cloud', 1)

        # 1 Hz로 확률 업데이트
        self.create_timer(1.0, self.update_probability_map)

    def cb_map(self, msg: OccupancyGrid):
        self.map = msg

    def update_probability_map(self):
        if self.map is None:
            return

        info = self.map.info
        W, H = info.width, info.height
        res = info.resolution
        ox, oy = info.origin.position.x, info.origin.position.y

        # 초기 확률 0.5
        P = np.full((H, W), 0.5, dtype=np.float32)

        acc_ang   = self.get_parameter('angle_acc').value
        acc_dst   = self.get_parameter('dist_acc').value
        sig_ang   = self.get_parameter('sigma_angle').value
        sig_dist  = self.get_parameter('sigma_dist').value
        threshold = self.get_parameter('prob_threshold').value

        # Monte Carlo 샘플 수
        N = 100

        for bin_t, ev_list in self.events.items():
            for ev in ev_list:
                cls, az, el, dist, on = (
                    ev['class'], ev['azimuth'], ev['elevation'],
                    ev['distance'], ev['on']
                )
                # 이벤트 신뢰도
                p_event = acc_ang[cls] * acc_dst[cls]

                # 샘플링
                thetas = np.random.normal(az, sig_ang[cls], N)
                rs     = np.random.normal(dist, sig_dist[cls], N)

                # 입자별 업데이트
                for theta, r in zip(thetas, rs):
                    x_loc = r * math.cos(math.radians(theta))
                    y_loc = r * math.sin(math.radians(theta))
                    pt = PointStamped()
                    pt.header.frame_id = 'base_link'
                    pt.header.stamp = self.get_clock().now().to_msg()
                    pt.point.x, pt.point.y, pt.point.z = x_loc, y_loc, 0.0
                    try:
                        pt_map = self.tfbuf.transform(pt, 'map',
                            timeout=rclpy.duration.Duration(seconds=0.1))
                    except:
                        continue

                    mx = int((pt_map.point.x - ox)/res)
                    my = int((pt_map.point.y - oy)/res)
                    if not (0 <= mx < W and 0 <= my < H):
                        continue
                    if self.map.data[my * W + mx] != 0:
                        continue

                    # 베이즈식 업데이트
                    P[my, mx] = 1 - (1 - P[my, mx]) * (1 - p_event)

        # P ≥ threshold 셀만 PointCloud2
        points = []
        for iy in range(H):
            for ix in range(W):
                prob = P[iy, ix]
                if prob >= threshold:
                    x = ox + (ix + 0.5) * res
                    y = oy + (iy + 0.5) * res
                    points.append((x, y, 0.0, prob))

        header = Header(stamp=self.get_clock().now().to_msg(), frame_id='map')
        self.pub_pc.publish(self.create_pointcloud2(header, points))

    def create_pointcloud2(self, header, points):
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('intensity', 12, PointField.FLOAT32, 1),
        ]
        data = b''.join(struct.pack('ffff', x, y, z, i) for (x, y, z, i) in points)
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
    node = ProbabilityMapper()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
