#!/usr/bin/python3
import math
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PointStamped

class HoughCornerNotifier(Node):
    def __init__(self):
        super().__init__('hough_corner_notifier')
        self.get_logger().info('Hough Corner Notifier node started.')

        self.declare_parameter('corner_thresh', 1.5)
        self.corner_thresh = self.get_parameter('corner_thresh').value
        self.detected = set()

        qos = QoSProfile(depth=1)
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        qos.reliability = ReliabilityPolicy.RELIABLE
        self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            qos_profile=qos
        )

        self.pub = self.create_publisher(PointStamped, 'corner_detected', 10)

    def map_callback(self, msg):
        self.get_logger().info('Map received, detecting corners...')
        w, h = msg.info.width, msg.info.height
        res = msg.info.resolution
        ox, oy = msg.info.origin.position.x, msg.info.origin.position.y

        self.get_logger().info(f'Map meta: width={w}, height={h}, resolution={res}, origin=({ox}, {oy})')

        # OccupancyGrid to binary image
        data = np.array(msg.data, dtype=np.int8).reshape((h, w))
        binary = np.uint8(data == 100) * 255

        # Morphological filtering to reduce noise
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Edge detection
        edges = cv2.Canny(binary, 20, 70)

        # Hough Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=250, minLineLength=10, maxLineGap=5)
        if lines is None:
            self.get_logger().warn('No lines detected.')
            return

        self.get_logger().info(f'✅ Raw lines detected: {len(lines)}')

        # Intersection calculation
        intersections = []
        for i in range(len(lines)):
            for j in range(i+1, len(lines)):
                l1 = lines[i][0]
                l2 = lines[j][0]
                angle = self.line_angle_diff(l1, l2)
                if not (math.radians(60) <= angle <= math.radians(120)):
                    continue
                pt = self.get_intersection(l1, l2)
                if pt is not None:
                    intersections.append(pt)

        self.get_logger().info(f'🔹 Intersections found: {len(intersections)}')

        # Merge close points
        merged = self.merge_close_points(intersections, threshold=0.3)
        self.get_logger().info(f'🔸 Merged corners: {len(merged)}')

        for px, py in merged:
            x = ox + px * res
            y = oy + (h - py) * res
            self.get_logger().info(f'📍 Corner at ({x:.3f}, {y:.3f})')
            msg_out = PointStamped()
            msg_out.header.frame_id = 'map'
            msg_out.point.x = x
            msg_out.point.y = y
            self.pub.publish(msg_out)

    def line_angle_diff(self, l1, l2):
        dx1, dy1 = l1[2] - l1[0], l1[3] - l1[1]
        dx2, dy2 = l2[2] - l2[0], l2[3] - l2[1]
        angle1 = math.atan2(dy1, dx1)
        angle2 = math.atan2(dy2, dx2)
        return abs(angle1 - angle2)

    def get_intersection(self, l1, l2):
        x1, y1, x2, y2 = l1
        x3, y3, x4, y4 = l2
        denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
        if denom == 0:
            return None
        px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
        py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
        return (px, py)

    def merge_close_points(self, points, threshold):
        merged = []
        for p in points:
            if all(math.hypot(p[0]-q[0], p[1]-q[1]) > threshold for q in merged):
                merged.append(p)
        return merged

def main():
    rclpy.init()
    node = HoughCornerNotifier()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()