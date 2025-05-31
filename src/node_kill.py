#!/usr/bin/env python3
import subprocess
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped

class TeleopKiller(Node):
    def __init__(self):
        super().__init__('inference_killer')
        self.threshold = 0.5
        self.killed = False

        self.get_logger().info('▶ Inference_Killer 노드 시작: /amcl_pose 감시')
        self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.pose_callback,
            10
        )

    def pose_callback(self, msg):
        if self.killed:
            return

        x = msg.pose.pose.position.x
        self.get_logger().info(f'[DEBUG] y={x:.3f}')
        if x > self.threshold:
            self.get_logger().info(f'y={x:.3f} > {self.threshold} → inference_node 종료 시도')
            # pkill로 프로세스명 'teleop_keyboard'에 SIGTERM
            subprocess.run(['pkill', '-f', 'inference_node'])
            self.killed = True
            self.get_logger().info('✔ inference_node 프로세스를 종료했습니다.')

def main(args=None):
    rclpy.init(args=args)
    node = TeleopKiller()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__=='__main__':
    main()