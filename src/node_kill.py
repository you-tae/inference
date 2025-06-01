#!/usr/bin/env python3
import subprocess
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped

class NodeKiller(Node):
    def __init__(self):
        super().__init__('inference_killer')
        self.threshold1 = 0.5
        self.threshold2 = 1.0
        self.killed = False
        self.launched = False  # 새로 띄울 노드 실행 플래그

        self.get_logger().info('▶ Inference_Killer 노드 시작: /amcl_pose 감시')
        self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.pose_callback,
            10
        )

    def pose_callback(self, msg):
        if self.killed and self.launched:
            return

        x = msg.pose.pose.position.x
        self.get_logger().info(f'[DEBUG] x={x:.3f}')
        if not self.killed and (x > self.threshold2):
            self.get_logger().info(f'x={x:.3f} > {self.threshold2} → inference_node 종료 시도')
            subprocess.run(['pkill', '-f', 'inference_node'])
            self.killed = True
            self.get_logger().info('✔ inference_node 프로세스를 종료했습니다.')

        if not self.launched and (x > self.threshold1):
            self.get_logger().info('▶ tf_buffer_node 프로세스 실행 시도')
            # ros2 run 방식 예시
            subprocess.Popen([
                'ros2', 'run', 'inference', 'tf_buffer_node.py'
            ])
            self.launched = True
            self.get_logger().info('✔ tf_buffer_node 프로세스를 백그라운드로 실행했습니다.')

def main(args=None):
    rclpy.init(args=args)
    node = NodeKiller()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__=='__main__':
    main()
