#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
import numpy as np
import sys

class AudioVisualizationNode(Node):
    def __init__(self):
        super().__init__('audio_visualization_node')

        # ROS2 로그 레벨 설정 (DEBUG로 설정하여 모든 로그 출력)
        self.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)

        # 왼쪽 채널과 오른쪽 채널 데이터 버퍼
        self.buf_l = np.zeros(24000, dtype=np.float32)  # 1초 데이터 버퍼
        self.buf_r = np.zeros(24000, dtype=np.float32)

        # 왼쪽과 오른쪽 채널의 토픽을 각각 구독 (큐 크기 100으로 늘림)
        self.create_subscription(Float32MultiArray, 'audio_buffer_left', self.audio_buffer_left_callback, 100)
        self.create_subscription(Float32MultiArray, 'audio_buffer_right', self.audio_buffer_right_callback, 100)

        # pyqtgraph 실시간 창 구성
        self.app = QtWidgets.QApplication([])  # QApplication 인스턴스 한 번만 생성
        self.win = pg.GraphicsLayoutWidget(title="Real-time MIC Monitor (16bit)")
        self.win.show()
        self.win.resize(1000, 600)

        # 왼쪽 채널 시각화
        self.plot_l = self.win.addPlot(title="Left Channel (Time Domain)")
        self.curve_l = self.plot_l.plot(pen='c')
        self.plot_l.setYRange(-32000, 32000)  # 16bit 범위 고정
        self.plot_l.setXRange(0, 1, padding=0)  # X축 설정 (시간 범위 설정)

        # 오른쪽 채널 시각화
        self.win.nextRow()
        self.plot_r = self.win.addPlot(title="Right Channel (Time Domain)")
        self.curve_r = self.plot_r.plot(pen='m')
        self.plot_r.setYRange(-32000, 32000)
        self.plot_r.setXRange(0, 1, padding=0)  # X축 설정 (시간 범위 설정)

        # 시간 데이터 (1초를 24000개 샘플로 나눈 값)
        self.t_data = np.linspace(0, 1, 24000)

        # 타이머 설정
        self.timer = pg.QtCore.QTimer()  # 타이머 생성
        self.timer.timeout.connect(self.update_plot)  # 타이머 이벤트에 시각화 함수 연결
        self.timer.start(30)  # 30ms마다 시각화 업데이트 (약 33fps)

    def audio_buffer_left_callback(self, msg):
        # 수신된 왼쪽 채널 데이터
        self.buf_l = np.array(msg.data, dtype=np.float32)  # 16비트 데이터를 그대로 받기
        self.get_logger().info(f"Left Channel Data (first 10): {self.buf_l[:10]}")  # 첫 10개 데이터 로그

    def audio_buffer_right_callback(self, msg):
        # 수신된 오른쪽 채널 데이터
        self.buf_r = np.array(msg.data, dtype=np.float32)  # 16비트 데이터를 그대로 받기
        self.get_logger().info(f"Right Channel Data (first 10): {self.buf_r[:10]}")  # 첫 10개 데이터 로그

    def update_plot(self):
        # 시각화 업데이트 (타이머가 30ms마다 호출)
        self.curve_l.setData(self.t_data, self.buf_l)
        self.curve_r.setData(self.t_data, self.buf_r)

    def run(self):
        # pyqtgraph 창을 실행하고 ROS2 spin
        QtWidgets.QApplication.instance().exec_()

def main():
    rclpy.init()
    node = AudioVisualizationNode()
    rclpy.spin(node)  # rclpy.spin()을 통해 콜백을 지속적으로 실행
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
