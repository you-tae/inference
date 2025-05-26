#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import socket
import subprocess
import numpy as np
import threading

from utils import extract_log_mel_spectrogram  # 멜 스펙트로그램 변환 함수 임포트
from parameters import params  # 파라미터 임포트

class AudioBufferNode(Node):
    def __init__(self):
        super().__init__('audio_buffer_node')
        
        # 파라미터 선언
        self.ws = 5  # 슬라이딩 윈도우 크기 (초)
        self.sr = params['sampling_rate']
        self.hop_length = int(self.sr * params['hop_length_s'])
        self.win_length = 2 * self.hop_length
        self.n_fft = 2 ** (self.win_length - 1).bit_length()
        self.nb_mels = params['nb_mels']
           
        # 5초 슬라이딩 윈도우 크기
        self.window_samps = int(self.sr * self.ws)
        self.buf_l = np.zeros(self.window_samps, dtype=np.int16)  # 왼쪽 채널 버퍼
        self.buf_r = np.zeros(self.window_samps, dtype=np.int16)  # 오른쪽 채널 버퍼

        # 하나의 토픽에 퍼블리시
        self.pub = self.create_publisher(Float32MultiArray, 'mel_spectogram', 10)

        # ADB reverse 자동 실행
        self.setup_adb_reverse()

        # 서버 소켓 설정
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('0.0.0.0', 5005))
        self.server_socket.listen(1)
        self.conn, _ = self.server_socket.accept()
        
        # 데이터 수신 스레드 시작
        threading.Thread(target=self.recv_loop, daemon=True).start()
        self.get_logger().info('AudioBufferNode started.')

    def setup_adb_reverse(self):
        """ADB reverse를 자동 실행하는 함수"""
        try:
            self.get_logger().info("📡 Setting up ADB reverse...")
            subprocess.run(['adb', 'reverse', 'tcp:5005', 'tcp:5005'], check=True)
            self.get_logger().info("✅ ADB reverse successful.")
        except subprocess.CalledProcessError as e:
            self.get_logger().error(f"❗ ADB reverse failed: {e}")
            rclpy.shutdown()

    def recv_loop(self):
        """TCP 소켓을 통해 데이터를 받는 함수"""
        while rclpy.ok():
            data = self.conn.recv(4096)
            if not data:
                break

            # int16 샘플 분리
            samples = np.frombuffer(data, dtype=np.int16)
            if samples.size % 2:
                samples = samples[:-1]
            l, r = samples[::2], samples[1::2]

            # 좌우 채널 바꾸기 (왼쪽 채널과 오른쪽 채널을 교환)
            self.buf_l = np.concatenate((self.buf_l, l))[-self.window_samps:]  # 왼쪽 채널 버퍼 갱신
            self.buf_r = np.concatenate((self.buf_r, r))[-self.window_samps:]  # 오른쪽 채널 버퍼 갱신

            # 16비트 정수를 32비트 부동소수점으로 변환
            buf_l_float = self.buf_l.astype(np.float32)
            buf_r_float = self.buf_r.astype(np.float32)

            combined_buf = np.stack([buf_l_float, buf_r_float], axis=0)  # 좌우 채널 결합
    
            # # 멜 스펙트로그램 변환 (여기서는 주석 처리)
            log_mel = extract_log_mel_spectrogram(combined_buf, self.sr, self.n_fft,
                                                  self.hop_length, self.win_length, self.nb_mels)
              
            # # 디버깅용으로 멜 스펙트로그램 크기 출력
            # self.get_logger().info(f"Stereo Mel Spectrogram shape: {log_mel.shape}")

            # 멜 스펙트로그램을 퍼블리시 (32비트 부동소수점으로 변환 후)
            msg = Float32MultiArray()
            msg.data = log_mel.astype(np.float32).ravel().tolist()  # 3D 배열을 1D 리스트로 변환하여 보내기
            self.pub.publish(msg)

        self.get_logger().info('AudioBufferNode terminating.')
        self.conn.close()

def main():
    rclpy.init()
    node = AudioBufferNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
