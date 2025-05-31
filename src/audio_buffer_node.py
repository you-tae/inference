#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import socket
import subprocess
import numpy as np
import threading

from utils import extract_log_mel_spectrogram  # 멜 스펙트로그램 변환 함수
from parameters import params  # data_generator.py와 동일한 파라미터 딕셔너리

class AudioBufferNode(Node):
    def __init__(self):
        super().__init__('audio_buffer_node')
        
        # ────────────────────────────────────────────
        # 1) params에서 필요한 값 불러오기
        # ────────────────────────────────────────────
        # 슬라이딩 윈도우 길이 (초)
        self.ws = 1  # 예시: 1초짜리 윈도우

        # data_generator.py와 동일한 키 이름 사용
        self.sr = params['SAMPLE_RATE']      # 예: 24000, 16000 등
        self.n_fft = params['N_FFT']         # 예: 1024, 2048 등
        self.hop_length = params['HOP_LENGTH']   # 예: 256, 512 등
        self.win_length = params['WIN_LENGTH']   # 예: 512, 1024 등
        self.nb_mels = params['nb_mels']     # 멜 밴드 개수 (예: 64, 80 등)

        # ────────────────────────────────────────────
        # 2) 슬라이딩 윈도우 버퍼 크기 계산
        # ────────────────────────────────────────────
        # 윈도우(슬라이딩) 길이: self.ws (초) * self.sr (샘플/초)
        self.window_samps = int(self.sr * self.ws)
        # 채널별 int16 버퍼 초기화
        self.buf_l = np.zeros(self.window_samps, dtype=np.int16)
        self.buf_r = np.zeros(self.window_samps, dtype=np.int16)

        # ────────────────────────────────────────────
        # 3) ROS 퍼블리셔 설정
        # ────────────────────────────────────────────
        # Float32MultiArray 메시지로 mel-spectrogram을 퍼블리시
        self.pub = self.create_publisher(Float32MultiArray, 'mel_spectogram', 10)

        # ────────────────────────────────────────────
        # 4) ADB reverse 자동 실행 (Android 디바이스 연결용)
        # ────────────────────────────────────────────
        self.setup_adb_reverse()

        # ────────────────────────────────────────────
        # 5) 서버 소켓 설정 (TCP로 오디오 샘플 수신)
        # ────────────────────────────────────────────
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('0.0.0.0', 5005))
        self.server_socket.listen(1)
        self.conn, _ = self.server_socket.accept()

        # ────────────────────────────────────────────
        # 6) 수신 전용 스레드 시작
        # ────────────────────────────────────────────
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
            # 한 번에 최대 4096바이트만큼 읽음
            data = self.conn.recv(4096)
            if not data:
                break

            # ────────────────────────────────────────────
            # 1) 받은 바이너리를 int16 샘플로 변환
            # ────────────────────────────────────────────
            samples = np.frombuffer(data, dtype=np.int16)
            # 홀수개 샘플이 들어올 경우 마지막 하나 버림
            if samples.size % 2:
                samples = samples[:-1]
            # 스테레오: 짝수 인덱스→왼쪽, 홀수 인덱스→오른쪽
            l, r = samples[::2], samples[1::2]

            # ────────────────────────────────────────────
            # 2) 슬라이딩 윈도우 버퍼 갱신
            # ────────────────────────────────────────────
            # 이전 버퍼 + 새 샘플을 붙이고, 뒤에서부터 window_samps만큼 유지
            self.buf_l = np.concatenate((self.buf_l, l))[-self.window_samps:]
            self.buf_r = np.concatenate((self.buf_r, r))[-self.window_samps:]

            # ────────────────────────────────────────────
            # 3) int16 → float32로 변환
            # ────────────────────────────────────────────
            buf_l_float = self.buf_l.astype(np.float32)
            buf_r_float = self.buf_r.astype(np.float32)

            # 채널 축(0) 기준으로 스택하여 (2, window_samps) 모양 생성
            combined_buf = np.stack([buf_l_float, buf_r_float], axis=0)

            # ────────────────────────────────────────────
            # 4) 로그 멜 스펙트로그램 추출
            # ────────────────────────────────────────────
            log_mel = extract_log_mel_spectrogram(
                combined_buf,
                sr=self.sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                nb_mels=self.nb_mels
            )
            # log_mel의 shape 예시: (2, time_frames, nb_mels) 
            # -- 채널 축이 첫 번째이므로 2개 채널 분리된 결과

            # ────────────────────────────────────────────
            # 5) ROS 토픽으로 퍼블리시
            # ────────────────────────────────────────────
            msg = Float32MultiArray()
            # 3D 배열을 1D 리스트로 펼쳐서 전송
            msg.data = log_mel.astype(np.float32).ravel().tolist()
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