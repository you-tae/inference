#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import torch
import pickle
import os
from utils import decode_logits
from inference.msg import SeldTimedResult  # 커스텀 메시지
import numpy as np
from main_model import main_model
import time
from parameters import params  # data_generator.py와 동일한 파라미터 딕셔너리

class InferenceNode(Node):
    def __init__(self):
        super().__init__('inference_node')

        # ────────────────────────────────────────────
        # 1) 장치 설정 (GPU 또는 CPU)
        # ────────────────────────────────────────────
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # ────────────────────────────────────────────
        # 2) 모델 인스턴스 생성
        # ────────────────────────────────────────────
        self.base_model = main_model(params).to(self.device)

        # ────────────────────────────────────────────
        # 3) 체크포인트 파일 경로 설정 및 존재 여부 확인
        # ────────────────────────────────────────────
        model_dir = "/home/dbxotjs/seld/src/inference/model_dir"
        model_checkpoint_file = os.path.join(model_dir, 'best_model.pth')

        if not os.path.exists(model_checkpoint_file):
            self.get_logger().error(f"Model checkpoint file not found: {model_checkpoint_file}")
            raise FileNotFoundError(f"Model checkpoint file not found: {model_checkpoint_file}")
        else:
            self.get_logger().info(f"Loading model from: {model_checkpoint_file}")
            state_dict = torch.load(model_checkpoint_file, map_location=self.device)
            self.base_model.load_state_dict(state_dict, strict=False)
            self.get_logger().info("Model weights loaded successfully.")

        # ────────────────────────────────────────────
        # 4) 모델을 평가 모드로 변경
        # ────────────────────────────────────────────
        self.base_model.eval()
        self.get_logger().info("Base model set to eval mode.")

        # ────────────────────────────────────────────
        # 5) ROS 구독/퍼블리셔 및 타이머 설정
        # ────────────────────────────────────────────
        # 'mel_spectogram' 토픽 구독
        self.subscription = self.create_subscription(
            Float32MultiArray,
            'mel_spectogram',
            self.audio_buffer_callback,
            10)

        # 'seld_timed_result' 토픽 퍼블리셔
        self.pub = self.create_publisher(SeldTimedResult, 'seld_timed_result', 10)

        # 10Hz 타이머 설정
        self.timer = self.create_timer(0.1, self.timer_callback)

        # ────────────────────────────────────────────
        # 6) 내부 변수 초기화
        # ────────────────────────────────────────────
        self.mel_data = None
        self.last_inference_time = time.time()
        self.total_inferences = 0
        self.total_time = 0

        self.get_logger().info("InferenceNode started.")

    def timer_callback(self):
        if self.mel_data is not None:
            self.perform_inference(self.mel_data)
    
    def perform_inference(self, mel_data):
        mel_tensor = torch.tensor(mel_data).reshape(2, 94, 64).to(self.device)
        
        with torch.no_grad():
            logits = self.base_model(mel_tensor.unsqueeze(0))
        
        self.get_logger().info(f'Logits shape: {logits.shape}')
        self.get_logger().info(f"Logits: {logits}")  
        
        # 디코딩
        decoded_logits = decode_logits(logits)
                    
        # 메시지 생성 및 퍼블리시
        msg = SeldTimedResult()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.time = self.get_clock().now().nanoseconds * 1e-9  # 현재 시간 (초 단위)
        msg.l_r = decoded_logits
        
        self.pub.publish(msg)

        # # # 로그: 퍼블리시한 메시지 확인
        # self.get_logger().info("Inference result published.")
        
        self.last_inference_time = time.time()  # 마지막 추론 시간 업데이트
        
    def audio_buffer_callback(self, msg):
                
        data = np.array(msg.data, dtype=np.float32)
        self.mel_data = data

def main():
    rclpy.init()
    node = InferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
