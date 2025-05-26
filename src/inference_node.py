#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import torch
import pickle
import os
from utils import get_multiaccdoa_labels, get_output_dict_format_multi_accdoa
from inference.msg import SeldTimedResult  # 커스텀 메시지
import numpy as np
from main_model import main_model
import time

class InferenceNode(Node):
    def __init__(self):
        super().__init__('inference_node')

        # 장치 설정 (GPU 또는 CPU)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 모델 로드
        # model_dir = "/home/doran/seld/src/inference/model_dir" 
        model_dir = "/home/jetson/seld/src/inference/model_dir"
        params_file = os.path.join(model_dir, 'config.pkl')
        model_checkpoint_file = os.path.join(model_dir, 'best_model.pth')

        with open(params_file, 'rb') as f:
            params = pickle.load(f)

        # 모델 로드 (weights_only=False)
        self.base_model = main_model(params).to(self.device)
        model_ckpt = torch.load(model_checkpoint_file, map_location=self.device, weights_only=False)
        self.base_model.load_state_dict(model_ckpt['seld_model'], strict=False)
        self.base_model.eval()  # 평가 모드로 설정

        self.get_logger().info("Model loaded successfully.")

        # 데이터 수신: 'audio_buffer_stereo' 토픽 구독
        self.subscription = self.create_subscription(
            Float32MultiArray,
            'mel_spectogram',
            self.audio_buffer_callback,
            10)
        
        # 결과 퍼블리시: SeldTimedResult 메시지
        self.pub = self.create_publisher(SeldTimedResult, 'seld_timed_result', 10)

        self.timer = self.create_timer(0.1, self.timer_callback)  # 10Hz로 타이머 설정 
        
        self.mel_data = None
        self.last_inference_time = time.time()
        self.total_inferences = 0
        self.total_time = 0
        
        self.get_logger().info("InferenceNode started.")

    def timer_callback(self):
        if self.mel_data is not None:
            self.perform_inference(self.mel_data)
    
    def perform_inference(self, mel_data):
        mel_tensor = torch.tensor(mel_data).reshape(2, 251, 64).to(self.device)
        
        with torch.no_grad():
            logits = self.base_model(mel_tensor.unsqueeze(0))

        # 디코딩
        modatlity = 'audio'

        (sed0, dummy_src_id0, doa0, dist0, on_screen0,
         sed1, dummy_src_id1, doa1, dist1, on_screen1,
         sed2, dummy_src_id2, doa2, dist2, on_screen2) = get_multiaccdoa_labels(logits, 13, 'audio')

        for i in range(sed0.size(0)):
            sed0_i, dummy_src_id0_i, doa0_i, dist0_i, on_screen0_i = sed0[i].cpu().numpy(), dummy_src_id0[i].cpu().numpy(), doa0[i].cpu().numpy(), dist0[i].cpu().numpy(), on_screen0[i].cpu().numpy()
            sed1_i, dummy_src_id1_i, doa1_i, dist1_i, on_screen1_i = sed1[i].cpu().numpy(), dummy_src_id1[i].cpu().numpy(), doa1[i].cpu().numpy(), dist1[i].cpu().numpy(), on_screen1[i].cpu().numpy()
            sed2_i, dummy_src_id2_i, doa2_i, dist2_i, on_screen2_i = sed2[i].cpu().numpy(), dummy_src_id2[i].cpu().numpy(), doa2[i].cpu().numpy(), dist2[i].cpu().numpy(), on_screen2[i].cpu().numpy()

            events = get_output_dict_format_multi_accdoa(sed0_i, dummy_src_id0_i, doa0_i, dist0_i, on_screen0_i,
                                                         sed1_i, dummy_src_id1_i, doa1_i, dist1_i, on_screen1_i, 
                                                         sed2_i, dummy_src_id2_i, doa2_i, dist2_i, on_screen2_i, 50, 13, True)

        # 결과를 딕셔너리로 변환
        now = self.get_clock().now().nanoseconds * 1e-9
        
        # 메시지 생성 및 퍼블리시
        msg = SeldTimedResult()
        msg.header.stamp = self.get_clock().now().to_msg()

        # 이벤트 처리
        for f_idx, ev_list in events.items():
            t_event = now - 5.0 + f_idx * 0.1
            for (cls, src, az, d, on) in ev_list:
                if cls == 0 or cls == 1:
                    msg.times.append(t_event)
                    msg.classes.append(cls)
                    msg.azimuth.append(az)
                    msg.distance.append(d)
                    msg.on_screen.append(bool(on))
                            # # 로그: 퍼블리시한 메시지 확인
                    self.get_logger().info("Inference result published.")
        
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
