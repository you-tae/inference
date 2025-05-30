#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import torch
import torch.nn as nn
import pickle
import os
from utils import get_multiaccdoa_labels, get_output_dict_format_multi_accdoa
from inference.msg import SeldTimedResult  # 커스텀 메시지
import numpy as np
from main_model import ResNetConformerSELDModel

import time

class InferenceNode(Node):
    def __init__(self):
        super().__init__('inference_node')

        # 장치 설정 (GPU 또는 CPU)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 모델 로드
        # model_dir = "/home/doran/seld/src/inference/model_dir" 
        model_dir = "/home/dbxotjs/seld/src/inference/model_dir"
        params_file = os.path.join(model_dir, 'config.pkl')
        model_checkpoint_file = os.path.join(model_dir, 'best_model.pth')

        with open(params_file, 'rb') as f:
            params = pickle.load(f)

        #print("Keys in config:", list(params.keys()))    

        # 모델 로드 (weights_only=False)
        self.base_model = ResNetConformerSELDModel().to(self.device)
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
    
    ###########################################################################################
    def extract_binary_sed_class1(self, sed):
        """
        Args:
            sed (Tensor): shape (50, 13), SED prediction scores
        Returns:
            Tensor: shape (50, 1), binary mask for class 1
        """
        class1 = sed[:, 1]
        binary_mask = (class1 > 0.5).float()
        return binary_mask.unsqueeze(-1)  # (50, 1)

    def extract_azimuth_class1(self, doa):
        """
        Args:
            doa (Tensor): shape (50, 39), DOA predictions (x1..x13, y1..y13, d1..d13)
        Returns:
            Tensor: shape (50, 1), azimuth in degrees for class 1
        """
        x = doa[:, 1]   # x2
        y = doa[:, 14]  # y2
        azi_rad = torch.atan2(y, x)
        azi_deg = azi_rad * 180 / np.pi
        return azi_deg.unsqueeze(-1)  # (50, 1)

    def process_model_outputs(self, logit):
        """
        Args:
            sed_logits (Tensor): shape (50, 13)
            doa_logits (Tensor): shape (50, 39)
        Returns:
            Tuple[Tensor, Tensor]: (binary_sed_class1, azimuth_deg_class1), both (50, 1)
        """
        sed_logits,doa_logits=logit
        sed_logits = sed_logits.squeeze(0)
        #self.get_logger().info(f"SED logits for class1:, {sed_logits[:,1].cpu().numpy()[:10]}")
        doa_logits = doa_logits.squeeze(0)

        sed = self.extract_binary_sed_class1(sed_logits)
        azimuth = self.extract_azimuth_class1(doa_logits)
        return sed, azimuth    
    ###########################################################################################

    def perform_inference(self, mel_data):
        # 1) 입력 데이터 → tensor → (2,251,64) → (1,2,251,64)
        mel_tensor = torch.tensor(mel_data, dtype=torch.float32).reshape(2, 251, 64).to(self.device)
        mel_tensor = mel_tensor.unsqueeze(0)

        # 2) 모델 추론 (grad off)
        with torch.no_grad():
            sed_mask, azimuth = self.process_model_outputs(self.base_model(mel_tensor))
            # sed_mask, azimuth: 각각 Tensor shape (50,1)

        # 3) (50,1) → (50,) 으로 차원 축소
        sed_vec = sed_mask.squeeze(-1)    # Tensor (50,)
        az_vec  = azimuth.squeeze(-1)     # Tensor (50,)

        # 4) NumPy로 옮긴 뒤 Python 리스트로 변환
        sed_arr = sed_vec.cpu().numpy()   # ndarray shape (50,)
        az_arr  = az_vec.cpu().numpy()    # ndarray shape (50,)

        # 5) 메시지 초기화
        msg = SeldTimedResult()
        msg.header.stamp = self.get_clock().now().to_msg()

        # 6) 프레임별 이벤트 append
        #    필요하다면 `if int(s)==1:` 으로 활성 이벤트만 필터링 가능
        now = self.get_clock().now().nanoseconds * 1e-9
        for i, (s, az) in enumerate(zip(sed_arr, az_arr)):
            t_event = now - 5.0 + i * 0.1
            msg.times.append(float(t_event))
            msg.classes.append(1)            # 0 또는 1    int(s)
            msg.azimuth.append(float(az))    # 방위각(deg)
            msg.distance.append(0.0)         # 모델에서 계산된다면 그 값으로 대체
            msg.on_screen.append(False)      # 마찬가지로 실제 플래그로 대체

        # 7) 퍼블리시
        self.pub.publish(msg)
        self.get_logger().info(f"Published {len(msg.times)} events.")
        self.last_inference_time = time.time()
        
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
