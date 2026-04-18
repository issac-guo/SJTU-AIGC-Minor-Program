"""
Demo1: 手势控制音量
功能：通过食指和拇指的距离控制电脑音量
技术栈：MediaPipe + OpenCV + pycaw（Windows音量控制）

依赖安装（推荐版本）：
 opencv-python numpy
pip install pycaw==20230407  # Windows音量控制（可选）
pip install comtypes==1.2.0  # pycaw依赖（可选）

注意：
- mediapipe 0.10.x 系列与当前代码兼容性最好
- pycaw 仅在Windows系统上可用，Mac/Linux会自动跳过音量控制功能
- 如果遇到mediapipe版本问题，可尝试：pip install mediapipe --upgrade
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import subprocess
from typing import Tuple

# 尝试导入pycaw进行音量控制，如果失败则使用模拟模式
try:
    from pycaw.pycaw import AudioUtilities
    VOLUME_CONTROL_AVAILABLE = True
except ImportError:
    VOLUME_CONTROL_AVAILABLE = False
    print("警告：pycaw未安装，将使用模拟音量显示模式")
    print("如需真实音量控制，请运行: pip install pycaw comtypes")

class HandVolumeController:
    def __init__(self):
        # 初始化MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # 音量控制初始化
        self.volume = None
        self.volume_range = (0, 100)
        
        if VOLUME_CONTROL_AVAILABLE:
            try:
                # 新版 pycaw API
                devices = AudioUtilities.GetSpeakers()
                self.volume = devices.EndpointVolume
                vol_range = self.volume.GetVolumeRange()
                self.volume_range = (vol_range[0], vol_range[1])
                print(f"音量范围: {self.volume_range}")
            except Exception as e:
                print(f"音量控制初始化失败: {e}")
                self.volume = None
        
        # 距离范围映射（像素距离 -> 音量百分比）
        self.min_distance = 20   # 最近距离（对应0%音量）
        self.max_distance = 200  # 最远距离（对应100%音量）
        
    def calculate_distance(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
        """计算两点之间的欧氏距离"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def map_distance_to_volume(self, distance: float) -> int:
        """将手指距离映射到音量百分比"""
        # 限制在有效范围内
        distance = max(self.min_distance, min(distance, self.max_distance))
        # 线性映射
        volume_percent = int((distance - self.min_distance) / 
                            (self.max_distance - self.min_distance) * 100)
        return volume_percent
    
    def set_system_volume(self, volume_percent: int):
        """设置系统音量"""
        if self.volume:
            try:
                # Windows音量控制 (使用pycaw)
                min_vol, max_vol = self.volume_range
                target_vol = min_vol + (max_vol - min_vol) * volume_percent / 100
                self.volume.SetMasterVolumeLevel(target_vol, None)
                return True
            except Exception as e:
                print(f"设置音量失败: {e}")
                return False
        return False
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, dict]:
        """处理单帧图像"""
        h, w, _ = frame.shape
        results_info = {
            "hand_detected": False,
            "distance": 0,
            "volume": 0
        }
        
        # 转换BGR到RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            results_info["hand_detected"] = True
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # 绘制手部关键点
            self.mp_draw.draw_landmarks(
                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
            )
            
            # 获取食指指尖(8)和拇指指尖(4)的坐标
            thumb_tip = hand_landmarks.landmark[4]
            index_tip = hand_landmarks.landmark[8]
            
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
            
            # 绘制连线
            cv2.line(frame, (thumb_x, thumb_y), (index_x, index_y), (0, 255, 255), 3)
            
            # 绘制关键点
            cv2.circle(frame, (thumb_x, thumb_y), 8, (0, 0, 255), -1)
            cv2.circle(frame, (index_x, index_y), 8, (0, 0, 255), -1)
            
            # 计算距离
            distance = self.calculate_distance((thumb_x, thumb_y), (index_x, index_y))
            results_info["distance"] = int(distance)
            
            # 映射到音量
            volume = self.map_distance_to_volume(distance)
            results_info["volume"] = volume
            
            # 设置系统音量
            self.set_system_volume(volume)
            
            # 显示信息
            mid_x = (thumb_x + index_x) // 2
            mid_y = (thumb_y + index_y) // 2
            cv2.putText(frame, f"Distance: {int(distance)}px", (mid_x, mid_y - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Volume: {volume}%", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            
            # 绘制音量条
            bar_x, bar_y = 10, 80
            bar_width, bar_height = 30, 200
            filled_height = int(bar_height * volume / 100)
            
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                         (255, 255, 255), 2)
            cv2.rectangle(frame, (bar_x, bar_y + bar_height - filled_height), 
                         (bar_x + bar_width, bar_y + bar_height), 
                         (0, 255, 0), -1)
        else:
            cv2.putText(frame, "No hand detected", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 显示操作说明
        cv2.putText(frame, "Pinch thumb and index to control volume", (10, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        return frame, results_info
    
    def run(self):
        """运行主循环"""
        # 尝试使用 DirectShow 后端
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            # 如果失败，尝试默认后端
            cap = cv2.VideoCapture(0)
        
        print("=" * 50)
        print("手势音量控制器已启动")
        print("操作：用食指和拇指做捏合手势")
        print("距离越近音量越小，越远音量越大")
        print("按 'Q' 退出")
        print("=" * 50)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("无法获取摄像头画面")
                break
            
            # 水平翻转（镜像效果）
            frame = cv2.flip(frame, 1)
            
            # 处理帧
            processed_frame, info = self.process_frame(frame)
            
            # 显示结果
            cv2.imshow("Hand Volume Controller", processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("程序已退出")

if __name__ == "__main__":
    controller = HandVolumeController()
    controller.run()
