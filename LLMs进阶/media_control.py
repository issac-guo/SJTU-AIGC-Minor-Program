"""
Medical Image Gesture-Controlled Viewer
医学影像非接触交互原型

功能：
- 左手：食指水平移动切换切片
- 右手：食指拇指距离控制缩放
- 右手OK手势：添加标记点

作者：AI Assistant
"""

import cv2
import numpy as np
import mediapipe as mp
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Callable
import time


# ============================================================================
#                              配置参数
# ============================================================================

@dataclass
class ViewerConfig:
    """系统配置参数"""
    # === 手势阈值 ===
    slice_move_threshold: float = 0.06      # 切片切换移动阈值（归一化坐标）
    pinch_min_distance: float = 0.03        # 最小pinch距离（归一化）
    pinch_max_distance: float = 0.20        # 最大pinch距离（归一化）
    ok_gesture_threshold: float = 0.035     # OK手势判定阈值
    finger_straight_threshold: float = 0.02 # 手指伸直判定阈值

    # === Cooldown (秒) ===
    slice_cooldown: float = 0.35            # 切片切换冷却时间
    marker_cooldown: float = 1.2            # 标记添加冷却时间

    # === 缩放范围 ===
    zoom_min: float = 0.5
    zoom_max: float = 3.0
    zoom_default: float = 1.0

    # === 平滑参数 ===
    smoothing_alpha: float = 0.25           # 指数平滑系数 (越小越平滑)

    # === 显示参数 ===
    main_display_size: Tuple[int, int] = (800, 600)
    webcam_inset_size: Tuple[int, int] = (240, 180)
    notification_duration: float = 1.5      # 通知显示时长

    # === MediaPipe 参数 ===
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.6

    # === 模拟影像参数 ===
    num_dummy_slices: int = 50
    image_width: int = 512
    image_height: int = 512


# ============================================================================
#                              数据结构定义
# ============================================================================

class SystemState(Enum):
    """系统状态枚举"""
    IDLE = "Idle"
    NAVIGATE = "Navigate"
    ZOOM = "Zoom"
    ADD_MARKER = "Add Marker"


@dataclass
class HandData:
    """单只手的追踪数据"""
    detected: bool = False
    landmarks: Optional[np.ndarray] = None  # 21个关键点 (21, 3) - x, y, z
    index_tip: Tuple[float, float] = (0.5, 0.5)  # 食指尖归一化坐标
    thumb_tip: Tuple[float, float] = (0.5, 0.5)  # 拇指尖归一化坐标
    pinch_distance: float = 0.1              # 食指拇指距离（归一化）
    is_ok_gesture: bool = False
    confidence: float = 0.0
    # 平滑后的值
    smoothed_index_x: float = 0.5
    smoothed_pinch: float = 0.1


@dataclass
class Marker:
    """图像标记点"""
    x: float  # 归一化图像坐标 (0-1)
    y: float
    timestamp: float
    slice_index: int

    def to_image_coords(self, img_w: int, img_h: int) -> Tuple[int, int]:
        """转换为像素坐标"""
        return (int(self.x * img_w), int(self.y * img_h))


@dataclass
class ViewerState:
    """查看器全局状态"""
    # 影像状态
    current_slice: int = 0
    total_slices: int = 50
    zoom_level: float = 1.0

    # 系统状态
    system_state: SystemState = SystemState.IDLE
    left_hand: HandData = field(default_factory=HandData)
    right_hand: HandData = field(default_factory=HandData)

    # 标记数据: slice_index -> List[Marker]
    markers: Dict[int, List[Marker]] = field(default_factory=dict)

    # 时间控制
    last_slice_change_time: float = 0.0
    last_marker_time: float = 0.0

    # 通知
    notification: Optional[str] = None
    notification_expire_time: float = 0.0

    # 导航参考位置
    navigate_reference_x: float = 0.5


# ============================================================================
#                              工具模块
# ============================================================================

class ExponentialSmoother:
    """指数移动平均平滑器"""

    def __init__(self, alpha: float = 0.25):
        """
        Args:
            alpha: 平滑系数，越大响应越快，越小越平滑
        """
        self.alpha = alpha
        self.value: Optional[float] = None

    def update(self, new_value: float) -> float:
        """更新并返回平滑后的值"""
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value

    def reset(self, initial_value: Optional[float] = None):
        """重置平滑器"""
        self.value = initial_value


class DummyMedicalImageGenerator:
    """模拟医学影像生成器"""

    def __init__(self, config: ViewerConfig):
        self.config = config
        self.cache: Dict[int, np.ndarray] = {}

    def generate(self, slice_index: int) -> np.ndarray:
        """生成指定切片的模拟医学影像"""
        if slice_index in self.cache:
            return self.cache[slice_index].copy()

        w, h = self.config.image_width, self.config.image_height
        img = np.zeros((h, w), dtype=np.uint8)

        # 归一化的切片位置 (0-1)
        t = slice_index / max(self.config.num_dummy_slices - 1, 1)

        # === 1. 背景噪声 ===
        noise = np.random.randint(0, 15, (h, w), dtype=np.uint8)
        img = cv2.add(img, noise)

        # === 2. 躯干轮廓（大椭圆）===
        center_x, center_y = w // 2, h // 2
        body_w = int(w * 0.4)
        body_h = int(h * (0.35 + 0.1 * np.sin(t * np.pi)))  # 随切片变化
        cv2.ellipse(img, (center_x, center_y), (body_w, body_h),
                    0, 0, 360, 80, -1)

        # === 3. 脊柱（中央亮线）===
        spine_x = center_x + int(20 * np.sin(t * 2 * np.pi))
        cv2.line(img, (spine_x, center_y - body_h + 50),
                 (spine_x, center_y + body_h - 50), 120, 8)

        # === 4. 主要器官（随切片变化）===
        # 器官1 - 右侧圆形
        organ1_x = center_x + int(body_w * 0.4)
        organ1_y = center_y - int(body_h * 0.2)
        organ1_r = int(40 + 20 * np.sin(t * 3 * np.pi))
        if 0.2 < t < 0.8:  # 只在中间切片显示
            cv2.circle(img, (organ1_x, organ1_y), organ1_r, 50, -1)
            cv2.circle(img, (organ1_x, organ1_y), organ1_r - 5, 60, -1)

        # 器官2 - 左侧椭圆
        organ2_x = center_x - int(body_w * 0.35)
        organ2_y = center_y + int(body_h * 0.1)
        organ2_w = int(35 + 15 * np.cos(t * 2 * np.pi))
        organ2_h = int(45 + 10 * np.sin(t * 2.5 * np.pi))
        if 0.1 < t < 0.7:
            cv2.ellipse(img, (organ2_x, organ2_y), (organ2_w, organ2_h),
                        15, 0, 360, 55, -1)

        # 器官3 - 底部结构
        organ3_y = center_y + int(body_h * 0.4)
        organ3_w = int(60 + 20 * t)
        if 0.3 < t < 0.9:
            cv2.ellipse(img, (center_x, organ3_y), (organ3_w, 30),
                        0, 0, 180, 65, -1)

        # === 5. 骨骼结构（肋骨切面 - 高亮环形）===
        num_ribs = 5
        for i in range(num_ribs):
            rib_y = center_y - body_h + 60 + i * (2 * body_h // num_ribs - 20)
            rib_x_left = center_x - body_w + 30
            rib_x_right = center_x + body_w - 30

            # 左侧肋骨
            cv2.circle(img, (rib_x_left + 20, rib_y), 12, 180, 3)
            cv2.circle(img, (rib_x_left + 20, rib_y), 8, 200, -1)

            # 右侧肋骨
            cv2.circle(img, (rib_x_right - 20, rib_y), 12, 180, 3)
            cv2.circle(img, (rib_x_right - 20, rib_y), 8, 200, -1)

        # === 6. 血管/细节纹理 ===
        for _ in range(3):
            vx = center_x + np.random.randint(-body_w + 30, body_w - 30)
            vy = center_y + np.random.randint(-body_h + 40, body_h - 40)
            cv2.circle(img, (vx, vy), 3, 100, -1)

        # === 7. 添加高斯噪声 ===
        gaussian_noise = np.random.normal(0, 5, (h, w))
        img = np.clip(img.astype(np.float32) + gaussian_noise, 0, 255).astype(np.uint8)

        # === 8. 边缘模糊（模拟真实CT）===
        img = cv2.GaussianBlur(img, (3, 3), 0)

        # 缓存
        self.cache[slice_index] = img.copy()
        return img


# ============================================================================
#                           核心模块：手部追踪
# ============================================================================

class HandTracker:
    """MediaPipe 手部追踪封装"""

    # 关键点索引
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

    def __init__(self, config: ViewerConfig):
        self.config = config
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils

        # 平滑器
        self.left_smoother = ExponentialSmoother(config.smoothing_alpha)
        self.right_pinch_smoother = ExponentialSmoother(config.smoothing_alpha)

    def process(self, frame: np.ndarray) -> Tuple[HandData, HandData, np.ndarray]:
        """
        处理一帧图像

        Returns:
            left_hand: 左手数据
            right_hand: 右手数据
            annotated_frame: 带有手部标注的图像
        """
        h, w = frame.shape[:2]
        left_hand = HandData()
        right_hand = HandData()

        # BGR -> RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):
                # 获取手的标签 (注意：MediaPipe 返回的是镜像后的标签)
                label = handedness.classification[0].label
                confidence = handedness.classification[0].score

                # 提取关键点
                landmarks = self._extract_landmarks(hand_landmarks, w, h)

                # 计算食指尖和拇指尖位置
                index_tip = (
                    hand_landmarks.landmark[self.INDEX_TIP].x,
                    hand_landmarks.landmark[self.INDEX_TIP].y
                )
                thumb_tip = (
                    hand_landmarks.landmark[self.THUMB_TIP].x,
                    hand_landmarks.landmark[self.THUMB_TIP].y
                )

                # 计算 pinch 距离（归一化）
                pinch_dist = np.sqrt(
                    (index_tip[0] - thumb_tip[0]) ** 2 +
                    (index_tip[1] - thumb_tip[1]) ** 2
                )

                # 检测 OK 手势
                is_ok = self._detect_ok_gesture(hand_landmarks, pinch_dist, w, h)

                # 绘制手部骨架
                self._draw_hand(frame, hand_landmarks)

                # 根据标签分配数据
                # MediaPipe 的 "Left" 实际是用户的右手（镜像）
                if label == "Right":  # 实际是用户的左手
                    left_hand.detected = True
                    left_hand.landmarks = landmarks
                    left_hand.index_tip = index_tip
                    left_hand.thumb_tip = thumb_tip
                    left_hand.pinch_distance = pinch_dist
                    left_hand.is_ok_gesture = is_ok
                    left_hand.confidence = confidence
                    left_hand.smoothed_index_x = self.left_smoother.update(index_tip[0])
                    left_hand.smoothed_pinch = pinch_dist

                else:  # label == "Left"，实际是用户的右手
                    right_hand.detected = True
                    right_hand.landmarks = landmarks
                    right_hand.index_tip = index_tip
                    right_hand.thumb_tip = thumb_tip
                    right_hand.pinch_distance = pinch_dist
                    right_hand.is_ok_gesture = is_ok
                    right_hand.confidence = confidence
                    right_hand.smoothed_index_x = index_tip[0]
                    right_hand.smoothed_pinch = self.right_pinch_smoother.update(pinch_dist)

        return left_hand, right_hand, frame

    def _extract_landmarks(self, hand_landmarks, w: int, h: int) -> np.ndarray:
        """提取关键点坐标"""
        landmarks = np.zeros((21, 3))
        for i, lm in enumerate(hand_landmarks.landmark):
            landmarks[i] = [lm.x * w, lm.y * h, lm.z]
        return landmarks

    def _detect_ok_gesture(self, hand_landmarks, pinch_dist: float,
                           w: int, h: int) -> bool:
        """检测 OK 手势"""
        # 条件1：食指和拇指距离很近
        if pinch_dist > self.config.ok_gesture_threshold:
            return False

        # 条件2：其他三指伸直
        fingers_straight = 0
        for tip_idx, pip_idx in [
            (self.MIDDLE_TIP, self.MIDDLE_PIP),
            (self.RING_TIP, self.RING_PIP),
            (self.PINKY_TIP, self.PINKY_PIP)
        ]:
            tip_y = hand_landmarks.landmark[tip_idx].y
            pip_y = hand_landmarks.landmark[pip_idx].y
            # 指尖在指节上方（y更小）表示伸直
            if tip_y < pip_y - self.config.finger_straight_threshold:
                fingers_straight += 1

        return fingers_straight >= 2

    def _draw_hand(self, frame: np.ndarray, hand_landmarks):
        """绘制手部骨架"""
        self.mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
            self.mp_draw.DrawingSpec(color=(255, 255, 0), thickness=2)
        )

    def reset_smoothers(self):
        """重置平滑器"""
        self.left_smoother.reset()
        self.right_pinch_smoother.reset()

    def release(self):
        """释放资源"""
        self.hands.close()


# ============================================================================
#                         核心模块：手势解释器
# ============================================================================

class GestureInterpreterResult:
    """手势解释结果"""
    def __init__(self):
        self.slice_delta: int = 0          # 切片变化量
        self.zoom_level: Optional[float] = None  # 新的缩放级别
        self.add_marker: bool = False      # 是否添加标记
        self.marker_position: Optional[Tuple[float, float]] = None


class GestureInterpreter:
    """手势解释器 - 状态机核心"""

    def __init__(self, config: ViewerConfig):
        self.config = config

    def interpret(
        self,
        left_hand: HandData,
        right_hand: HandData,
        state: ViewerState,
        current_time: float
    ) -> Tuple[GestureInterpreterResult, SystemState]:
        """
        解释手势并生成控制指令

        Args:
            left_hand: 左手数据
            right_hand: 右手数据
            state: 当前系统状态
            current_time: 当前时间戳

        Returns:
            result: 解释结果
            new_system_state: 新的系统状态
        """
        result = GestureInterpreterResult()
        new_state = SystemState.IDLE

        # === 左手：切片导航 ===
        if left_hand.detected:
            new_state = SystemState.NAVIGATE

            # 计算与参考位置的偏移
            delta_x = left_hand.smoothed_index_x - state.navigate_reference_x

            # 判断是否超过阈值
            if abs(delta_x) > self.config.slice_move_threshold:
                # 检查 cooldown
                if current_time - state.last_slice_change_time > self.config.slice_cooldown:
                    # 确定方向
                    result.slice_delta = 1 if delta_x > 0 else -1
                    # 更新参考位置（部分更新，避免跳变）
                    new_ref = state.navigate_reference_x + delta_x * 0.5
                    state.navigate_reference_x = np.clip(new_ref, 0.2, 0.8)

        # === 右手：缩放和标记 ===
        if right_hand.detected:
            if new_state == SystemState.NAVIGATE:
                new_state = SystemState.NAVIGATE  # 双手同时工作，保持 NAVIGATE
            else:
                new_state = SystemState.ZOOM

            # 缩放控制（连续）
            pinch = right_hand.smoothed_pinch
            # 映射 pinch 距离到缩放级别
            pinch_normalized = (pinch - self.config.pinch_min_distance) / \
                              (self.config.pinch_max_distance - self.config.pinch_min_distance)
            pinch_normalized = np.clip(pinch_normalized, 0, 1)

            zoom = self.config.zoom_min + pinch_normalized * \
                   (self.config.zoom_max - self.config.zoom_min)
            result.zoom_level = zoom

            # OK 手势：添加标记（离散触发）
            if right_hand.is_ok_gesture:
                # 检查 cooldown
                if current_time - state.last_marker_time > self.config.marker_cooldown:
                    # 获取标记位置（使用食指尖位置）
                    mx, my = right_hand.index_tip

                    # 验证坐标在有效区域内
                    if 0.15 < mx < 0.85 and 0.15 < my < 0.85:
                        result.add_marker = True
                        result.marker_position = (mx, my)
                        new_state = SystemState.ADD_MARKER

        return result, new_state


# ============================================================================
#                         核心模块：影像查看器
# ============================================================================

class ImageViewer:
    """影像数据管理与渲染"""

    def __init__(self, config: ViewerConfig):
        self.config = config
        self.generator = DummyMedicalImageGenerator(config)

        # 状态
        self.current_slice = 0
        self.total_slices = config.num_dummy_slices
        self.zoom_level = config.zoom_default
        self.markers: Dict[int, List[Marker]] = {}

    def set_slice(self, slice_index: int):
        """设置当前切片"""
        self.current_slice = np.clip(slice_index, 0, self.total_slices - 1)

    def change_slice(self, delta: int):
        """切换切片"""
        self.set_slice(self.current_slice + delta)

    def set_zoom(self, zoom_level: float):
        """设置缩放级别"""
        self.zoom_level = np.clip(zoom_level,
                                   self.config.zoom_min,
                                   self.config.zoom_max)

    def add_marker(self, x: float, y: float, slice_index: int) -> Marker:
        """添加标记点"""
        marker = Marker(x=x, y=y, timestamp=time.time(), slice_index=slice_index)

        if slice_index not in self.markers:
            self.markers[slice_index] = []
        self.markers[slice_index].append(marker)

        return marker

    def get_current_markers(self) -> List[Marker]:
        """获取当前切片的标记点"""
        return self.markers.get(self.current_slice, [])

    def clear_markers(self, slice_index: Optional[int] = None):
        """清除标记点"""
        if slice_index is None:
            self.markers.clear()
        elif slice_index in self.markers:
            del self.markers[slice_index]

    def render(self, target_size: Tuple[int, int]) -> np.ndarray:
        """
        渲染当前切片图像

        Args:
            target_size: 目标尺寸 (width, height)

        Returns:
            渲染后的 BGR 图像
        """
        # 获取原始图像
        img = self.generator.generate(self.current_slice)

        # 转换为 BGR
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # 应用缩放
        img_bgr = self._apply_zoom(img_bgr)

        # 绘制标记点
        self._draw_markers(img_bgr)

        # 调整到目标尺寸
        img_bgr = cv2.resize(img_bgr, target_size)

        return img_bgr

    def _apply_zoom(self, img: np.ndarray) -> np.ndarray:
        """应用缩放变换"""
        if abs(self.zoom_level - 1.0) < 0.01:
            return img

        h, w = img.shape[:2]

        # 计算缩放后的尺寸
        new_w = int(w * self.zoom_level)
        new_h = int(h * self.zoom_level)

        # 放大
        if self.zoom_level > 1.0:
            zoomed = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            # 裁剪中心区域
            start_x = (new_w - w) // 2
            start_y = (new_h - h) // 2
            return zoomed[start_y:start_y + h, start_x:start_x + w]
        else:
            # 缩小
            zoomed = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            # 填充到原始尺寸
            result = np.zeros_like(img)
            start_x = (w - new_w) // 2
            start_y = (h - new_h) // 2
            result[start_y:start_y + new_h, start_x:start_x + new_w] = zoomed
            return result

    def _draw_markers(self, img: np.ndarray):
        """在图像上绘制标记点"""
        h, w = img.shape[:2]
        markers = self.get_current_markers()

        for i, marker in enumerate(markers):
            # 转换为像素坐标
            px, py = marker.to_image_coords(w, h)

            # 处理缩放偏移
            if self.zoom_level != 1.0:
                # 调整坐标以匹配缩放后的图像
                center_x, center_y = w // 2, h // 2
                px = int(center_x + (px - center_x) * self.zoom_level)
                py = int(center_y + (py - center_y) * self.zoom_level)

                # 边界检查
                if not (0 <= px < w and 0 <= py < h):
                    continue

            # 绘制标记（黄色空心圆 + 中心点）
            cv2.circle(img, (px, py), 12, (0, 255, 255), 2)
            cv2.circle(img, (px, py), 4, (0, 255, 255), -1)

            # 绘制序号
            cv2.putText(img, str(i + 1), (px + 15, py - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)


# ============================================================================
#                            UI 模块：叠加渲染
# ============================================================================

class OverlayRenderer:
    """UI 叠加层渲染器"""

    def __init__(self, config: ViewerConfig):
        self.config = config
        # 颜色定义 (BGR)
        self.color_bg = (10, 10, 10)           # 深黑背景
        self.color_text = (136, 255, 0)        # 青绿文字
        self.color_warning = (68, 68, 255)     # 红色警告
        self.color_marker = (0, 255, 255)      # 黄色标记
        self.color_notification = (255, 255, 255)  # 白色通知

    def render(
        self,
        main_image: np.ndarray,
        webcam_frame: np.ndarray,
        state: ViewerState
    ) -> np.ndarray:
        """
        渲染完整界面

        Args:
            main_image: 主影像（已渲染）
            webcam_frame: 摄像头帧（已标注手部）
            state: 系统状态

        Returns:
            完整的界面图像
        """
        display_w, display_h = self.config.main_display_size
        inset_w, inset_h = self.config.webcam_inset_size

        # 创建画布
        canvas = np.full((display_h, display_w, 3), self.color_bg, dtype=np.uint8)

        # 计算主影像位置（居中）
        main_h, main_w = main_image.shape[:2]
        main_x = (display_w - main_w) // 2
        main_y = (display_h - main_h) // 2

        # 绘制主影像
        canvas[main_y:main_y + main_h, main_x:main_x + main_w] = main_image

        # 绘制主影像边框
        cv2.rectangle(canvas, (main_x - 2, main_y - 2),
                     (main_x + main_w + 2, main_y + main_h + 2),
                     (100, 100, 100), 1)

        # 绘制左上角状态面板
        self._draw_status_panel(canvas, state)

        # 绘制左下角手势状态
        self._draw_hand_status(canvas, state)

        # 绘制右下角摄像头 inset
        self._draw_webcam_inset(canvas, webcam_frame, inset_w, inset_h)

        # 绘制通知（如果有）
        self._draw_notification(canvas, state)

        return canvas

    def _draw_status_panel(self, canvas: np.ndarray, state: ViewerState):
        """绘制状态面板"""
        x, y = 15, 25
        line_height = 28
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        # 背景
        panel_h = 130
        panel_w = 220
        overlay = canvas.copy()
        cv2.rectangle(overlay, (10, 10), (panel_w, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)

        # Slice
        text = f"Slice: {state.current_slice + 1}/{state.total_slices}"
        cv2.putText(canvas, text, (x, y), font, font_scale, self.color_text, thickness)

        # Zoom
        y += line_height
        text = f"Zoom: {state.zoom_level:.2f}x"
        cv2.putText(canvas, text, (x, y), font, font_scale, self.color_text, thickness)

        # Mode
        y += line_height
        mode_color = self._get_mode_color(state.system_state)
        text = f"Mode: {state.system_state.value}"
        cv2.putText(canvas, text, (x, y), font, font_scale, mode_color, thickness)

        # Markers
        y += line_height
        marker_count = len(state.markers.get(state.current_slice, []))
        text = f"Markers: {marker_count}"
        cv2.putText(canvas, text, (x, y), font, font_scale,
                   self.color_marker if marker_count > 0 else self.color_text, thickness)

    def _draw_hand_status(self, canvas: np.ndarray, state: ViewerState):
        """绘制手势检测状态"""
        x, y = 15, canvas.shape[0] - 65
        line_height = 25
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1

        # 背景
        overlay = canvas.copy()
        cv2.rectangle(overlay, (10, y - 20), (200, canvas.shape[0] - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)

        # Left Hand
        left_color = self.color_text if state.left_hand.detected else self.color_warning
        left_text = "Detected" if state.left_hand.detected else "Not Detected"
        text = f"Left Hand: {left_text}"
        cv2.putText(canvas, text, (x, y), font, font_scale, left_color, thickness)

        # Right Hand
        y += line_height
        right_color = self.color_text if state.right_hand.detected else self.color_warning
        right_text = "Detected" if state.right_hand.detected else "Not Detected"
        text = f"Right Hand: {right_text}"
        cv2.putText(canvas, text, (x, y), font, font_scale, right_color, thickness)

    def _draw_webcam_inset(self, canvas: np.ndarray, webcam_frame: np.ndarray,
                           inset_w: int, inset_h: int):
        """绘制摄像头画中画"""
        display_h, display_w = canvas.shape[:2]

        # 调整 webcam 尺寸
        webcam_resized = cv2.resize(webcam_frame, (inset_w, inset_h))

        # 位置：右下角
        x = display_w - inset_w - 15
        y = display_h - inset_h - 15

        # 半透明背景
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x - 3, y - 3), (x + inset_w + 3, y + inset_h + 3),
                     (50, 50, 50), -1)
        cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0, canvas)

        # 绘制 webcam
        canvas[y:y + inset_h, x:x + inset_w] = webcam_resized

        # 边框
        cv2.rectangle(canvas, (x - 2, y - 2), (x + inset_w + 2, y + inset_h + 2),
                     (80, 80, 80), 2)

        # 标签
        cv2.putText(canvas, "Camera", (x + 5, y - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    def _draw_notification(self, canvas: np.ndarray, state: ViewerState):
        """绘制通知"""
        if state.notification is None:
            return

        current_time = time.time()
        if current_time > state.notification_expire_time:
            state.notification = None
            return

        # 计算透明度（淡出效果）
        remaining = state.notification_expire_time - current_time
        alpha = min(1.0, remaining / 0.5)  # 最后0.5秒淡出

        display_h, display_w = canvas.shape[:2]
        text = state.notification

        # 计算文字位置
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 3
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

        x = (display_w - text_w) // 2
        y = display_h // 2

        # 背景
        overlay = canvas.copy()
        padding = 20
        cv2.rectangle(overlay,
                     (x - padding, y - text_h - padding),
                     (x + text_w + padding, y + padding),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6 * alpha, canvas, 0.4, 0, canvas)

        # 文字
        color = tuple(int(c * alpha) for c in self.color_notification)
        cv2.putText(canvas, text, (x, y), font, font_scale, color, thickness)

    def _get_mode_color(self, mode: SystemState) -> Tuple[int, int, int]:
        """获取模式对应的颜色"""
        colors = {
            SystemState.IDLE: (150, 150, 150),      # 灰色
            SystemState.NAVIGATE: (136, 255, 0),    # 青绿
            SystemState.ZOOM: (255, 200, 0),        # 青色
            SystemState.ADD_MARKER: (0, 255, 255),  # 黄色
        }
        return colors.get(mode, self.color_text)


# ============================================================================
#                          主控制器
# ============================================================================

class MedicalGestureViewer:
    """医学影像手势控制器 - 主类"""

    def __init__(self, config: Optional[ViewerConfig] = None):
        self.config = config or ViewerConfig()

        # 初始化各模块
        self.hand_tracker = HandTracker(self.config)
        self.gesture_interpreter = GestureInterpreter(self.config)
        self.image_viewer = ImageViewer(self.config)
        self.overlay_renderer = OverlayRenderer(self.config)

        # 系统状态
        self.state = ViewerState(
            total_slices=self.config.num_dummy_slices,
            zoom_level=self.config.zoom_default
        )

        # 回调
        self.on_slice_change: Optional[Callable[[int], None]] = None
        self.on_zoom_change: Optional[Callable[[float], None]] = None
        self.on_marker_add: Optional[Callable[[Marker], None]] = None

        # 运行状态
        self.is_running = False

    def process_frame(self, webcam_frame: np.ndarray) -> np.ndarray:
        """
        处理一帧摄像头图像

        Args:
            webcam_frame: BGR 格式的摄像头帧

        Returns:
            渲染后的完整界面图像
        """
        current_time = time.time()

        # 镜像翻转
        webcam_frame = cv2.flip(webcam_frame, 1)

        # 1. 手部追踪
        left_hand, right_hand, annotated_webcam = self.hand_tracker.process(webcam_frame)

        # 更新状态
        self.state.left_hand = left_hand
        self.state.right_hand = right_hand

        # 2. 手势解释
        result, new_system_state = self.gesture_interpreter.interpret(
            left_hand, right_hand, self.state, current_time
        )

        # 3. 执行控制指令

        # 切片切换
        if result.slice_delta != 0:
            self.image_viewer.change_slice(result.slice_delta)
            self.state.current_slice = self.image_viewer.current_slice
            self.state.last_slice_change_time = current_time
            self.state.navigate_reference_x = left_hand.smoothed_index_x

            if self.on_slice_change:
                self.on_slice_change(self.state.current_slice)

        # 缩放
        if result.zoom_level is not None:
            self.image_viewer.set_zoom(result.zoom_level)
            self.state.zoom_level = self.image_viewer.zoom_level

            if self.on_zoom_change:
                self.on_zoom_change(self.state.zoom_level)

        # 添加标记
        if result.add_marker and result.marker_position:
            mx, my = result.marker_position
            marker = self.image_viewer.add_marker(mx, my, self.state.current_slice)

            self.state.last_marker_time = current_time
            self.state.notification = "MARKER ADDED"
            self.state.notification_expire_time = current_time + self.config.notification_duration

            if self.on_marker_add:
                self.on_marker_add(marker)

        # 更新系统状态
        self.state.system_state = new_system_state

        # 4. 渲染主影像
        main_image = self.image_viewer.render(self.config.main_display_size)

        # 5. 渲染完整界面
        canvas = self.overlay_renderer.render(main_image, annotated_webcam, self.state)

        return canvas

    def load_images(self, images: List[np.ndarray]):
        """
        加载真实图像序列

        Args:
            images: 图像列表（灰度或BGR）
        """
        # 这里可以替换 DummyMedicalImageGenerator
        # 暂时用切片数更新
        self.image_viewer.total_slices = len(images)
        self.state.total_slices = len(images)
        # TODO: 实现真实图像加载

    def set_total_slices(self, total: int):
        """设置总切片数"""
        self.image_viewer.total_slices = total
        self.state.total_slices = total
        self.image_viewer.generator.cache.clear()

    def get_markers(self) -> Dict[int, List[Marker]]:
        """获取所有标记点"""
        return self.image_viewer.markers

    def clear_markers(self, slice_index: Optional[int] = None):
        """清除标记点"""
        self.image_viewer.clear_markers(slice_index)

    def reset(self):
        """重置状态"""
        self.state = ViewerState(
            total_slices=self.config.num_dummy_slices,
            zoom_level=self.config.zoom_default
        )
        self.image_viewer.current_slice = 0
        self.image_viewer.zoom_level = self.config.zoom_default
        self.image_viewer.markers.clear()
        self.hand_tracker.reset_smoothers()

    def run(self, camera_index: int = 0):
        """
        运行独立演示

        Args:
            camera_index: 摄像头索引
        """
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            print("Error: Cannot open camera")
            return

        self.is_running = True
        print("=" * 60)
        print("Medical Image Gesture-Controlled Viewer")
        print("=" * 60)
        print("\nControls:")
        print("  - Left hand index finger: Swipe left/right to navigate slices")
        print("  - Right hand pinch: Control zoom level")
        print("  - Right hand OK gesture: Add marker at current position")
        print("\nKeyboard:")
        print("  - 'q': Quit")
        print("  - 'r': Reset viewer")
        print("  - 'c': Clear markers on current slice")
        print("=" * 60)

        cv2.namedWindow("Medical Viewer", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Medical Viewer", *self.config.main_display_size)

        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                break

            # 处理帧
            canvas = self.process_frame(frame)

            # 显示
            cv2.imshow("Medical Viewer", canvas)

            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.is_running = False
            elif key == ord('r'):
                self.reset()
                print("Viewer reset")
            elif key == ord('c'):
                self.clear_markers(self.state.current_slice)
                print(f"Markers cleared on slice {self.state.current_slice + 1}")

        cap.release()
        cv2.destroyAllWindows()

    def release(self):
        """释放资源"""
        self.hand_tracker.release()
        self.is_running = False


# ============================================================================
#                              Gradio 接口
# ============================================================================

def create_gradio_interface(viewer: Optional[MedicalGestureViewer] = None):
    """
    创建 Gradio 界面

    Args:
        viewer: MedicalGestureViewer 实例

    Returns:
        Gradio Interface
    """
    try:
        import gradio as gr
    except ImportError:
        raise ImportError("Gradio is required. Install with: pip install gradio")

    viewer = viewer or MedicalGestureViewer()

    def process_webcam(frame):
        """Gradio 回调函数"""
        if frame is None:
            return None

        # 处理帧
        canvas = viewer.process_frame(frame)

        # 转换 RGB -> BGR 用于 Gradio
        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    def get_status():
        """获取当前状态"""
        return {
            "slice": f"{viewer.state.current_slice + 1}/{viewer.state.total_slices}",
            "zoom": f"{viewer.state.zoom_level:.2f}x",
            "mode": viewer.state.system_state.value,
            "markers": len(viewer.get_markers())
        }

    interface = gr.Interface(
        fn=process_webcam,
        inputs=gr.Webcam(label="Camera Input", streaming=True),
        outputs=gr.Image(label="Medical Viewer"),
        title="Medical Image Gesture Controller",
        description="Use hand gestures to control the medical image viewer",
        live=True
    )

    return interface, viewer


# ============================================================================
#                              主入口
# ============================================================================

if __name__ == "__main__":
    # 创建配置
    config = ViewerConfig()

    # 创建查看器
    viewer = MedicalGestureViewer(config)

    # 设置回调（可选）
    def on_slice_change(slice_index):
        pass  # print(f"Slice: {slice_index + 1}")

    def on_marker_add(marker):
        print(f"Marker added at ({marker.x:.2f}, {marker.y:.2f}) on slice {marker.slice_index + 1}")

    viewer.on_slice_change = on_slice_change
    viewer.on_marker_add = on_marker_add

    # 运行
    try:
        viewer.run()
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        viewer.release()
