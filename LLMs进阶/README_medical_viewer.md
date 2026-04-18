# Medical Image Gesture-Controlled Viewer

医学影像非接触交互原型 - 使用说明

## 快速开始

### 安装依赖

```bash
pip install opencv-python mediapipe numpy
```

### 运行

```bash
python media_control.py
```

## 交互说明

| 手势 | 功能 |
|------|------|
| 左手食指左右移动 | 切换切片（左移上一张，右移下一张） |
| 右手食指拇指捏合距离 | 控制缩放（0.5x - 3.0x） |
| 右手 OK 手势 | 在当前切片添加标记点 |

### 键盘快捷键

- `q` - 退出
- `r` - 重置查看器
- `c` - 清除当前切片的标记点

---

## 代码结构

```
media_control.py
│
├── ViewerConfig              # 配置参数
│
├── SystemState               # 系统状态枚举
├── HandData                  # 手部追踪数据
├── Marker                    # 标记点数据
├── ViewerState               # 全局状态
│
├── ExponentialSmoother       # 指数平滑滤波器
├── DummyMedicalImageGenerator # 模拟医学影像生成器
│
├── HandTracker               # 手部追踪模块
├── GestureInterpreter        # 手势解释模块（状态机）
├── ImageViewer               # 影像管理模块
├── OverlayRenderer           # UI渲染模块
│
├── MedicalGestureViewer      # 主控制器
│
└── create_gradio_interface() # Gradio接口
```

---

## 后续扩展

### 1. 接入真实 DICOM 文件

```python
import pydicom
import numpy as np

def load_dicom_series(directory: str) -> List[np.ndarray]:
    """加载 DICOM 序列"""
    import os
    import glob

    files = glob.glob(os.path.join(directory, "*.dcm"))
    files.sort(key=lambda x: pydicom.dcmread(x).InstanceNumber)

    images = []
    for f in files:
        ds = pydicom.dcmread(f)
        img = ds.pixel_array

        # 窗宽窗位调整
        window_center = ds.WindowCenter if hasattr(ds, 'WindowCenter') else 127
        window_width = ds.WindowWidth if hasattr(ds, 'WindowWidth') else 256

        img = np.clip(img, window_center - window_width//2,
                      window_center + window_width//2)
        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)

        images.append(img)

    return images

# 使用
viewer = MedicalGestureViewer()
images = load_dicom_series("/path/to/dicom/folder")
viewer.set_total_slices(len(images))
# 进一步修改 ImageViewer 类以使用真实图像
```

### 2. 封装为独立类库

代码已按模块化设计，可拆分为多文件：

```
medical_gesture_viewer/
├── __init__.py
├── config.py              # ViewerConfig
├── models.py              # 数据结构
├── utils/
│   ├── smoother.py        # ExponentialSmoother
│   └── dummy_data.py      # DummyMedicalImageGenerator
├── core/
│   ├── hand_tracker.py    # HandTracker
│   ├── gesture.py         # GestureInterpreter
│   └── image_viewer.py    # ImageViewer
├── ui/
│   └── renderer.py        # OverlayRenderer
└── viewer.py              # MedicalGestureViewer
```

### 3. 接入 Gradio Web UI

```python
import gradio as gr
from media_control import MedicalGestureViewer, create_gradio_interface

# 方式1：使用内置接口
interface, viewer = create_gradio_interface()
interface.launch()

# 方式2：自定义界面
viewer = MedicalGestureViewer()

with gr.Blocks() as demo:
    gr.Markdown("# Medical Image Gesture Controller")

    with gr.Row():
        webcam = gr.Webcam(label="Camera", streaming=True)
        output = gr.Image(label="Viewer")

    with gr.Row():
        slice_info = gr.Textbox(label="Slice")
        zoom_info = gr.Textbox(label="Zoom")

    def process(frame):
        if frame is None:
            return None, "", ""
        canvas = viewer.process_frame(frame)
        return (cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB),
                f"{viewer.state.current_slice + 1}/{viewer.state.total_slices}",
                f"{viewer.state.zoom_level:.2f}x")

    webcam.stream(process, inputs=[webcam],
                  outputs=[output, slice_info, zoom_info])

demo.launch()
```

---

## 配置参数调优

```python
config = ViewerConfig(
    # 手势灵敏度
    slice_move_threshold=0.06,    # 降低=更敏感
    slice_cooldown=0.35,          # 增大=更稳定

    # 缩放范围
    zoom_min=0.5,
    zoom_max=3.0,

    # 平滑程度
    smoothing_alpha=0.25,         # 降低=更平滑但响应慢

    # 显示尺寸
    main_display_size=(1024, 768),
    webcam_inset_size=(320, 240),
)

viewer = MedicalGestureViewer(config)
```

---

## 注意事项

1. **光照条件**：确保摄像头画面清晰，光照充足
2. **手势识别**：保持手掌朝向摄像头，手指尽量伸展
3. **OK手势**：食指拇指成圈后，其他三指需伸直
4. **性能**：如需更高帧率，可降低 `min_detection_confidence`
