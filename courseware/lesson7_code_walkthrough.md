# 第七课新增代码逐段讲解：目标检测接入 Web 系统

这份文档专门讲第七课新增和改动的代码。它不是只解释“这一行是什么意思”，而是按照课堂讲解顺序，把每个代码文件放回整个项目闭环里理解：

```text
用户上传图片
  -> templates/index.html 收集图片和模式
  -> app/web_app.py 接收请求
  -> 分类：FineTunedLoader + MobileNetV3
  -> 检测：YOLOv8Detector + draw_detections
  -> Flask 返回 JSON
  -> 前端渲染分类结果或带框检测结果
```

第六课已经完成了“图片分类 Web 页面”。第七课不是重写项目，而是在第六课基础上新增“目标检测”能力。所以讲代码时要一直提醒学生：分类和检测是并列的两条推理路径，共用同一个 Flask 页面、上传逻辑和结果展示框架。

## 一、先让学生建立总关系

| 文件 | 在第七课中的角色 | 和第六课的关系 |
|---|---|---|
| `app/object_detector.py` | 新增：封装目标检测模型 | 类似第六课 `model_loader.py` 的抽象思想，但服务检测任务 |
| `app/preprocess.py` | 新增：把检测结果画回图片 | 第六课只显示分类概率，第七课要把结果可视化 |
| `app/web_app.py` | 改动：在 `/predict` 旁边新增 `/detect` | 保留第六课分类接口，新增检测接口 |
| `templates/index.html` | 改动：从单一分类页面升级成分类/检测双模式页面 | 第六课页面只上传并显示 Top-3，第七课多了检测 Tab 和阈值滑块 |
| `requirements.txt` | 改动：增加检测相关依赖 | 第六课 Flask + PyTorch，第七课还需要 `ultralytics` |

课堂可以先用一句话概括：

> 第六课的系统会回答“这张图是什么”，第七课的系统会进一步回答“图里有什么、它在哪里”。

## 二、`app/object_detector.py` 逐段讲解

文件路径：`app/object_detector.py`

这个文件是第七课最核心的新增文件。它把“目标检测”封装成一个独立模块，避免把 YOLOv8 的细节全部塞进 Flask 后端。

### 第 1-7 行：文件说明和 CPSC210 映射

```python
"""Object detection abstractions and YOLOv8 implementation.

CPSC 210 mapping:
- ObjectDetector is the interface used by the Web layer.
- YOLOv8Detector is one replaceable implementation of that interface.
- DetectionResult is a small value object for moving detection data around.
"""
```

小白解释：

- 这段不是程序运行逻辑，而是给读代码的人看的说明。
- 它告诉我们这个文件有三个核心角色：`DetectionResult`、`ObjectDetector`、`YOLOv8Detector`。
- 这也正好对应 CPSC210 里的几个概念：接口、实现类、值对象。

和前面课程的联系：

- 第六课的 `model_loader.py` 里有 `ModelLoader`、`FineTunedLoader`。
- 第七课这里用同样思想写了 `ObjectDetector`、`YOLOv8Detector`。
- 学生可以理解为：第六课抽象“怎么加载分类模型”，第七课抽象“怎么执行检测模型”。

### 第 9-14 行：导入依赖

```python
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
```

逐项解释：

- `annotations`：让类型标注写法更灵活。
- `os`：后面用于设置 `YOLO_CONFIG_DIR` 环境变量。
- `ABC`、`abstractmethod`：用来定义抽象接口。
- `dataclass`：快速创建只保存数据的小类。
- `Path`：更稳定地处理 Windows/macOS/Linux 路径。

课堂提醒：

- 小白学生容易把 `import` 理解成“复制代码”。可以解释为：导入工具箱，后面代码要用这些工具。

### 第 17-36 行：`DetectionResult` 检测结果对象

```python
@dataclass(frozen=True)
class DetectionResult:
    x1: float
    y1: float
    x2: float
    y2: float
    label: str
    confidence: float

    def to_dict(self) -> dict[str, float | str]:
        return {
            "x1": round(self.x1, 2),
            "y1": round(self.y1, 2),
            "x2": round(self.x2, 2),
            "y2": round(self.y2, 2),
            "label": self.label,
            "confidence": round(self.confidence * 100.0, 2),
        }
```

小白解释：

- 每检测到一个物体，就创建一个 `DetectionResult`。
- `x1, y1, x2, y2` 是检测框坐标。
- `label` 是类别名，比如 `cat`、`dog`、`person`。
- `confidence` 是置信度，比如 `0.83` 表示模型认为这个结果比较可信。
- `to_dict()` 是把 Python 对象转成字典，方便 Flask 返回 JSON。

为什么要这么写：

- 如果直接在项目各处传一堆散乱变量，后面很难维护。
- 用一个对象保存一个检测结果，代码更清楚。

和其他文件的联系：

- `app/object_detector.py` 负责创建 `DetectionResult`。
- `app/preprocess.py` 读取 `DetectionResult` 的坐标来画框。
- `app/web_app.py` 调用 `to_dict()`，把检测结果返回给前端。
- `templates/index.html` 读取 JSON 里的 `x1/y1/x2/y2/label/confidence` 并展示。

课堂讲法：

> `DetectionResult` 就像一张检测小卡片：上面写着“物体是谁、框在哪里、模型有多确定”。

### 第 39-44 行：`ObjectDetector` 抽象接口

```python
class ObjectDetector(ABC):
    """Interface for object detectors."""

    @abstractmethod
    def detect(self, image_path: str | Path) -> list[DetectionResult]:
        """Detect objects in one image and return normalized result objects."""
```

小白解释：

- `ObjectDetector` 不直接检测图片。
- 它规定：任何检测器都必须有一个 `detect()` 方法。
- `detect()` 输入图片路径，输出检测结果列表。

CPSC210 连接：

- 这就是 interface / abstraction。
- Web 层不需要知道底层是 YOLOv8、YOLOv10 还是其他检测模型。
- 只要新模型也实现 `detect()`，后端调用方式就可以保持一致。

和第六课联系：

- 第六课 `ModelLoader` 规定模型加载器必须有 `load_model()`。
- 第七课 `ObjectDetector` 规定检测器必须有 `detect()`。
- 两者都是“先规定接口，再写具体实现”的思想。

### 第 47-72 行：`YOLOv8Detector.__init__()` 初始化检测器

```python
class YOLOv8Detector(ObjectDetector):
    def __init__(
        self,
        model_path: str | Path = "yolov8n.pt",
        conf_threshold: float = 0.5,
    ) -> None:
        if not 0.0 <= conf_threshold <= 1.0:
            raise ValueError("conf_threshold must be between 0.0 and 1.0.")

        config_dir = Path(__file__).resolve().parents[1] / ".ultralytics"
        config_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))

        from ultralytics import YOLO

        self.model_path = str(model_path)
        self.conf_threshold = conf_threshold
        self.model = YOLO(self.model_path)
```

逐段解释：

- `YOLOv8Detector(ObjectDetector)`：说明它是 `ObjectDetector` 的一种具体实现。
- `model_path="yolov8n.pt"`：默认使用 YOLOv8 nano 版本权重。
- `conf_threshold=0.5`：默认只保留置信度不低于 0.5 的结果。
- `if not 0.0 <= conf_threshold <= 1.0`：防止学生传入无效阈值。
- `.ultralytics` 目录：把 YOLO 的配置写在项目目录里，减少用户目录权限问题。
- `from ultralytics import YOLO`：导入 YOLOv8 官方库。
- `self.model = YOLO(self.model_path)`：真正加载检测模型。

和课堂操作联系：

- 如果没有安装 `ultralytics`，这里会报错，所以课件里要求先执行：

```bash
pip install -r requirements.txt
```

- 如果项目根目录没有 `yolov8n.pt`，Ultralytics 可能会尝试自动下载。
- 第一次运行慢，通常是因为在下载或加载 YOLO 权重。

### 第 74-108 行：`detect()` 执行目标检测

```python
def detect(self, image_path: str | Path) -> list[DetectionResult]:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    results = self.model(str(path), conf=self.conf_threshold, verbose=False)
    if not results:
        return []

    result = results[0]
    boxes = result.boxes
    if boxes is None:
        return []

    detections: list[DetectionResult] = []
    for box in boxes:
        confidence = float(box.conf[0].item())
        if confidence < self.conf_threshold:
            continue

        x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
        class_id = int(box.cls[0].item())
        label = str(result.names[class_id])
        detections.append(
            DetectionResult(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                label=label,
                confidence=confidence,
            )
        )

    return detections
```

逐段解释：

- 先把传入路径转成 `Path`，并检查图片是否存在。
- `self.model(...)` 是真正让 YOLOv8 做推理。
- `results[0]` 表示当前只处理一张上传图片。
- `result.boxes` 是 YOLO 找到的所有检测框。
- 遍历每个 `box`，读取置信度、坐标、类别编号。
- `result.names[class_id]` 把类别编号转成类别名。
- 最后把每个框整理成 `DetectionResult`。

和其他文件的联系：

- `web_app.py` 的 `detect_image()` 会调用这里的 `detector.detect(image_path)`。
- `preprocess.py` 的 `draw_detections()` 会拿这里返回的检测结果画框。
- 前端页面会显示这里最终返回的类别、置信度和坐标。

课堂讲法：

> 这一段就是“把 YOLO 的原始输出翻译成我们项目能理解的数据格式”。

## 三、`app/preprocess.py` 逐段讲解

文件路径：`app/preprocess.py`

这个文件负责后处理。目标检测模型输出的是数字坐标，但用户不想只看数字，所以要把检测框画回图片。

### 第 1-10 行：导入检测结果类型

```python
from pathlib import Path

try:
    from app.object_detector import DetectionResult
except ModuleNotFoundError:
    from object_detector import DetectionResult
```

小白解释：

- 这里要用 `DetectionResult`，因为画框需要读取 `x1/y1/x2/y2/label/confidence`。
- `try/except` 是为了兼容两种运行方式：
  - 从项目根目录运行：`from app.object_detector import DetectionResult`
  - 从 `app` 文件夹内部运行：`from object_detector import DetectionResult`

和第六课联系：

- 第六课也遇到过“从哪里运行脚本会影响导入路径”的问题。
- 这里是为了降低学生复现时的路径错误概率。

### 第 13-25 行：`draw_detections()` 函数入口和 OpenCV 依赖

```python
def draw_detections(
    image_path: str | Path,
    detections: list[DetectionResult],
    output_dir: str | Path | None = None,
) -> Path:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "opencv-python is not installed. Run `pip install -r requirements.txt`."
        ) from exc
```

小白解释：

- `image_path`：原始图片路径。
- `detections`：YOLO 检测出来的结果列表。
- `output_dir`：画完框以后保存到哪里。
- `cv2` 是 OpenCV，用来读图、画矩形、写文字、保存图片。

为什么在函数里面导入 `cv2`：

- 如果学生只跑分类，不一定立刻用到 OpenCV。
- 真正执行检测画框时再导入，报错信息也更接近问题来源。

### 第 27-34 行：读取图片并做安全检查

```python
source = Path(image_path)
if not source.exists():
    raise FileNotFoundError(f"Image not found: {source}")

image = cv2.imread(str(source))
if image is None:
    raise ValueError(f"Unable to read image file: {source}")
```

小白解释：

- 先确认图片文件存在。
- 再用 OpenCV 读取图片。
- 如果文件不是有效图片，`cv2.imread()` 可能返回 `None`。

课堂强调：

- 报错不是坏事，清楚的报错能帮助定位问题。
- 这里属于“输入检查”，防止后面画框时出现更难懂的错误。

### 第 35-44 行：裁剪坐标并画矩形框

```python
height, width = image.shape[:2]
for detection in detections:
    x1 = max(0, min(width - 1, int(detection.x1)))
    y1 = max(0, min(height - 1, int(detection.y1)))
    x2 = max(0, min(width - 1, int(detection.x2)))
    y2 = max(0, min(height - 1, int(detection.y2)))

    color = (39, 174, 96)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
```

逐段解释：

- `height, width` 取出图片高度和宽度。
- `max/min` 是为了防止检测框坐标超出图片边界。
- `color = (39, 174, 96)` 是绿色，在 OpenCV 里顺序是 BGR，不是 RGB。
- `cv2.rectangle()` 把矩形框画到图片上。

和检测模型联系：

- YOLO 输出的是坐标。
- 这段代码把坐标变成用户看得见的框。

### 第 45-70 行：绘制类别名和置信度

```python
label = f"{detection.label}: {detection.confidence * 100:.1f}%"
(text_width, text_height), _ = cv2.getTextSize(...)
cv2.rectangle(...)
cv2.putText(...)
```

小白解释：

- `label` 把类别和置信度拼成文字，比如 `cat: 91.3%`。
- `getTextSize()` 先计算文字会占多大空间。
- 第一段 `cv2.rectangle()` 画一个绿色文字背景。
- `cv2.putText()` 把白色文字写上去。

为什么要先画背景：

- 如果直接把文字写在图片上，可能被复杂背景遮住。
- 加一个色块可以让学生和用户更容易看清结果。

### 第 72-76 行：保存带框图片

```python
target_dir = Path(output_dir) if output_dir is not None else source.parent
target_dir.mkdir(parents=True, exist_ok=True)
target = target_dir / f"{source.stem}_detected{source.suffix or '.jpg'}"
cv2.imwrite(str(target), image)
return target
```

小白解释：

- 如果指定了输出目录，就保存到指定目录。
- 如果没指定，就保存到原图同目录。
- 文件名会变成 `原文件名_detected.jpg` 这种形式。
- 返回 `target`，让 Flask 知道带框图片在哪里。

和其他文件的联系：

- `web_app.py` 把 `output_dir` 设置成 `static/uploads`。
- Flask 通过 `image_url()` 把这个本地路径转成浏览器能访问的 URL。
- `index.html` 里的 `resultImage.src = payload.image_url` 显示这张图片。

## 四、`app/web_app.py` 逐段讲解

文件路径：`app/web_app.py`

这个文件是第六课和第七课的连接点。第六课已经有分类功能，第七课在它旁边加上检测功能。

### 第 1-10 行：文件说明

```python
"""Flask web application for the image recognition project.

Lesson 7 extends the lesson 6 classification app with an object-detection
workflow. The Web app now exposes two endpoints:
- /predict for image classification.
- /detect for YOLOv8 object detection.
"""
```

课堂讲法：

- `/predict` 是第六课保留下来的分类接口。
- `/detect` 是第七课新增的检测接口。
- 一个 Web 应用可以同时提供多个 AI 能力。

### 第 14-23 行：基础依赖导入

```python
import argparse
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
```

逐项解释：

- `argparse`：支持命令行参数，比如改端口。
- `lru_cache`：缓存类别名，避免重复扫描数据集。
- `Path`：处理路径。
- `uuid4`：给上传图片生成唯一文件名，避免重名覆盖。
- `torch`、`F.softmax`：分类推理和概率计算。
- `PIL.Image`、`transforms`：把图片变成模型能吃的 tensor。
- `ImageFolder`：读取 `data/train` 下的类别名。

和第六课联系：

- 这些多数是第六课分类 Web 已经用到的内容。
- 第七课保留它们，因为分类功能没有被删除。

### 第 25-40 行：Flask 和项目内部模块导入

```python
from flask import Flask, jsonify, render_template, request, url_for
from werkzeug.utils import secure_filename

from app.model_loader import FineTunedLoader
from app.object_detector import YOLOv8Detector
from app.preprocess import draw_detections
```

小白解释：

- `Flask`：创建后端应用。
- `jsonify`：把 Python 字典变成 JSON 响应。
- `render_template`：返回 HTML 页面。
- `request`：读取浏览器上传的文件和表单参数。
- `url_for`：生成静态文件 URL。
- `secure_filename`：把用户上传的文件名变安全。

第七课新增点：

- `YOLOv8Detector`：检测模型。
- `draw_detections`：画检测框。

文件之间的关系：

- `web_app.py` 不直接写 YOLO 细节，而是调用 `object_detector.py`。
- `web_app.py` 不直接写 OpenCV 画框细节，而是调用 `preprocess.py`。

### 第 43-67 行：项目路径、配置和全局缓存

```python
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "oxford_pet_split"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "oxford_pet_mobilenet_epoch1.pth"
DEFAULT_DETECTOR_MODEL = PROJECT_ROOT / "yolov8n.pt"
STATIC_FOLDER = PROJECT_ROOT / "static"
TEMPLATE_FOLDER = PROJECT_ROOT / "templates"
UPLOAD_FOLDER = STATIC_FOLDER / "uploads"
IMAGE_SIZE = 224
DEFAULT_DETECTION_THRESHOLD = 0.5
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
```

小白解释：

- `PROJECT_ROOT` 是项目根目录。
- `DEFAULT_DATA_ROOT` 是默认数据集目录。
- `DEFAULT_MODEL_PATH` 是分类模型权重路径。
- `DEFAULT_DETECTOR_MODEL` 是 YOLOv8 权重路径。
- `UPLOAD_FOLDER` 是用户上传图片保存的位置。
- `IMAGE_SIZE = 224` 来自 MobileNetV3 的输入尺寸要求。
- `DEFAULT_DETECTION_THRESHOLD = 0.5` 是默认检测阈值。
- `ALLOWED_EXTENSIONS` 限制上传格式。

和课堂操作联系：

- 学生报 `No fine-tuned checkpoint found`，就检查 `DEFAULT_MODEL_PATH`。
- 学生报找不到数据，就检查 `DEFAULT_DATA_ROOT` 或 `data/train`。
- 学生检测第一次慢，就检查 `yolov8n.pt` 是否存在。

### 第 54-67 行：创建 Flask app 和模型缓存变量

```python
app = Flask(__name__, static_folder=str(STATIC_FOLDER), template_folder=str(TEMPLATE_FOLDER))
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

_model = None
_detector = None
_detector_threshold: float | None = None
_metadata: dict[str, str] = {}
```

小白解释：

- `static_folder` 告诉 Flask 静态文件在哪里。
- `template_folder` 告诉 Flask HTML 模板在哪里。
- `MAX_CONTENT_LENGTH` 限制上传文件大小，防止上传过大图片。
- `_model` 缓存分类模型。
- `_detector` 缓存检测模型。
- `_detector_threshold` 记录当前检测器用的阈值。

为什么要缓存模型：

- 模型加载很慢。
- 如果每次上传图片都重新加载模型，网页会非常卡。
- 所以第一次请求时加载，后面复用。

和第六课联系：

- 第六课已经讲过“模型单例”思想。
- 第七课把同样思想扩展到检测模型。

### 第 70-108 行：解析数据、分类模型、检测模型路径

```python
def resolve_data_root() -> Path:
    for root in (DEFAULT_DATA_ROOT, PROJECT_ROOT / "data"):
        train_dir = root / "train"
        if train_dir.exists():
            return root

def resolve_model_path() -> Path:
    for candidate in (DEFAULT_MODEL_PATH, PROJECT_ROOT / "finetuned_mobilenet.pth"):
        if candidate.exists():
            return candidate

def resolve_detector_model_path() -> str:
    if DEFAULT_DETECTOR_MODEL.exists():
        return str(DEFAULT_DETECTOR_MODEL)
    return "yolov8n.pt"
```

小白解释：

- `resolve_data_root()`：找到数据集目录。
- `resolve_model_path()`：找到第六课训练好的分类权重。
- `resolve_detector_model_path()`：找到 YOLOv8 检测权重。

为什么要支持多个位置：

- 有的学生数据在 `data/oxford_pet_split/train`。
- 有的学生数据在 `data/train`。
- 有的模型叫 `models/oxford_pet_mobilenet_epoch1.pth`，有的叫 `finetuned_mobilenet.pth`。
- 代码多检查几个常见位置，可以减少课堂环境差异造成的问题。

### 第 82-88 行：`load_class_names()` 读取类别名

```python
@lru_cache(maxsize=1)
def load_class_names() -> tuple[str, ...]:
    train_dir = resolve_data_root() / "train"
    dataset = ImageFolder(train_dir)
    return tuple(dataset.classes)
```

小白解释：

- `ImageFolder` 会把 `train` 下面的子文件夹名当成类别名。
- 例如 `train/cat`、`train/dog`，类别就是 `cat`、`dog`。
- `lru_cache(maxsize=1)` 表示只读取一次，后面直接复用结果。

和前面数据集课程联系：

- 前面整理 `train/val` 文件夹不是形式主义。
- 模型训练和网页推理都依赖这个目录结构。

### 第 111-138 行：分类模型和检测模型的单例加载

```python
def get_model():
    global _model, _metadata
    if _model is None:
        class_names = load_class_names()
        model_path = resolve_model_path()
        loader = FineTunedLoader(model_path, num_classes=len(class_names))
        _model = loader.load_model()
    return _model

def get_detector(conf_threshold: float = DEFAULT_DETECTION_THRESHOLD):
    global _detector, _detector_threshold
    if _detector is None or _detector_threshold != conf_threshold:
        _detector = YOLOv8Detector(...)
        _detector_threshold = conf_threshold
    return _detector
```

小白解释：

- `get_model()` 管分类模型。
- `get_detector()` 管检测模型。
- 第一次调用时加载模型，后面直接返回已加载的模型。
- 检测阈值变化时，会重新创建检测器。

和 CPSC210 联系：

- 这是对象复用和职责分离。
- Flask 路由不直接关心模型怎么加载，只调用 `get_model()` 或 `get_detector()`。

### 第 141-179 行：分类路径，来自第六课

```python
def preprocess_image(image_path: Path) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    pipeline = transforms.Compose([...])
    return pipeline(image).unsqueeze(0)

def predict_image(image_path: Path, topk: int = 3):
    model = get_model()
    class_names = load_class_names()
    image_tensor = preprocess_image(image_path)
    with torch.no_grad():
        logits = model(image_tensor)
        probs = F.softmax(logits[0], dim=0)
        top_prob, top_idx = torch.topk(...)
```

小白解释：

- `preprocess_image()` 把用户上传的图片变成模型输入。
- `Resize`、`CenterCrop`、`Normalize` 都是为了匹配预训练模型的输入习惯。
- `predict_image()` 执行分类推理。
- `softmax` 把模型输出变成概率。
- `topk` 取概率最高的前 3 个类别。

和第七课联系：

- 这段没有被删除，因为分类功能还在。
- 第七课是在它旁边新增检测路径，而不是替换它。

### 第 182-191 行：检测路径，新增核心函数

```python
def detect_image(
    image_path: Path,
    conf_threshold: float = DEFAULT_DETECTION_THRESHOLD,
) -> tuple[list[dict[str, float | str]], Path]:
    detector = get_detector(conf_threshold)
    detections = detector.detect(image_path)
    rendered_path = draw_detections(image_path, detections, output_dir=UPLOAD_FOLDER)
    return [detection.to_dict() for detection in detections], rendered_path
```

逐段解释：

- `get_detector(conf_threshold)`：拿到 YOLOv8 检测器。
- `detector.detect(image_path)`：执行目标检测。
- `draw_detections(...)`：把检测框画到图片上。
- `to_dict()`：把检测结果转成 JSON 友好的格式。
- 返回两部分：检测结果列表、带框图片路径。

文件协作关系：

```text
web_app.py.detect_image()
  -> object_detector.py.YOLOv8Detector.detect()
  -> preprocess.py.draw_detections()
  -> templates/index.html.renderDetections()
```

课堂讲法：

> `detect_image()` 是第七课检测闭环的中转站：它把模型推理和图片可视化连起来。

### 第 194-205 行：解析检测阈值

```python
def parse_conf_threshold() -> float:
    raw_value = request.form.get("conf_threshold", str(DEFAULT_DETECTION_THRESHOLD))
    try:
        value = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("置信度阈值必须是 0 到 1 之间的数字。") from exc

    if not 0.0 <= value <= 1.0:
        raise ValueError("置信度阈值必须在 0 到 1 之间。")
    return value
```

小白解释：

- 前端滑块传过来的阈值本质上是字符串。
- 后端要把它转成浮点数。
- 如果不是数字，或者不在 0 到 1 之间，就报错。

和课堂概念联系：

- 阈值越低，检测框可能越多。
- 阈值越高，结果更严格，但可能漏检。

### 第 208-232 行：保存上传图片并生成浏览器 URL

```python
def save_upload(upload) -> Path:
    original_name = Path(upload.filename or "").name
    filename = secure_filename(original_name)
    if not filename:
        raise ValueError("文件名为空。")
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError("只支持 JPG、PNG、BMP、WEBP 等常见图片格式。")

    target = UPLOAD_FOLDER / f"{uuid4().hex}_{filename}"
    upload.save(target)
    return target

def image_url(image_path: Path) -> str:
    return url_for("static", filename=f"uploads/{image_path.name}")
```

小白解释：

- `save_upload()` 把浏览器上传的图片保存到 `static/uploads`。
- `secure_filename()` 防止危险文件名。
- `uuid4()` 防止不同学生上传同名图片互相覆盖。
- `image_url()` 把服务器上的文件路径变成浏览器能访问的 URL。

和前端联系：

- 前端上传的是文件。
- 后端保存后返回 URL。
- 前端用这个 URL 展示图片。

### 第 237-248 行：首页路由 `/`

```python
@app.route("/")
def index():
    class_names = load_class_names()
    model_path = resolve_model_path()
    return render_template(
        "index.html",
        model_path=str(model_path.relative_to(PROJECT_ROOT)),
        detector_model=resolve_detector_model_path(),
        data_root=str(resolve_data_root().relative_to(PROJECT_ROOT)),
        class_count=len(class_names),
        detection_threshold=DEFAULT_DETECTION_THRESHOLD,
    )
```

小白解释：

- 用户打开 `http://127.0.0.1:5000` 时，会进入这个函数。
- 它返回 `templates/index.html` 页面。
- 同时把模型路径、数据路径、类别数量、检测阈值传给页面。

和前面问题联系：

- 如果报 `TemplateNotFound: index.html`，说明模板路径或运行目录有问题。
- 如果报 `No fine-tuned checkpoint found`，说明分类权重没准备好。

### 第 250-271 行：分类接口 `/predict`

```python
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "没有上传文件。"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "文件名为空。"}), 400

    image_path = save_upload(file)
    results = predict_image(image_path, topk=3)
    return jsonify({
        "results": results,
        "image_url": image_url(image_path),
    })
```

小白解释：

- 这个接口是第六课核心。
- 它接收上传图片，保存图片，调用分类模型，返回 Top-3。
- 前端 `renderClassification()` 会读取 `results` 并显示概率条。

和第七课联系：

- 它保留了第六课能力。
- 第七课新增 `/detect` 后，学生可以比较两个接口的共同点和不同点。

### 第 273-297 行：检测接口 `/detect`

```python
@app.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return jsonify({"error": "没有上传文件。"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "文件名为空。"}), 400

    conf_threshold = parse_conf_threshold()
    image_path = save_upload(file)
    detections, rendered_path = detect_image(image_path, conf_threshold)
    return jsonify({
        "detections": detections,
        "count": len(detections),
        "image_url": image_url(rendered_path),
        "source_image_url": image_url(image_path),
        "conf_threshold": conf_threshold,
        "detector_model": resolve_detector_model_path(),
    })
```

逐段解释：

- 前半段和 `/predict` 一样，都是检查并保存上传文件。
- `parse_conf_threshold()` 读取前端滑块传来的阈值。
- `detect_image()` 运行 YOLOv8 并画框。
- 返回 JSON，其中包括检测列表、检测数量、带框图片 URL、原图 URL、阈值和模型路径。

和前端联系：

- `templates/index.html` 的 `uploadFile()` 会在检测模式下请求 `/detect`。
- `renderDetections()` 会读取这里返回的 `detections` 和 `image_url`。

课堂对比：

| 接口 | 调用模型 | 返回重点 |
|---|---|---|
| `/predict` | MobileNetV3 分类模型 | Top-3 类别概率 |
| `/detect` | YOLOv8 检测模型 | 检测框、类别、置信度、带框图片 |

### 第 300-318 行：命令行启动入口

```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Flask image recognition app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
```

小白解释：

- 这段让我们可以用 `python app/web_app.py` 启动项目。
- `--port` 可以换端口，比如 `--port 5001`。
- `--debug` 可以开启 Flask 调试模式。

课堂操作：

```bash
python app/web_app.py
python app/web_app.py --port 5001
python app/web_app.py --debug
```

## 五、`templates/index.html` 逐段讲解

文件路径：`templates/index.html`

这是前端页面。学生要理解：HTML 页面不直接运行 AI 模型，它只是收集用户操作，然后通过 HTTP 请求调用 Flask 后端。

### 第 1-7 行：HTML 基础结构

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AI 图像识别系统</title>
```

小白解释：

- `lang="zh-CN"` 表示中文页面。
- `UTF-8` 防止中文乱码。
- `viewport` 让页面在手机和电脑上都能正常缩放。

和之前问题联系：

- 如果浏览器中文乱码，优先检查 `meta charset="UTF-8"` 和文件保存编码。

### 第 8-287 行：CSS 样式

这一大段负责页面外观。可以分组讲，不需要逐行背。

| CSS 区域 | 作用 |
|---|---|
| `:root` | 定义颜色变量，比如背景色、强调色、错误色 |
| `body` / `.page` | 控制整体页面宽度、字体、背景 |
| `.tabs` / `.tab` | 控制分类和检测两个切换按钮 |
| `.layout` | 左右两栏布局 |
| `.panel` | 白色内容面板 |
| `.dropzone` | 图片拖拽上传区域 |
| `.controls` | 检测阈值滑块区域 |
| `.preview` / `.result-image` | 原图预览和检测后图片 |
| `.status` | 展示加载、成功、错误状态 |
| `.result` | 结果卡片 |
| `@media` | 小屏幕时改成单栏布局 |

和第六课联系：

- 第六课页面只需要上传和分类结果。
- 第七课新增了 Tab、检测阈值、检测图片展示，所以 CSS 也变多了。

课堂提醒：

- CSS 不负责模型推理。
- CSS 只负责“看起来怎么样”。

### 第 289-348 行：页面主体 HTML

```html
<h1>AI 图像识别系统</h1>
<div class="meta">
  数据集：{{ data_root }} · 类别数：{{ class_count }} · 分类模型：{{ model_path }}
  检测模型：{{ detector_model }} · 默认检测阈值：{{ detection_threshold }}
</div>
```

小白解释：

- `{{ data_root }}`、`{{ class_count }}` 这些是 Jinja 模板变量。
- Flask 的 `index()` 函数会把这些值传进来。
- 页面打开时，学生能看到当前使用的数据集和模型。

双模式 Tab：

```html
<button data-mode="classify">图像分类</button>
<button data-mode="detect">目标检测</button>
```

- `data-mode="classify"` 表示分类模式。
- `data-mode="detect"` 表示检测模式。
- JavaScript 会根据当前模式决定请求 `/predict` 还是 `/detect`。

上传区域：

```html
<div id="dropzone" class="dropzone">...</div>
<input id="file-input" type="file" accept="image/*" />
```

- `dropzone` 是用户看见的拖拽区域。
- `file-input` 是真正的文件选择控件，但被隐藏了。
- 点击拖拽区域时，JavaScript 会触发 `fileInput.click()`。

检测阈值区域：

```html
<input id="threshold" type="range" min="0.1" max="0.9" step="0.1" />
```

- 这是第七课新增控件。
- 它会把阈值传给后端的 `parse_conf_threshold()`。

### 第 350-381 行：获取页面元素和模式配置

```javascript
const tabs = Array.from(document.querySelectorAll('.tab'));
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
...
let currentMode = 'classify';
let selectedFile = null;

const modeConfig = {
  classify: {...},
  detect: {...}
};
```

小白解释：

- `document.getElementById()` 是从页面里找到某个元素。
- `currentMode` 记录当前是分类还是检测。
- `selectedFile` 记录当前选中的图片。
- `modeConfig` 保存不同模式下的标题、提示文案、加载文字和成功文字。

和后端联系：

- `currentMode` 决定后面 `fetch()` 请求 `/predict` 还是 `/detect`。

### 第 383-409 行：切换模式和清空结果

```javascript
function switchMode(mode) {
  currentMode = mode;
  tabs.forEach((tab) => tab.classList.toggle('active', tab.dataset.mode === mode));
  modeTitle.textContent = modeConfig[mode].title;
  dropCopy.textContent = modeConfig[mode].copy;
  detectionControls.classList.toggle('visible', mode === 'detect');
  resetResults();
  if (selectedFile) {
    uploadFile(selectedFile);
  }
}
```

小白解释：

- 切换 Tab 时，更新当前模式。
- 改变按钮高亮状态。
- 改变标题和提示文案。
- 如果是检测模式，就显示阈值滑块。
- 如果已经选过图片，切换模式后自动重新上传并调用对应接口。

课堂对比：

- 分类模式：不显示阈值。
- 检测模式：显示阈值，因为检测需要过滤置信度。

### 第 411-420 行：读取用户选择的图片并预览

```javascript
function handleFile(file) {
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (event) => {
    preview.src = event.target.result;
    preview.style.display = 'block';
  };
  reader.readAsDataURL(file);
  uploadFile(file);
}
```

小白解释：

- `FileReader` 可以在浏览器里读取本地图片。
- `preview.src = ...` 让页面先显示用户选的原图。
- 最后调用 `uploadFile(file)`，把图片传给后端。

和后端联系：

- 前端预览不等于模型推理。
- 真正的 AI 推理发生在 Flask 后端。

### 第 422-458 行：上传图片并调用 Flask 接口

```javascript
function uploadFile(file) {
  const formData = new FormData();
  formData.append('file', file);
  if (currentMode === 'detect') {
    formData.append('conf_threshold', threshold.value);
  }

  fetch(currentMode === 'detect' ? '/detect' : '/predict', {
    method: 'POST',
    body: formData
  })
  ...
}
```

逐段解释：

- `FormData` 用来模拟表单上传文件。
- 一定会传 `file`。
- 检测模式额外传 `conf_threshold`。
- `fetch()` 发送 HTTP POST 请求。
- 如果当前模式是检测，调用 `/detect`。
- 如果当前模式是分类，调用 `/predict`。

和 `web_app.py` 对应关系：

| 前端请求 | 后端函数 |
|---|---|
| `fetch('/predict')` | `@app.route("/predict") def predict()` |
| `fetch('/detect')` | `@app.route("/detect") def detect()` |

课堂讲法：

> 前端不是直接调用 Python 函数，而是通过 HTTP 请求调用 Flask 暴露出来的接口。

### 第 460-473 行：渲染分类结果

```javascript
function renderClassification(payload) {
  resultsBox.innerHTML = payload.results.map((item, index) => {
    const percent = Math.max(0, Math.min(100, item.score));
    return `
      <div class="result ${index === 0 ? 'top' : ''}">
        <strong>${index + 1}. ${escapeHtml(item.label)}</strong>
        <span>${percent.toFixed(2)}%</span>
      </div>
    `;
  }).join('');
}
```

小白解释：

- `payload.results` 来自后端 `/predict`。
- 每个 `item` 包含 `label` 和 `score`。
- `map()` 把每个结果变成一段 HTML。
- `index === 0` 表示第一名结果，样式更突出。

和第六课联系：

- 这就是第六课 Top-3 分类结果展示逻辑。
- 第七课仍然保留，说明系统支持旧功能。

### 第 475-507 行：渲染检测结果

```javascript
function renderDetections(payload) {
  resultImage.src = payload.image_url;
  resultImage.style.display = 'block';

  if (!payload.detections.length) {
    ...
    return;
  }

  resultsBox.innerHTML = payload.detections.map((item, index) => {
    const percent = Math.max(0, Math.min(100, item.confidence));
    return `
      <strong>${index + 1}. ${escapeHtml(item.label)}</strong>
      <span>${percent.toFixed(2)}%</span>
      框坐标：(${item.x1}, ${item.y1}) - (${item.x2}, ${item.y2})
    `;
  }).join('');
}
```

小白解释：

- `payload.image_url` 是后端生成的带框图片。
- 如果没有检测到目标，页面提示学生可以降低阈值。
- 如果检测到目标，就显示类别、置信度和框坐标。

和后端联系：

- `payload.detections` 来自 `DetectionResult.to_dict()`。
- `payload.image_url` 来自 `draw_detections()` 保存后的图片路径。

和第七课概念联系：

- 分类结果主要看 `label + score`。
- 检测结果要看 `label + confidence + box`。

### 第 509-555 行：安全转义和事件绑定

```javascript
function escapeHtml(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

tabs.forEach((tab) => {
  tab.addEventListener('click', () => switchMode(tab.dataset.mode));
});
...
clearBtn.addEventListener('click', resetAll);
```

小白解释：

- `escapeHtml()` 防止把不安全内容直接插进页面。
- `addEventListener()` 给按钮、滑块、拖拽区域绑定行为。
- 点击 Tab 会切换模式。
- 改变阈值会重新上传图片并检测。
- 点击清空会重置页面。

课堂提醒：

- HTML 是页面结构。
- CSS 是页面样式。
- JavaScript 是页面交互。
- Flask 是后端推理服务。

## 六、`requirements.txt` 逐段讲解

文件路径：`requirements.txt`

```text
torch
torchvision
tqdm
opencv-python
requests
Pillow
Flask
ultralytics
```

逐项解释：

- `torch`：PyTorch 主库，负责模型张量计算。
- `torchvision`：提供 MobileNetV3、图像预处理和 ImageFolder。
- `tqdm`：训练时显示进度条。
- `opencv-python`：第七课新增重点，用来画检测框。
- `requests`：项目中可能用于下载或网络请求。
- `Pillow`：读取和处理图片。
- `Flask`：Web 后端。
- `ultralytics`：第七课新增重点，提供 YOLOv8。

课堂联系：

- 如果只做第六课分类，核心是 `torch`、`torchvision`、`Pillow`、`Flask`。
- 第七课做检测，必须有 `ultralytics` 和 `opencv-python`。

## 七、把四个主要文件串成一条线

### 分类路径

```text
templates/index.html
  uploadFile()
  fetch('/predict')
        |
        v
app/web_app.py
  predict()
  save_upload()
  predict_image()
        |
        v
app/model_loader.py
  FineTunedLoader
  MobileNetV3
        |
        v
web_app.py 返回 JSON
        |
        v
templates/index.html
  renderClassification()
```

第六课重点就在这条路径。

### 检测路径

```text
templates/index.html
  uploadFile()
  fetch('/detect')
  额外提交 conf_threshold
        |
        v
app/web_app.py
  detect()
  parse_conf_threshold()
  save_upload()
  detect_image()
        |
        v
app/object_detector.py
  YOLOv8Detector.detect()
  返回 DetectionResult 列表
        |
        v
app/preprocess.py
  draw_detections()
  保存带框图片
        |
        v
web_app.py 返回 JSON
        |
        v
templates/index.html
  renderDetections()
```

第七课重点就在这条路径。

## 八、课堂讲解顺序建议

不要按文件在目录里的顺序讲，建议按学生理解路径讲：

1. 先讲用户看到什么：`templates/index.html` 页面有分类和检测两个 Tab。
2. 再讲前端怎么发请求：分类发 `/predict`，检测发 `/detect`。
3. 再讲 Flask 怎么接请求：`web_app.py` 里两个 route。
4. 再讲检测怎么实现：`object_detector.py` 封装 YOLOv8。
5. 再讲结果怎么变成图片：`preprocess.py` 画框。
6. 最后讲依赖：`requirements.txt` 为什么要加 `ultralytics` 和 `opencv-python`。

## 九、学生容易混淆的点

| 容易混淆的问题 | 正确理解 |
|---|---|
| 分类和检测是不是同一个模型 | 不是。分类用微调后的 MobileNetV3，检测用 YOLOv8。 |
| 前端是不是在运行模型 | 不是。前端只上传图片和展示结果，模型在 Flask 后端运行。 |
| `DetectionResult` 是不是模型 | 不是。它只是保存一个检测结果的数据对象。 |
| `draw_detections()` 是不是检测 | 不是。它只负责把检测结果画出来。 |
| 阈值越高是不是越好 | 不一定。阈值高会减少误检，但也可能漏检。 |
| `/predict` 和 `/detect` 哪个是第七课新增 | `/detect` 是第七课新增，`/predict` 是第六课保留。 |

## 十、可以让学生复述的标准答案

第七课新增的目标检测功能是这样工作的：

用户在网页选择目标检测模式并上传图片，前端 JavaScript 把图片和置信度阈值通过 `fetch('/detect')` 发给 Flask。`web_app.py` 的 `/detect` 接口接收请求后，先保存图片，再读取阈值，然后调用 `detect_image()`。`detect_image()` 会通过 `YOLOv8Detector` 执行检测，得到若干个 `DetectionResult`，再调用 `draw_detections()` 把检测框画到原图上。最后 Flask 把检测结果和带框图片地址以 JSON 返回给前端，前端用 `renderDetections()` 显示图片、类别、置信度和框坐标。

这和第六课的分类功能形成对比：第六课的 `/predict` 返回 Top-3 类别，第七课的 `/detect` 返回目标位置、类别和带框图片。两个功能共用同一个 Web 系统，但调用的是不同模型和不同处理流程。
