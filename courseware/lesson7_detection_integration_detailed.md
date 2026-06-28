# 第七课时：把目标检测接入 Web 图像识别系统

从“判断这张图是什么”到“找出图里有什么、在哪里”

## 本节课一句话目标

前面我们已经能让网页判断一张图片属于哪个类别。本节课要让网页进一步画出图片里的目标位置，并返回目标名称、置信度和检测框。

学生最终要完成一个闭环：启动 Flask 页面，上传图片，选择分类或检测，看到模型推理结果，并能解释每个文件在这个闭环中的作用。

## 一、为什么要上这堂课

第六课我们已经把图像分类模型接入了网页。分类模型解决的问题是：给它一张图，它告诉我们这张图最像哪个类别。但真实应用里经常还要回答：

- 图里有几个物体？
- 物体分别在哪里？
- 哪些目标可信度高，哪些只是模型猜测？
- 能不能把结果直接画在图片上给用户看？

所以第七课从 `classification` 过渡到 `object detection`。一句话区别：

| 能力 | 解决的问题 | 输出 |
|---|---|---|
| 图像分类 | 这张图整体是什么 | 类别和概率 |
| 目标检测 | 图里有什么，分别在哪里 | 检测框、类别、置信度 |

## 二、和前面课程的联系

| 前面内容 | 第七课怎么用 | 学生要理解 |
|---|---|---|
| train / val 数据集 | 分类功能仍然读取类别名 | 数据目录不仅用于训练，也用于推理解释 |
| 微调 MobileNetV3 | 分类 Tab 继续使用第六课模型 | 第七课是在已有系统旁边新增检测能力 |
| Flask Web | 继续使用 `app/web_app.py` | AI 模型要通过接口给用户使用 |
| CPSC210 抽象 | `ObjectDetector` 是接口，`YOLOv8Detector` 是实现 | 面向接口编程方便以后换模型 |

## 三、项目结构

```text
AI-Image-Recognition - 1/
├─ app/
│  ├─ web_app.py
│  ├─ object_detector.py
│  ├─ preprocess.py
│  └─ model_loader.py
├─ templates/
│  └─ index.html
├─ static/uploads/
├─ data/train/ 或 data/oxford_pet_split/train/
├─ models/oxford_pet_mobilenet_epoch1.pth
├─ yolov8n.pt
└─ requirements.txt
```

## 四、核心概念

- 分类：给整张图贴标签。
- 检测：找出图中每个目标的位置和类别。
- 检测框：用矩形框标出物体区域。
- 置信度：模型对检测结果的确定程度。
- 阈值：过滤低置信度检测结果的门槛。
- JSON：后端返回给前端的结构化结果。

## 五、课堂操作步骤

1. 打开项目根目录。
2. 激活 conda 环境。

```bash
conda activate pytorch_env
```

3. 安装依赖。

```bash
pip install -r requirements.txt
```

4. 确认分类权重存在：`models/oxford_pet_mobilenet_epoch1.pth` 或 `finetuned_mobilenet.pth`。
5. 确认数据目录存在：`data/train` 或 `data/oxford_pet_split/train`。
6. 启动 Flask。

```bash
python app/web_app.py
```

7. 浏览器打开 `http://127.0.0.1:5000`。
8. 测试分类 Tab，观察 Top-3。
9. 测试检测 Tab，观察检测框、类别和置信度。
10. 调整阈值 0.3、0.5、0.7，观察检测框数量变化。

## 六、每个文件的作用

### `app/object_detector.py`

- `DetectionResult`：保存一个检测结果，包括框坐标、类别、置信度。
- `ObjectDetector`：检测器接口，规定必须有 `detect()` 方法。
- `YOLOv8Detector`：YOLOv8 的具体实现，负责加载 `yolov8n.pt` 并执行检测。

### `app/preprocess.py`

- 读取原图。
- 根据检测结果画矩形框。
- 写上类别和置信度。
- 保存带框图片给网页展示。

### `app/web_app.py`

- `/predict`：第六课分类接口。
- `/detect`：第七课检测接口。
- `get_model()`：分类模型只加载一次。
- `get_detector()`：检测模型只加载一次。
- `detect_image()`：检测并生成带框图片。

### `templates/index.html`

- 分类和检测两个 Tab。
- 图片上传和预览。
- 阈值滑块。
- 使用 `fetch()` 调用 Flask 接口。
- 把 JSON 结果渲染到页面。

## 七、常见问题

| 现象 | 原因 | 解决 |
|---|---|---|
| Internal Server Error | 后端异常 | 看终端 Traceback 最后三行 |
| No fine-tuned checkpoint found | 分类权重缺失 | 放入 `.pth` 权重文件 |
| ModuleNotFoundError: ultralytics | 依赖没装到当前环境 | 激活环境后重新 `pip install -r requirements.txt` |
| 第一次检测慢 | 首次下载或加载 YOLOv8 | 等待下载完成，或提前放好 `yolov8n.pt` |
| TemplateNotFound | 模板路径不对或文件缺失 | 确认从项目根目录运行，且 `templates/index.html` 存在 |

## 八、课后作业

| 作业 | 要求 |
|---|---|
| 基础复现 | 分别提交分类截图和检测截图 |
| 阈值观察 | 用同一张图测试 0.3、0.5、0.7 |
| 代码解释 | 用自己的话解释 `/detect` 完整流程 |
| 错误排查 | 解释一个 500 错误的原因和解决方案 |
| 拓展挑战 | 尝试按置信度排序检测结果 |

## 九、本节课总结

第六课解决“这张图是什么”，第七课解决“图里有什么、在哪里”。学生要掌握的不只是 YOLOv8，而是一个完整 AI Web 系统如何把模型能力交给用户使用。
