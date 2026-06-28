# 第七课时：YOLOv8 目标检测拓展与系统联调优化

> 本课基于第六课 Flask 图像分类 Web 系统继续扩展。目标是让同一个页面同时支持“图像分类”和“目标检测”，完成分类 + 检测双模型闭环。

---

## 一、本课目标

本课完成 5 件事：

1. 理解图像分类和目标检测的区别。
2. 使用 Ultralytics YOLOv8 加载 `yolov8n.pt` 预训练模型。
3. 解析检测结果：边界框、类别名、置信度。
4. 使用 OpenCV 在原图上画框并保存结果图。
5. 将检测功能集成到 Flask Web，形成双 Tab 应用。

---

## 二、第七课新增和更新的项目文件

新增文件：

```text
app/object_detector.py
app/preprocess.py
courseware/lesson7_detection_integration.md
```

更新文件：

```text
app/web_app.py
templates/index.html
requirements.txt
README.md
```

---

## 三、分类和检测的区别

| 任务 | 输入 | 输出 | 适合场景 |
|---|---|---|---|
| 图像分类 | 一张图片 | 整张图最像哪个类别 | 宠物品种识别、商品分类 |
| 目标检测 | 一张图片 | 每个物体的位置、类别、置信度 | 安防、自动驾驶、工业质检 |

分类回答：

```text
这张图是什么？
```

检测回答：

```text
图里有哪些物体？它们分别在哪里？
```

---

## 四、后端检测闭环

检测链路如下：

```text
浏览器上传图片
-> Flask /detect 接收文件
-> 保存到 static/uploads/
-> YOLOv8Detector.detect() 推理
-> draw_detections() 用 OpenCV 画框
-> 返回 JSON
-> 前端展示带框图片和检测列表
```

核心文件：

- `app/object_detector.py`: 定义 `ObjectDetector` 抽象接口和 `YOLOv8Detector` 实现。
- `app/preprocess.py`: 定义 `draw_detections()`，把检测框画到原图上。
- `app/web_app.py`: 新增 `/detect` 路由和检测器单例。

---

## 五、前端双 Tab 闭环

`templates/index.html` 现在有两个 Tab：

```text
图像分类
目标检测
```

分类 Tab：

```text
上传图片 -> 调用 /predict -> 显示 Top-3 分类结果
```

检测 Tab：

```text
上传图片 -> 调用 /detect -> 显示带框图片和检测列表
```

检测 Tab 额外提供置信度阈值滑块。课堂上可以让学生分别尝试：

```text
0.3
0.5
0.7
```

观察检测框数量变化。

---

## 六、运行方式

第七课仍然建议在 conda 环境中运行。

```bash
conda activate pytorch_env
pip install -r requirements.txt
python app/web_app.py
```

打开：

```text
http://127.0.0.1:5000
```

首次使用检测功能时，Ultralytics 可能会自动下载：

```text
yolov8n.pt
```

这是 YOLOv8 nano 预训练权重，已经被 `.gitignore` 忽略，不需要提交到 GitHub。

---

## 七、学生必须验证的闭环

学生至少完成以下验证：

1. 分类 Tab 能上传图片并返回 Top-3。
2. 检测 Tab 能上传图片并返回带框图片。
3. 检测结果列表中能看到类别名和置信度。
4. 阈值从 0.3 改到 0.7 后，检测框数量可能发生变化。
5. 上传非图片文件时，后端能返回错误提示，而不是直接崩溃。

---

## 八、CPSC 210 对应关系

| CPSC 210 概念 | 本课对应代码 |
|---|---|
| Abstraction | `ObjectDetector` 抽象接口 |
| Dependency Inversion | Web 层依赖检测接口，而不是直接依赖 YOLO 细节 |
| Open-Closed Principle | 以后换检测模型时新增类，不需要重写 Web 层 |
| Liskov Substitution | `YOLOv8Detector` 可以替换 `ObjectDetector` 被调用 |
| Single Responsibility | 检测、画框、Web 路由分别放在不同文件里 |

---

## 九、课后任务

必做：

1. 跑通检测功能，截图分类 Tab 和检测 Tab 的结果。
2. 使用同一张图分别设置阈值 0.3、0.5、0.7，记录检测框数量。
3. 用 CPSC 210 术语解释为什么 `ObjectDetector` 设计方便以后替换 YOLOv9 或 YOLOv10。

选做：

1. 找一个检测数据集，了解 YOLO 格式标签。
2. 尝试用 Ultralytics 的 `model.train()` 微调一个自定义检测模型。

---

## 十、本课总结

第七课的核心不是“多加一个模型”，而是把新的 AI 能力用清晰的接口接入已有系统。完成后，项目从单一分类应用升级为分类 + 检测双功能 Web 应用。
