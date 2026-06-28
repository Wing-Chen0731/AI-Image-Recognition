# 第七课时自我实操手册：从启动到完成目标检测

这份手册给同学课后自己复现使用。请按顺序操作，不要跳步。每一步完成后再进入下一步。

## 一、实操目标

完成一个完整的 AI Web 检测系统运行闭环：

```text
打开项目
-> 激活 conda 环境
-> 安装依赖
-> 确认模型和数据存在
-> 启动 Flask
-> 浏览器上传图片
-> 测试分类
-> 测试目标检测
-> 调节检测阈值
-> 记录结果
```

## 二、准备工作

### 1. 打开项目根目录

在 VS Code 中打开项目：

```text
AI-Image-Recognition - 1
```

终端路径应该类似：

```text
/c/Vscode/AI-Image-Recognition - 1
```

或者 Windows PowerShell 中类似：

```text
C:\Vscode\AI-Image-Recognition - 1
```

注意：不要只打开 `app` 文件夹，否则 Flask 可能找不到 `templates/index.html`。

## 三、激活 conda 环境

在 VS Code 终端执行：

```bash
conda activate pytorch_env
```

成功后，终端前面应该出现：

```text
(pytorch_env)
```

如果你的环境名字不是 `pytorch_env`，就换成自己的环境名。

## 四、安装项目依赖

执行：

```bash
pip install -r requirements.txt
```

如果看到很多：

```text
Requirement already satisfied
```

说明依赖已经安装过，不是错误。

第七课重点依赖包括：

```text
Flask
torch
torchvision
opencv-python
ultralytics
```

其中：

- `Flask`：启动 Web 后端。
- `torch`、`torchvision`：运行分类模型。
- `ultralytics`：运行 YOLOv8。
- `opencv-python`：把 YOLO 检测结果画到图片上。

## 五、确认分类模型权重存在

第七课虽然新增了 YOLO 检测，但页面首页仍然会加载第六课分类信息，所以分类权重也要存在。

在项目根目录执行：

```bash
python -c "from pathlib import Path; print('models/oxford_pet_mobilenet_epoch1.pth:', Path('models/oxford_pet_mobilenet_epoch1.pth').exists()); print('finetuned_mobilenet.pth:', Path('finetuned_mobilenet.pth').exists())"
```

如果至少有一个输出是：

```text
True
```

就可以继续。

如果两个都是：

```text
False
```

说明缺少分类模型权重，打开首页可能报：

```text
No fine-tuned checkpoint found
```

需要把训练好的 `.pth` 文件放到下面任意一个位置：

```text
models/oxford_pet_mobilenet_epoch1.pth
finetuned_mobilenet.pth
```

## 六、确认数据目录存在

执行：

```bash
python -c "from pathlib import Path; print('data/oxford_pet_split/train:', Path('data/oxford_pet_split/train').exists()); print('data/train:', Path('data/train').exists())"
```

如果至少有一个是：

```text
True
```

说明数据目录存在。

如果两个都是 `False`，首页可能报：

```text
No dataset split found
```

需要检查数据是否放错位置。

## 七、确认 YOLOv8 权重

执行：

```bash
python -c "from pathlib import Path; print('yolov8n.pt:', Path('yolov8n.pt').exists())"
```

如果输出：

```text
True
```

说明 YOLOv8 权重已经在项目根目录。

如果输出：

```text
False
```

第一次运行检测时，`ultralytics` 可能会尝试自动下载 `yolov8n.pt`。如果网络不稳定，可能需要老师提前提供这个文件。

## 八、启动 Flask 项目

执行：

```bash
python app/web_app.py
```

看到类似输出：

```text
* Serving Flask app 'web_app'
* Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment.
* Running on http://127.0.0.1:5000
```

说明启动成功。

这里的 warning 不是错误。它只是提醒：当前是开发服务器，适合本地课堂运行，不适合正式上线部署。

## 九、打开浏览器

在浏览器地址栏输入：

```text
http://127.0.0.1:5000
```

如果页面正常打开，说明 Flask 首页可访问。

如果浏览器显示：

```text
Internal Server Error
```

不要只看浏览器页面，要回到终端看最后几行 Traceback。

## 十、测试图像分类

操作：

1. 确认页面处于“图像分类”模式。
2. 点击“选择图片”，或把图片拖到上传区域。
3. 上传一张猫或狗图片。
4. 等待页面显示 Top-3 分类结果。

需要记录：

```text
Top-1 类别：
Top-1 概率：
Top-2 类别：
Top-3 类别：
```

如果分类失败，优先检查：

- 分类模型 `.pth` 是否存在。
- 数据目录 `train` 是否存在。
- 当前 conda 环境是否正确。

## 十一、测试目标检测

操作：

1. 点击页面上方的“目标检测”。
2. 上传一张包含常见物体的图片。
3. 等待页面显示带框图片。
4. 观察检测结果列表。

建议使用包含这些物体的图片：

- 人
- 猫
- 狗
- 车
- 椅子
- 杯子

因为 `yolov8n.pt` 是通用 COCO 预训练模型，它更擅长识别常见物体。

需要记录：

```text
检测到几个目标：
第一个目标类别：
第一个目标置信度：
第一个目标框坐标：
```

## 十二、调节检测阈值

这个操作在网页里完成，不是在终端里完成。

步骤：

1. 点击“目标检测”。
2. 上传同一张图片。
3. 找到“检测置信度阈值”滑块。
4. 分别调到：

```text
0.3
0.5
0.7
```

5. 每次观察检测框数量。

填写表格：

| 阈值 | 检测框数量 | 我的观察 |
|---|---:|---|
| 0.3 |  |  |
| 0.5 |  |  |
| 0.7 |  |  |

理解：

- 阈值低：更容易显示检测框，可能检测更多，也可能误检更多。
- 阈值高：结果更严格，框可能更少，也可能漏检。

## 十三、保存截图

至少保存三张截图：

1. Flask 终端启动成功截图。
2. 图像分类结果截图。
3. 目标检测带框结果截图。

如果做了阈值对比，建议额外保存：

- 阈值 `0.3` 截图。
- 阈值 `0.5` 截图。
- 阈值 `0.7` 截图。

## 十四、常见问题排查

### 问题 1：右下角提示创建虚拟环境，需要管吗？

如果你已经看到终端前面有：

```text
(pytorch_env)
```

并且安装路径类似：

```text
C:\Users\33768\miniconda3\envs\pytorch_env\Lib\site-packages
```

说明依赖已经装进 conda 环境，可以忽略 VS Code 的创建虚拟环境提示。

### 问题 2：Flask development server warning

看到：

```text
WARNING: This is a development server.
```

不用解决。课堂本地运行正常。

### 问题 3：`No fine-tuned checkpoint found`

说明缺分类模型权重。

检查：

```text
models/oxford_pet_mobilenet_epoch1.pth
finetuned_mobilenet.pth
```

### 问题 4：`ModuleNotFoundError: ultralytics`

说明没有安装 YOLOv8 依赖。

解决：

```bash
conda activate pytorch_env
pip install -r requirements.txt
```

### 问题 5：检测不到目标

可能原因：

- 阈值太高。
- 图片太模糊。
- 图片里的物体不是 YOLOv8 预训练模型认识的常见类别。

可以尝试：

- 把阈值调到 `0.3`。
- 换一张包含人、猫、狗、车的清晰图片。

### 问题 6：`TemplateNotFound: index.html`

可能原因：

- 没有从项目根目录运行。
- `templates/index.html` 文件不存在。

正确方式：

```bash
cd "C:\Vscode\AI-Image-Recognition - 1"
python app/web_app.py
```

## 十五、实操完成后的自查清单

完成后逐项打勾：

```text
[ ] 我能激活 conda 环境
[ ] 我能安装 requirements.txt
[ ] 我确认分类模型权重存在
[ ] 我确认数据目录存在
[ ] 我能启动 Flask
[ ] 我能打开 http://127.0.0.1:5000
[ ] 我能完成图像分类
[ ] 我能完成目标检测
[ ] 我能调节阈值并解释变化
[ ] 我能解释 YOLO 和 OpenCV 的分工
```

## 十六、最终提交内容

请提交：

1. 分类结果截图。
2. 检测结果截图。
3. 阈值对比表。
4. 一段 100-200 字说明：

```text
第七课中，YOLOv8 负责什么，OpenCV 负责什么，Flask 和前端页面分别负责什么。
```

