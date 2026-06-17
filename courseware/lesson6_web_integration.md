# 第六课时：Flask Web界面开发与系统集成

> 基于当前项目结构，并结合最新课件重写的第六课内容  
> 关联代码：`app/model_loader.py`、`app/classifier.py`、`app/transforms.py`、`app/compare_inference.py`、`app/web_app.py`

---

## Slide 1：本课时目标

标题：给 AI 能力层穿上 Web 的外衣

今天你要完成 5 件事：
1. 理解 Flask 的路由、视图函数、模板渲染和 JSON 响应。
2. 把现有图像识别能力封装成一个可被 Web 调用的服务。
3. 用单例思想管理模型实例，保证模型在整个进程中只加载一次。
4. 完成“上传图片 -> 预处理 -> 推理 -> 结果展示”的闭环。
5. 让前端具备加载中、成功、失败三种状态反馈。

讲师提示：
- 本课不是前端美工课，重点是系统集成。
- 页面只需要清晰、能用、能讲透流程。
- 代码尽量保持简单，方便后续第 7 课继续扩展到检测任务。

---

## Slide 2：为什么要做 Web 层

标题：从“能跑”到“能用”

当前项目已经有了这些能力：
- `app/classifier.py`：命令行图像推理。
- `app/finetune.py`：模型微调。
- `app/compare_inference.py`：微调前后对比。
- `app/model_loader.py`：模型加载与微调模型构建。

但这些能力还有一个明显缺口：
- 只能在终端里调用。
- 用户必须知道命令和路径。
- 不适合展示和交互。

因此我们需要 Web 层：
```text
浏览器上传图片
-> Flask 接收请求
-> 预处理图像
-> 调用模型推理
-> 返回 JSON
-> 前端渲染结果
```

---

## Slide 3：Web 架构设计

标题：MVC 分层，各司其职

我们把系统拆成三层：

- Model：`app/model_loader.py`、`app/classifier.py`、`app/transforms.py`
- View：`templates/index.html`
- Controller：`app/web_app.py`

职责边界：
- Model 负责加载模型和执行推理。
- View 负责页面结构和状态展示。
- Controller 负责接收请求、组织数据、返回结果。

这样做的好处：
- 修改界面不影响推理逻辑。
- 替换模型不需要重写前端。
- 课堂讲解时层次更清楚。

---

## Slide 4：Flask 应用主体

标题：一个轻量级的 Python Web 入口

核心文件：`app/web_app.py`

推荐讲解顺序：
1. 创建 Flask 应用。
2. 初始化上传目录。
3. 用单例加载模型。
4. 定义 `/` 和 `/predict` 两个路由。
5. 前端提交图片后返回 JSON。

关键代码结构：
```python
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

@app.route("/")
def index():
    return render_template("index.html", ...)

@app.route("/predict", methods=["POST"])
def predict():
    ...
    return jsonify({"results": results, "image_url": image_url})
```

讲师备注：
- `render_template` 用来渲染 HTML 页面。
- `request.files` 用来接收上传文件。
- `jsonify` 用来返回结构化结果。

---

## Slide 5：模型加载单例

标题：模型只加载一次

Web 服务和命令行脚本不一样：
- 命令行程序跑完就结束。
- Web 服务会连续接收很多请求。

如果每次请求都重新加载模型：
- 速度会明显变慢。
- 内存占用会升高。
- 用户体验会很差。

所以我们在 `app/web_app.py` 里采用单例：
```python
_model = None

def get_model():
    global _model
    if _model is None:
        loader = FineTunedLoader(model_path, num_classes=len(class_names))
        _model = loader.load_model()
    return _model
```

对应的 loader 在 `app/model_loader.py` 里：
- `MobileNetV3Loader`
- `FineTunedLoader`

---

## Slide 6：微调模型加载

标题：把 checkpoint 变成可用模型

新增的 loader 负责三步：
1. 构建 MobileNetV3 结构。
2. 替换最后一层分类头。
3. 加载 checkpoint 权重。

关键代码：
```python
class FineTunedLoader(ModelLoader):
    def __init__(self, model_path, num_classes, freeze_features=True):
        ...

    def load_model(self):
        model = build_finetune_model(
            num_classes=self.num_classes,
            freeze_features=self.freeze_features,
        )
        state_dict = torch.load(self.model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        return model
```

课堂要点：
- 这里不是重新训练，而是加载已经训练好的权重。
- `num_classes` 必须和训练数据一致。
- `eval()` 很重要，推理时要关闭训练行为。

---

## Slide 7：数据集与类别名

标题：类别名来自文件夹

当前项目使用的是 `ImageFolder` 目录结构：

```text
data/oxford_pet_split/
  train/
    Abyssinian/
    Bengal/
    British_Shorthair/
    ...
  val/
    Abyssinian/
    Bengal/
    British_Shorthair/
    ...
```

这样做的好处：
- 类名直接从文件夹名读取。
- 不需要额外写标签表。
- 适合课堂演示和快速扩展。

对应代码：
- `app/web_app.py` 中的 `load_class_names()`
- `app/compare_inference.py` 中的类别读取逻辑

---

## Slide 8：图像预处理

标题：上传图片先变成模型能吃的张量

Web 端上传的原始图片不能直接送进模型，必须先做预处理：

```python
transforms.Resize(256)
transforms.CenterCrop(224)
transforms.ToTensor()
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

这套处理和训练时的验证预处理保持一致：
- 尺寸稳定。
- 结果可复现。
- 便于和训练阶段对齐。

对应代码：
- `app/transforms.py`
- `app/web_app.py` 的 `preprocess_image()`

---

## Slide 9：前端页面

标题：一个轻量但完整的识别界面

核心文件：`templates/index.html`

页面功能：
- 点击上传。
- 拖拽上传。
- 图片预览。
- 识别中状态。
- Top-3 结果展示。
- 置信度条。

推荐讲解点：
- 页面不依赖前端框架。
- HTML、CSS、原生 JavaScript 就能完成闭环。
- 重点是状态切换，而不是样式堆砌。

状态设计：
```text
空闲 -> 上传中/识别中 -> 成功
                 \-> 失败
```

---

## Slide 10：请求与响应

标题：浏览器和后端如何对话

前端提交图片：
```javascript
fetch('/predict', {
  method: 'POST',
  body: formData
})
```

后端返回 JSON：
```json
{
  "results": [
    {"label": "British_Shorthair", "score": 92.14},
    {"label": "Russian_Blue", "score": 5.22}
  ],
  "image_url": "/static/uploads/demo.jpg"
}
```

课堂强调：
- 前端只关心结果，不关心模型细节。
- 后端只返回结构化数据，不直接拼复杂 HTML。
- 这样前后端耦合更低。

---

## Slide 11：项目目录结构更新

标题：第六课后，项目多了哪些东西

```text
AI-Image-Recognition/
  app/
    classifier.py
    compare_inference.py
    dataset.py
    exceptions.py
    finetune.py
    model_loader.py
    transforms.py
    train.py
    web_app.py          # 本课新增
  courseware/
    lesson6_web_integration.md
    lesson6_student_practice.md
  data/
    images/
    oxford_pet_split/
      train/
      val/
  models/
    oxford_pet_mobilenet_epoch1.pth
  static/
    uploads/
  templates/
    index.html          # 本课新增
```

讲师备注：
- 这就是一个最小可讲、最小可跑的 Web 集成版本。
- 后续第 7 课可以继续叠加检测模块。

---

## Slide 12：运行方式

标题：先确保能启动，再谈美化

推荐运行命令：

```bash
python app/web_app.py
```

如果需要指定端口：

```bash
python app/web_app.py --host 127.0.0.1 --port 5000
```

课堂演示路径：
1. 打开浏览器。
2. 访问 `http://127.0.0.1:5000`
3. 上传一张测试图片。
4. 查看识别结果和置信度。

---

## Slide 13：课时总结

标题：本课你真正打通了什么

今天完成的闭环：
1. 浏览器上传图片。
2. Flask 接收请求。
3. 模型单例加载。
4. 图像预处理。
5. 推理和 Top-K 输出。
6. 前端结果展示。

核心结论：
- 模型能力要变成产品能力，必须有 Web 层。
- 单例、分层、状态管理，是这个阶段最关键的工程点。
- 第 6 课的重点不是“写一个网页”，而是“把 AI 模型接进应用”。

