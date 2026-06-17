# 第六课时学生实操课件：Flask Web 界面开发与系统集成

> 适用对象：已经完成第 4 课微调和第 5 课推理整合的同学  
> 实操目标：跑通一个能上传图片、返回分类结果的 Flask Web 页面

---

## 一、这节课你要完成什么

你要完成 4 件事：
1. 看懂 Web 版项目结构，知道新增了哪些文件。
2. 跑通 Flask Web 服务。
3. 上传一张图片，得到 Top-3 分类结果。
4. 理解为什么模型要用单例加载。

---

## 二、课前准备

先确认项目根目录打开正确：

```text
C:\Vscode\AI-Image-Recognition - 1
```

再确认依赖已经安装：
- `torch`
- `torchvision`
- `Flask`
- `tqdm`

如果缺少 Flask，请先安装依赖：

```bash
pip install -r requirements.txt
```

---

## 三、先看项目结构

这节课新增的关键文件：

```text
app/web_app.py
templates/index.html
static/uploads/
```

你还会继续使用这些旧文件：
- `app/model_loader.py`
- `app/transforms.py`
- `app/classifier.py`
- `app/compare_inference.py`

---

## 四、先理解 Web 服务是怎么工作的

一句话版本：

> 用户上传图片，Flask 接住文件，模型做预测，前端显示结果。

完整流程：

```text
浏览器上传图片
-> Flask 路由 /predict 接收文件
-> 保存到 static/uploads/
-> 图像预处理
-> 模型推理
-> 返回 JSON
-> 页面渲染结果
```

---

## 五、先运行一次 Web 服务

在项目根目录执行：

```bash
python app/web_app.py
```

如果一切正常，你会看到类似输出：

```text
 * Running on http://127.0.0.1:5000
```

然后在浏览器打开：

```text
http://127.0.0.1:5000
```

---

## 六、页面上你应该看到什么

页面至少要有这些元素：
- 图片上传区
- 图片预览区
- 识别状态
- 结果展示区

上传一张图片后，页面应该出现：
1. 预览图。
2. “识别中...” 状态。
3. Top-3 结果。
4. 每个类别对应的置信度。

---

## 七、这节课的核心代码你要看懂

### 1. 单例模型加载

```python
_model = None

def get_model():
    global _model
    if _model is None:
        loader = FineTunedLoader(model_path, num_classes=len(class_names))
        _model = loader.load_model()
    return _model
```

### 2. 图片预处理

```python
def preprocess_image(image_path):
    ...
    return pipeline(image).unsqueeze(0)
```

### 3. 推理接口

```python
@app.route('/predict', methods=['POST'])
def predict():
    ...
    return jsonify({'results': results, 'image_url': image_url})
```

---

## 八、你要重点观察的输出

训练完或启动后，重点看这些信息：
- 模型是否加载成功。
- 类别数是否和数据集一致。
- 上传图片后是否返回 JSON。
- 前端是否正常显示结果。

如果报错，先看这 4 类问题：
- Flask 没装。
- 模型权重文件不存在。
- 数据集目录不存在。
- 上传文件不是有效图片。

---

## 九、课后必做

1. 跑通 Web 服务

要求：
- `python app/web_app.py` 能启动。
- 浏览器能打开首页。
- 至少上传 3 张图片。

2. 验证单例

要求：
- 在 `get_model()` 里加日志。
- 刷新页面多次，确认模型不会重复加载。

3. 思考页面改进

要求：
- 给结果区加一个“Top-1 预测”高亮。
- 给上传区加一个“清空图片”按钮。

---

## 十、选做

1. 增加错误提示

比如：
- 没有上传文件
- 文件为空
- 不是图片格式

2. 增加结果排序可视化

把置信度最高的结果放在最上面，并加宽进度条。

3. 支持更多模型

思考如何在 Web 页面里切换：
- 预训练模型
- 微调模型

---

## 十一、提交要求

你最后需要交这些内容：
1. 能运行的 Web 服务截图。
2. 上传图片后的结果截图。
3. 你对单例模式的一段解释。
4. 你对 Web 端“状态切换”的理解。

---

## 十二、最后总结

这节课你不是只学了 Flask。

你真正学到的是：
1. 如何把模型能力包装成服务。
2. 如何把服务包装成页面。
3. 如何让一个 AI 项目从“脚本”变成“应用”。
