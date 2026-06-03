# Lesson 4：数据集准备、MobileNetV3 微调与推理对比

> 基于当前最新代码生成  
> 项目路径：`C:\Vscode\AI-Image-Recognition`

---

## Slide 1：本节课目标

标题：从“会用预训练模型”到“训练自己的分类器”

本节课完成 5 件事：

1. 理解为什么需要微调。
2. 掌握 `ImageFolder` 数据集目录结构。
3. 理解训练集和验证集为什么使用不同预处理。
4. 使用 MobileNetV3 完成猫狗二分类微调。
5. 对比微调前和微调后的推理结果。

本节课的核心代码：

- `app/train.py`：课堂主讲训练脚本。
- `app/finetune.py`：完整工程版训练脚本。
- `app/model_loader.py`：加载预训练模型、冻结特征层、替换分类头。
- `app/dataset.py`：构建 `ImageFolder` 和 `DataLoader`。
- `app/transforms.py`：定义训练和验证预处理。
- `app/compare_inference.py`：对比微调前后推理效果。

---

## Slide 2：本项目不是 LoRA 微调

标题：本节课使用传统 CNN 迁移学习

本项目使用的是：

```text
预训练 MobileNetV3
-> 冻结 features 特征提取层
-> 替换 classifier 最后一层
-> 只训练新的分类头
```

它不是 LoRA。

LoRA 的核心是：

```text
冻结原模型参数
-> 在 Transformer 的 attention/linear 层插入低秩适配器
-> 只训练 adapter 参数
```

课堂结论：

- 本项目模型是 MobileNetV3，属于 CNN。
- 本项目代码没有注入 LoRA adapter。
- 本项目采用的是“冻结主干 + 替换分类头”的传统迁移学习方法。

---

## Slide 3：数据集目录结构

标题：文件夹名就是标签

本项目使用 `torchvision.datasets.ImageFolder` 读取数据。  
目录必须整理成下面这种结构：

```text
data/
  train/
    cat/
    dog/
  val/
    cat/
    dog/
```

含义：

- `data/train`：训练集。
- `data/val`：验证集。
- `cat/`、`dog/`：类别名称。
- `ImageFolder` 会自动把类别名映射成数字标签。

运行时会看到类似输出：

```text
Classes: {'cat': 0, 'dog': 1}
Training samples: 275
Validation samples: 70
```

对应代码：

- `app/dataset.py`：`create_dataloaders()`
- `app/finetune.py`：`ImageFolder(data_dir / "train", train_transform)`

---

## Slide 4：训练集和验证集预处理

标题：训练集要有变化，验证集要稳定

训练集预处理：

```python
transforms.RandomResizedCrop(224)
transforms.RandomHorizontalFlip()
transforms.ToTensor()
transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
```

目的：

- 制造更多图片变化。
- 减少模型死记硬背。
- 提高泛化能力。

验证集预处理：

```python
transforms.Resize(256)
transforms.CenterCrop(224)
transforms.ToTensor()
transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
```

目的：

- 不引入随机性。
- 保证每次验证结果可比较。
- 更准确观察模型是否真的变好。

对应代码：

- `app/transforms.py`
- `app/dataset.py` 的 `TrainingPreprocessingStrategy`
- `app/dataset.py` 的 `ValidationPreprocessingStrategy`

---

## Slide 5：为什么要加载预训练模型

标题：不要从零开始训练

MobileNetV3 已经在 ImageNet 上学过大量图像特征。

它已经学会的内容包括：

- 边缘
- 纹理
- 轮廓
- 局部形状
- 一些通用视觉模式

所以本节课不从零训练，而是：

```text
借用预训练模型的通用视觉能力
只让最后的分类层学习 cat/dog 任务
```

对应代码：

```python
model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
```

位置：

- `app/model_loader.py` 的 `MobileNetV3Loader`
- `app/model_loader.py` 的 `build_finetune_model()`

---

## Slide 6：微调模型结构

标题：冻结 features，替换 classifier

核心函数：

```python
def build_finetune_model(num_classes: int, freeze_features: bool = True) -> nn.Module:
    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)

    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model
```

逐段解释：

- `num_classes`：当前数据集有几个类别。
- `freeze_features=True`：默认冻结特征提取层。
- `model.features`：MobileNetV3 前面的图像特征提取部分。
- `requires_grad=False`：这些参数训练时不更新。
- `model.classifier[3]`：MobileNetV3 最后一层分类头。
- `nn.Linear(in_features, num_classes)`：替换成适合当前类别数的新分类层。

对应代码：

- `app/model_loader.py`

---

## Slide 7：训练脚本到底用哪个

标题：课堂主讲 train.py，工程版看 finetune.py

当前项目有两个训练脚本：

```text
app/train.py
app/finetune.py
```

它们不是冲突关系。

`app/train.py`：

- 课堂主讲版本。
- 结构更短。
- 适合讲清楚训练闭环。
- 使用 `create_dataloaders()` 统一创建数据加载器。

`app/finetune.py`：

- 完整工程版本。
- 多了数据目录检查。
- 多了类别数检查。
- 直接展示 `ImageFolder`、`DataLoader`、异常处理。

课堂建议：

- 主讲 `app/train.py`。
- 补充展示 `app/finetune.py`。
- 实操训练可以优先运行 `app/finetune.py`，因为它检查更完整。

---

## Slide 7.1：train.py 和 finetune.py 的代码差异

标题：一个适合讲训练主线，一个适合讲工程完整性

两份脚本的训练目标相同：

```text
读取数据 -> 构建 MobileNetV3 微调模型 -> 训练 -> 验证 -> 保存模型
```

但代码组织方式不同。

| 对比点 | `app/train.py` | `app/finetune.py` |
|---|---|---|
| 定位 | 课堂精简版 | 工程完整版 |
| 数据加载 | 调用 `dataset.py` 里的 `create_dataloaders()` | 在脚本内直接使用 `ImageFolder` 和 `DataLoader` |
| transform 来源 | 来自 `dataset.py` 内部策略类 | 直接导入 `app/transforms.py` 的 `train_transform` 和 `val_transform` |
| 数据目录检查 | 由 `dataset.py` 的 `validate_split_dir()` 简单检查 | 由 `validate_data_dir()` 做更明确的训练前检查 |
| 类别数检查 | 自动读取类别数，不单独校验用户输入 | 支持 `--num-classes`，可检查期望类别数和实际类别数是否一致 |
| 异常处理 | 更少，流程更短 | 使用 `DataDirectoryNotFoundError` 和 `ClassCountMismatchError` |
| 适合课堂讲解 | 适合主讲训练闭环 | 适合补充工程化写法 |

### 代码差异 1：数据加载方式不同

`app/train.py` 使用封装好的数据加载函数：

```python
train_loader, val_loader, class_to_idx = create_dataloaders(
    data_dir=args.data_dir,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
)
num_classes = len(class_to_idx)
```

位置：`app/train.py` 第 140-145 行。

作用：

- 把数据读取细节藏到 `dataset.py`。
- 课堂讲解时主线更清楚。
- 学生先理解训练流程，不容易被工程细节分散注意力。

`app/finetune.py` 直接在脚本中展示数据加载细节：

```python
train_set = ImageFolder(data_dir / "train", train_transform)
val_set = ImageFolder(data_dir / "val", val_transform)
discovered_classes = len(train_set.classes)
```

位置：`app/finetune.py` 第 151-153 行。

作用：

- 让学生看到 `ImageFolder` 是如何读取 `data/train` 和 `data/val` 的。
- 更适合讲工程版代码，因为数据集对象和类别数都暴露得更清楚。

### 代码差异 2：前置检查不同

`app/train.py` 依赖 `dataset.py` 中的目录检查：

```python
validate_split_dir(split_path)
```

位置：`app/dataset.py` 第 85-87 行。

作用：

- 检查 `data/train` 或 `data/val` 是否存在。
- 检查目录下是否有类别子文件夹。

`app/finetune.py` 自己定义了更完整的检查函数：

```python
def validate_data_dir(data_dir: Path) -> None:
    if not data_dir.exists():
        raise DataDirectoryNotFoundError(...)

    for split in ("train", "val"):
        split_dir = data_dir / split
        ...
```

位置：`app/finetune.py` 第 42-58 行。

作用：

- 训练开始前检查 `data`、`data/train`、`data/val`。
- 如果目录结构错误，提前抛出明确异常。
- 这是工程代码里常见的“前置条件检查”。

### 代码差异 3：类别数校验不同

`app/train.py` 自动根据数据集类别数决定分类头输出维度：

```python
num_classes = len(class_to_idx)
model = build_finetune_model(num_classes=num_classes, ...)
```

位置：`app/train.py` 第 145-150 行。

作用：

- 简洁。
- 不需要学生手动设置类别数。
- 适合课堂快速跑通。

`app/finetune.py` 允许用户用 `--num-classes` 显式检查类别数：

```python
if args.num_classes is not None and discovered_classes != args.num_classes:
    raise ClassCountMismatchError(...)
```

位置：`app/finetune.py` 第 155-159 行。

作用：

- 防止“以为是 3 类，实际只读到 2 类”的错误。
- 适合数据集类别较多时使用。
- 对后续做猫品种细分、多分类任务更有帮助。

### 代码差异 4：训练循环本质相同

`app/train.py` 的训练函数：

```python
def train_one_epoch(...):
    model.train()
    ...
    loss.backward()
    optimizer.step()
```

位置：`app/train.py` 第 47-77 行。

`app/finetune.py` 的训练函数：

```python
def train_epoch(...):
    model.train()
    ...
    loss.backward()
    optimizer.step()
```

位置：`app/finetune.py` 第 61-89 行。

结论：

- 两者训练逻辑相同。
- 都是 `forward -> loss -> backward -> step`。
- 不同的是外层工程组织方式，不是训练算法不同。

### 课堂讲法建议

课堂上可以这样解释：

```text
train.py 负责让同学看懂“微调怎么训练”。
finetune.py 负责让同学看懂“正式项目里还要多做哪些检查”。
```

所以：

- 讲训练主线：看 `app/train.py`。
- 讲数据目录检查：看 `app/finetune.py`。
- 讲类别数校验：看 `app/finetune.py`。
- 讲模型结构修改：看 `app/model_loader.py`。
- 讲数据封装和策略模式：看 `app/dataset.py`。

---

## Slide 8：训练核心步骤

标题：一次 epoch 里发生了什么

训练一轮的标准流程：

```python
model.train()
optimizer.zero_grad()
outputs = model(images)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
```

逐句解释：

- `model.train()`：切换到训练模式。
- `optimizer.zero_grad()`：清空上一次梯度。
- `outputs = model(images)`：模型预测。
- `loss = criterion(outputs, labels)`：计算预测和真实标签的差距。
- `loss.backward()`：反向传播，计算参数该怎么改。
- `optimizer.step()`：优化器更新参数。

对应代码：

- `app/train.py` 的 `train_one_epoch()`
- `app/finetune.py` 的 `train_epoch()`

---

## Slide 9：验证核心步骤

标题：验证集是考试，不是训练

验证流程：

```python
model.eval()
with torch.no_grad():
    outputs = model(images)
    loss = criterion(outputs, labels)
```

逐句解释：

- `model.eval()`：切换到验证模式。
- `torch.no_grad()`：不计算梯度，节省显存和时间。
- `outputs = model(images)`：只做预测。
- `loss = criterion(...)`：评估错误程度。

验证阶段不会执行：

```python
loss.backward()
optimizer.step()
```

原因：

- 验证集用于评估模型效果。
- 验证集不用于更新模型参数。

对应代码：

- `app/train.py` 的 `validate()`
- `app/finetune.py` 的 `validate_epoch()`

---

## Slide 10：损失函数和优化器

标题：一个负责判断错多少，一个负责改参数

损失函数：

```python
criterion = nn.CrossEntropyLoss()
```

作用：

- 衡量预测结果和真实标签之间的差距。
- 分类任务常用交叉熵损失。
- loss 越小，说明模型预测越接近真实标签。

优化器：

```python
optimizer = Adam(
    (param for param in model.parameters() if param.requires_grad),
    lr=args.lr,
)
```

作用：

- 根据梯度更新模型参数。
- 只更新 `requires_grad=True` 的参数。
- 默认冻结 `features` 时，主要训练新的分类头。

对应代码：

- `app/train.py` 的 `main()`
- `app/finetune.py` 的 `main()`

---

## Slide 11：运行训练

标题：先跑 1 轮，再跑 3 轮

先确认流程能跑通：

```bash
conda run -n pytorch_env python app/finetune.py --data-dir data --epochs 1 --batch-size 16 --num-workers 0
```

成功后再跑 3 轮：

```bash
conda run -n pytorch_env python app/finetune.py --data-dir data --epochs 3 --batch-size 16 --num-workers 0
```

参数解释：

- `--data-dir data`：数据集目录。
- `--epochs 1`：完整训练集跑 1 遍。
- `--epochs 3`：完整训练集跑 3 遍。
- `--batch-size 16`：每次送 16 张图片进模型。
- `--num-workers 0`：Windows 环境下更稳。

运行成功后会生成：

```text
finetuned_mobilenet.pth
```

---

## Slide 12：如何看训练结果

标题：不要只看 train acc

示例输出：

```text
Device: cuda
Classes: {'cat': 0, 'dog': 1}
Training samples: 275
Validation samples: 70
Epoch 1/3: train loss 0.3744, acc 0.8036 | val loss 0.0973, acc 0.9571
Epoch 2/3: train loss 0.2064, acc 0.9091 | val loss 0.0625, acc 0.9714
Epoch 3/3: train loss 0.1566, acc 0.9309 | val loss 0.0782, acc 0.9714
Saved model to finetuned_mobilenet.pth
```

怎么看：

- `Device: cuda`：使用 GPU。
- `Classes`：类别映射正确。
- `Training samples`：训练集样本数。
- `Validation samples`：验证集样本数。
- `train loss`：训练集错误程度。
- `train acc`：训练集准确率。
- `val loss`：验证集错误程度。
- `val acc`：验证集准确率。

课堂结论：

- 1 轮用于验证流程。
- 3 轮用于观察模型是否继续收敛。
- 如果 `train acc` 上升但 `val acc` 不升，可能已经接近饱和或开始过拟合。

---

## Slide 13：微调前后推理对比

标题：证明微调确实改变了模型任务

运行命令：

```bash
conda run -n pytorch_env python app/compare_inference.py
```

该脚本会读取：

- 图片：`images/cat.jpg`
- 微调模型：`finetuned_mobilenet.pth`
- 自定义类别：`data/train` 下的文件夹名

输出包含两部分：

```text
Pretrained MobileNetV3 (ImageNet)
Fine-tuned MobileNetV3 (custom classes)
```

含义：

- 微调前：输出 ImageNet 1000 类标签。
- 微调后：输出当前项目的自定义类别，例如 `cat` 和 `dog`。

对应代码：

- `app/compare_inference.py`

---

## Slide 14：ImageNet 1000 类标签有什么用

标题：标签表负责把数字翻译成人能看懂的类别

预训练模型输出的是 1000 个分数：

```text
[score_0, score_1, score_2, ..., score_999]
```

模型本身不会直接输出文字。  
ImageNet 1000 类标签的作用是：

```text
类别编号 -> 类别名称
```

例如：

```text
281 -> tabby
285 -> Egyptian cat
```

所以：

- 预训练模型负责算分。
- ImageNet 标签负责解释编号。
- 微调后的模型不再使用 ImageNet 1000 类标签，而是使用 `data/train` 里的自定义类别名。

对应代码：

- `app/compare_inference.py` 中 `weights.meta["categories"]`
- `app/compare_inference.py` 中 `ImageFolder(data_dir / "train").classes`

---

## Slide 15：代表性调优 1：冻结 vs 解冻

标题：让模型学多少新东西

基线：冻结 `features`，只训练分类头。

```bash
conda run -n pytorch_env python app/finetune.py --data-dir data --epochs 3 --batch-size 16 --num-workers 0
```

调优：解冻 `features`，让特征层也参与训练。

```bash
conda run -n pytorch_env python app/finetune.py --data-dir data --epochs 3 --batch-size 16 --num-workers 0 --unfreeze-features
```

观察指标：

- `train loss`
- `train acc`
- `val loss`
- `val acc`

讲解重点：

- 冻结更稳，适合小数据集。
- 解冻更灵活，适合数据更多或任务更细。
- 如果解冻后训练集变好但验证集不变，说明不一定有泛化收益。

---

## Slide 16：代表性调优 2：学习率与数据增强

标题：让模型学得更稳

基线学习率：

```bash
conda run -n pytorch_env python app/finetune.py --data-dir data --epochs 3 --batch-size 16 --num-workers 0 --lr 1e-3
```

尝试更小学习率：

```bash
conda run -n pytorch_env python app/finetune.py --data-dir data --epochs 3 --batch-size 16 --num-workers 0 --lr 5e-4
```

可以进一步尝试：

```bash
conda run -n pytorch_env python app/finetune.py --data-dir data --epochs 3 --batch-size 16 --num-workers 0 --lr 1e-4
```

数据增强可改位置：

- `app/transforms.py`

示例增强：

```python
transforms.ColorJitter(0.2, 0.2, 0.2)
transforms.RandomRotation(10)
```

注意：

- 只给训练集加随机增强。
- 验证集不要加随机增强。
- 如果任务依赖颜色，例如金渐层和银渐层识别，颜色增强不要过强。

---

## Slide 17：如果要做猫品种细分

标题：先改数据集，再谈模型

当前任务：

```text
cat / dog
```

如果要识别猫品种或花色，需要改成：

```text
data/
  train/
    british_shorthair_silver_shaded/
    british_shorthair_golden_shaded/
    british_shorthair_blue/
  val/
    british_shorthair_silver_shaded/
    british_shorthair_golden_shaded/
    british_shorthair_blue/
```

代码层面：

- `ImageFolder` 会自动读取新类别。
- `num_classes` 会自动变成新类别数。
- `build_finetune_model()` 会自动替换分类头输出维度。

细粒度任务建议：

- 数据质量优先。
- 每类尽量有足够样本。
- 可尝试更高输入分辨率。
- 可尝试 `--unfreeze-features`。
- 学习率建议降低，例如 `1e-4` 或 `5e-5`。

---

## Slide 18：本节课总结

标题：完整微调闭环

本节课完整流程：

1. 准备 `data/train` 和 `data/val`。
2. 使用 `ImageFolder` 读取类别。
3. 训练集使用随机增强，验证集使用稳定预处理。
4. 加载 ImageNet 预训练 MobileNetV3。
5. 冻结 `features`。
6. 替换 `classifier` 最后一层。
7. 使用 `CrossEntropyLoss` 和 `Adam` 训练。
8. 每轮训练后验证。
9. 保存最佳模型 `finetuned_mobilenet.pth`。
10. 使用 `compare_inference.py` 对比微调前后效果。

最重要的三句话：

- 微调不是从零训练。
- 本项目不是 LoRA，而是传统 CNN 迁移学习。
- 想做更细分类，第一步永远是准备更细、更干净的数据集。
