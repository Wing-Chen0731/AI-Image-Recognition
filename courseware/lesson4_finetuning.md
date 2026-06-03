# 第四课时：数据集准备与微调基础（核心新增①）

---

## Slide 1：为什么需要微调？

标题：让预训练模型适配你的业务场景

- ImageNet 的 1000 类不一定包含你的业务场景，例如“口罩佩戴”“宠物品种细分”。
- 微调让模型在你的数据上再学一点点，准确率可以明显提升，而且通常只需要少量样本，每类几十张就能开始。
- 其实就是 CPSC 210 里的“针对特定规约优化实现”：通用模型已经能识别很多视觉特征，我们只把最后的分类规则改成自己的规约。

讲师备注：这一页要强调“微调不是从零训练”。学生只需要先理解：预训练模型负责通用视觉能力，我们负责把它接到自己的类别上。

---

## Slide 2：数据集构建策略

标题：先把数据整理成模型能读懂的样子

- 推荐使用 Kaggle 小数据集，例如猫狗分类、Food-101 子集，也可以自己拍摄收集。
- 数据目录结构必须规范，这对应 CPSC 210 的模块化思维：数据、模型、训练代码各司其职。

```text
data/
  train/
    cat/
    dog/
  val/
    cat/
    dog/
```

- 使用 `torchvision.datasets.ImageFolder` 自动读取标签。
- 文件夹名就是类别名，例如 `cat/` 和 `dog/` 会自动映射成数字标签。

课堂代码位置：

- 数据集目录检查：`C:\Vscode\AI-Image-Recognition\app\dataset.py` 第 65-77 行，`validate_split_dir()`。
- ImageFolder 封装读取：`C:\Vscode\AI-Image-Recognition\app\dataset.py` 第 80-87 行，`build_image_folder()`。
- DataLoader 构建：`C:\Vscode\AI-Image-Recognition\app\dataset.py` 第 90-120 行，`create_dataloaders()`。
- 完整训练脚本中的 ImageFolder 读取：`C:\Vscode\AI-Image-Recognition\app\finetune.py` 第 151-152 行。
- 完整训练脚本中的 DataLoader 构建：`C:\Vscode\AI-Image-Recognition\app\finetune.py` 第 162-173 行。

---

## Slide 3：数据预处理管线（Data Pipeline）

标题：训练集要“丰富”，验证集要“稳定”

- 可以继承 `torch.utils.data.Dataset`，也可以直接使用 `ImageFolder + DataLoader`。
- 训练集：随机裁剪、水平翻转，用数据增强提高泛化能力。
- 验证集：仅 `Resize + CenterCrop`，避免验证结果被随机增强干扰。
- 体现抽象与单一职责：预处理变换的组合可以用 `Compose` 模式。
- 对应 CPSC 210 Design：组合模式把多个 transform 串成一个处理管线；策略模式让训练和验证使用不同预处理策略。

课堂代码片段：

```python
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
```

课堂代码位置：

- 独立 transform 文件：`C:\Vscode\AI-Image-Recognition\app\transforms.py` 第 8-29 行。
- 策略模式接口：`C:\Vscode\AI-Image-Recognition\app\dataset.py` 第 22-27 行，`PreprocessingStrategy`。
- 训练集预处理策略：`C:\Vscode\AI-Image-Recognition\app\dataset.py` 第 30-44 行，`TrainingPreprocessingStrategy`。
- 验证集预处理策略：`C:\Vscode\AI-Image-Recognition\app\dataset.py` 第 47-62 行，`ValidationPreprocessingStrategy`。
- 完整训练脚本导入 transform：`C:\Vscode\AI-Image-Recognition\app\finetune.py` 第 24-31 行。

---

## Slide 4：微调训练的核心步骤

标题：只改该改的地方

1. 加载预训练模型。
2. 替换分类头：以 MobileNetV3 为例，修改 `classifier` 最后一层输出为你的类别数。
3. 冻结特征提取层：让前面层不更新权重，或只更新部分层。
4. 定义损失函数：分类任务使用交叉熵 `CrossEntropyLoss`。
5. 定义优化器：课堂演示使用 `Adam`。
6. 训练 3 到 5 个 epoch，并在验证集上观察准确率。

讲师备注：冻结 `features` 的意义是保留 ImageNet 预训练学到的边缘、纹理、形状等通用特征；新分类头负责学习当前数据集的类别边界。

课堂代码位置：

- 微调模型构建函数：`C:\Vscode\AI-Image-Recognition\app\model_loader.py` 第 66-95 行，`build_finetune_model()`。
- 冻结特征提取层：`C:\Vscode\AI-Image-Recognition\app\model_loader.py` 第 89-91 行。
- 替换分类头：`C:\Vscode\AI-Image-Recognition\app\model_loader.py` 第 93-94 行。
- 训练脚本调用模型构建：`C:\Vscode\AI-Image-Recognition\app\train.py` 第 147-150 行。
- 完整脚本调用模型构建：`C:\Vscode\AI-Image-Recognition\app\finetune.py` 第 175-184 行。

---

## Slide 5：实战：微调脚本编写（train.py）

标题：从预训练模型到你的专属分类器

现场演示关键代码片段：

```python
model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)

# 替换分类头
model.classifier[3] = nn.Linear(in_features=1280, out_features=num_classes)

# 冻结特征部分
for param in model.features.parameters():
    param.requires_grad = False
```

- 用 `tqdm` 显示进度，观察 loss 是否下降。
- 保存模型：

```python
torch.save(model.state_dict(), "finetuned_mobilenet.pth")
```

运行方式：

```bash
python app/train.py --data-dir data --epochs 3 --batch-size 16
```

课堂代码位置：

- 课堂版训练脚本入口：`C:\Vscode\AI-Image-Recognition\app\train.py` 第 136-184 行。
- 训练一个 epoch：`C:\Vscode\AI-Image-Recognition\app\train.py` 第 47-77 行。
- 验证一个 epoch：`C:\Vscode\AI-Image-Recognition\app\train.py` 第 80-106 行。
- 保存最优模型：`C:\Vscode\AI-Image-Recognition\app\train.py` 第 177-180 行。
- 完整课堂演示版：`C:\Vscode\AI-Image-Recognition\app\finetune.py` 第 146-213 行。

---

## Slide 6：CPSC 210 Design：策略模式与模板方法

标题：把训练代码写成可扩展的流程

- 预处理策略可抽成 `PreprocessingStrategy` 接口，训练和验证使用不同策略。
- 训练循环可定义成模板方法：每个 epoch 固定步骤是训练、验证、保存。
- 具体模型、数据路径、类别数、学习率、epoch 数通过参数注入。
- 异常处理体现前置条件检查：数据目录不存在、类别数不匹配，应在训练开始前立即报错。

对应代码设计：

- 策略模式接口和实现：`C:\Vscode\AI-Image-Recognition\app\dataset.py` 第 22-62 行。
- 模板方法式训练流程：`C:\Vscode\AI-Image-Recognition\app\train.py` 第 165-180 行。
- 完整版训练/验证循环：`C:\Vscode\AI-Image-Recognition\app\finetune.py` 第 61-117 行。
- 数据目录异常：`C:\Vscode\AI-Image-Recognition\app\exceptions.py` 第 20-25 行。
- 训练前置条件检查：`C:\Vscode\AI-Image-Recognition\app\finetune.py` 第 42-58 行、第 155-159 行。

---

## 课时总结

- ImageFolder 要求严格的数据目录结构，文件夹名就是标签。
- 训练集增强、验证集稳定，是微调实验的基本原则。
- MobileNetV3 微调的核心是替换分类头，并按数据量决定是否冻结特征层。
- CPSC 210 的策略模式、模板方法和异常处理，可以自然落到训练代码里。

---

# 代码文件关系

## 项目中的数据流

```text
data/train, data/val
        |
        v
app/transforms.py
        |
        v
app/dataset.py 或 app/finetune.py
        |
        v
app/model_loader.py
        |
        v
app/train.py 或 app/finetune.py
        |
        v
finetuned_mobilenet.pth
```

## 各文件职责

- `C:\Vscode\AI-Image-Recognition\app\transforms.py`：定义训练集和验证集的图像预处理规则。
- `C:\Vscode\AI-Image-Recognition\app\dataset.py`：把 `data/train`、`data/val` 读取成 `ImageFolder` 和 `DataLoader`。
- `C:\Vscode\AI-Image-Recognition\app\model_loader.py`：加载预训练模型，并提供 `build_finetune_model()` 替换分类头。
- `C:\Vscode\AI-Image-Recognition\app\train.py`：课堂主训练脚本，使用 `dataset.py` 和 `model_loader.py` 完成训练。
- `C:\Vscode\AI-Image-Recognition\app\finetune.py`：完整演示脚本，把数据读取、异常检查、训练、验证、保存流程集中展示。
- `C:\Vscode\AI-Image-Recognition\app\exceptions.py`：定义模型加载、图片加载、数据目录、类别数相关异常。
- `C:\Vscode\AI-Image-Recognition\finetuned_mobilenet.pth`：训练产出的模型权重文件。

## 调用关系

```text
app/train.py
  -> app/dataset.py:create_dataloaders()
  -> app/model_loader.py:build_finetune_model()
  -> torch.nn.CrossEntropyLoss
  -> torch.optim.Adam
  -> torch.save()

app/finetune.py
  -> app/transforms.py:train_transform, val_transform
  -> app/model_loader.py:build_finetune_model()
  -> app/exceptions.py:DataDirectoryNotFoundError, ClassCountMismatchError
  -> torchvision.datasets.ImageFolder
  -> torch.utils.data.DataLoader
  -> torch.save()

app/dataset.py
  -> torchvision.datasets.ImageFolder
  -> torch.utils.data.DataLoader
  -> torchvision.transforms.Compose

app/model_loader.py
  -> torchvision.models.mobilenet_v3_large
  -> torch.nn.Linear
```

---

# 新增/修改代码逐段解读

## `app/transforms.py`

代码位置：`C:\Vscode\AI-Image-Recognition\app\transforms.py` 第 1-29 行。

- 第 1 行：说明本文件负责可复用的 `torchvision` 图像预处理管线。
- 第 3 行：启用未来版本的类型注解行为，提高类型标注兼容性。
- 第 5 行：导入 `torchvision.transforms`，用于组合裁剪、翻转、张量转换和归一化操作。
- 第 8-9 行：定义 ImageNet 预训练模型要求的均值和标准差。MobileNetV3 使用 ImageNet 预训练权重，因此输入图像也应按 ImageNet 的统计量归一化。
- 第 12-19 行：定义 `train_transform`。训练集使用 `RandomResizedCrop(224)` 和 `RandomHorizontalFlip()` 做数据增强，再转成 tensor 并归一化。
- 第 22-29 行：定义 `val_transform`。验证集只做 `Resize(256)`、`CenterCrop(224)`、`ToTensor()`、`Normalize()`，保证验证结果稳定可比较。

教学重点：训练集和验证集的 transform 不应该完全一样。训练集需要随机性提高泛化能力；验证集需要确定性保证评估稳定。

## `app/dataset.py`

代码位置：`C:\Vscode\AI-Image-Recognition\app\dataset.py` 第 1-128 行。

- 第 1-6 行：文件说明。这个模块服务于第四课时，把预处理策略和 DataLoader 构建分离。
- 第 8-15 行：导入依赖。`ABC` 和 `abstractmethod` 用于策略模式接口；`Path` 用于跨平台路径处理；`DataLoader` 和 `ImageFolder` 用于数据集加载。
- 第 18-19 行：定义 ImageNet 归一化参数，和 `app/transforms.py` 中保持一致。
- 第 22-27 行：定义 `PreprocessingStrategy` 抽象类。它要求所有预处理策略都实现 `build()` 方法，返回一个 `transforms.Compose`。
- 第 30-44 行：定义 `TrainingPreprocessingStrategy`。它封装训练集增强策略，包括随机裁剪、水平翻转、转 tensor、归一化。
- 第 47-62 行：定义 `ValidationPreprocessingStrategy`。它封装验证集策略，包括 resize、中心裁剪、转 tensor、归一化。
- 第 65-77 行：定义 `validate_split_dir()`。它检查 `data/train` 或 `data/val` 是否存在、是否是目录、是否包含类别子目录。
- 第 80-87 行：定义 `build_image_folder()`。它先检查目录，再用 `datasets.ImageFolder` 创建数据集对象。
- 第 90-120 行：定义 `create_dataloaders()`。它分别创建训练集和验证集，然后封装成 `DataLoader`，最后返回 `train_loader`、`val_loader` 和 `class_to_idx`。
- 第 123-128 行：定义 `count_classes()`。它统计 `data/train` 下的类别文件夹数量，可用于快速确认分类头输出维度。

教学重点：这个文件展示了 CPSC 210 的策略模式。训练预处理和验证预处理都遵循同一个接口，但具体实现不同。

## `app/model_loader.py`

代码位置：`C:\Vscode\AI-Image-Recognition\app\model_loader.py` 第 1-95 行。

- 第 1-7 行：文件说明。这里强调 `ModelLoader` 是模型加载抽象，`build_finetune_model()` 是微调模型构建函数。
- 第 9-18 行：导入依赖和异常。`torch.nn` 用于创建新的 `Linear` 分类头；`ModelDownloadError` 用于包装模型权重加载失败。
- 第 21-27 行：定义 `ModelLoader` 抽象类。它规定子类必须实现 `load_model()`。
- 第 29-48 行：定义 `MobileNetV3Loader`。它加载 ImageNet 预训练的 MobileNetV3-Large，并切换到 `eval()`，用于推理。
- 第 51-63 行：定义 `ResNet18Loader`。这是另一个模型加载器示例，说明只要实现同一个接口，就可以替换模型。
- 第 66-75 行：定义 `build_finetune_model()` 的函数签名和说明。`num_classes` 决定新分类头输出维度，`freeze_features` 决定是否冻结特征提取层。
- 第 77-78 行：检查类别数。分类任务至少需要两个类别。
- 第 80-87 行：加载 MobileNetV3-Large 的 ImageNet 预训练权重。如果加载失败，抛出 `ModelDownloadError`。
- 第 89-91 行：如果 `freeze_features=True`，则把 `model.features` 中所有参数的 `requires_grad` 设为 `False`。
- 第 93-94 行：读取旧分类头输入维度，然后把 `classifier[3]` 替换成新的 `nn.Linear(in_features, num_classes)`。
- 第 95 行：返回已经适配当前类别数的模型。

教学重点：这就是本节课微调的核心。主干特征提取器保持不动，只替换最后的分类头。

## `app/train.py`

代码位置：`C:\Vscode\AI-Image-Recognition\app\train.py` 第 1-184 行。

- 第 1-15 行：说明训练脚本的预期数据目录结构和运行方式。
- 第 17-25 行：导入训练需要的标准库和 PyTorch 组件。
- 第 26-29 行：尝试导入 `tqdm`。如果环境中没有安装，脚本仍然可以运行，只是不显示进度条。
- 第 31-36 行：导入 `create_dataloaders()` 和 `build_finetune_model()`。这里兼容两种运行方式：`python app/train.py` 和包导入。
- 第 39-44 行：定义 `iter_progress()`。它负责统一处理是否使用 `tqdm`。
- 第 47-77 行：定义 `train_one_epoch()`。核心步骤是训练模式、前向传播、计算 loss、反向传播、优化器更新、统计准确率。
- 第 80-106 行：定义 `validate()`。核心步骤是验证模式、关闭梯度、前向传播、计算 loss、统计准确率。
- 第 109-133 行：定义命令行参数，包括数据目录、epoch 数、batch size、学习率、输出路径、是否解冻特征层。
- 第 136-145 行：进入 `main()`，解析参数，选择 `cuda` 或 `cpu`，创建训练和验证 DataLoader。
- 第 147-155 行：构建微调模型、损失函数和 Adam 优化器。优化器只接收 `requires_grad=True` 的参数。
- 第 157-160 行：打印设备、类别映射、训练样本数和验证样本数。
- 第 162-180 行：执行训练循环。每个 epoch 先训练，再验证；如果验证准确率不低于历史最好值，就保存模型权重。
- 第 183-184 行：脚本入口。直接运行该文件时会调用 `main()`。

教学重点：这个文件是课堂最推荐演示的训练入口，结构比 `finetune.py` 更精简，适合现场讲解训练主流程。

## `app/finetune.py`

代码位置：`C:\Vscode\AI-Image-Recognition\app\finetune.py` 第 1-213 行。

- 第 1-5 行：说明这是完整的 MobileNetV3 微调脚本，并给出运行命令。
- 第 7-17 行：导入训练所需依赖，包括 `torch`、`DataLoader`、`ImageFolder`。
- 第 19-22 行：导入 `tqdm`，如果没有安装，则退化为无进度条运行。
- 第 24-31 行：导入本项目内部模块，包括异常类、模型构建函数、训练和验证 transform。
- 第 34-39 行：定义 `progress()`，统一封装进度条逻辑。
- 第 42-58 行：定义 `validate_data_dir()`。训练开始前检查 `data`、`data/train`、`data/val` 是否存在，并确认每个 split 下有类别文件夹。
- 第 61-89 行：定义 `train_epoch()`。它执行一个 epoch 的训练，并返回平均 loss 和 accuracy。
- 第 92-117 行：定义 `validate_epoch()`。它执行一个 epoch 的验证，不计算梯度，不更新参数。
- 第 120-143 行：定义命令行参数。相比 `train.py`，它多了 `--num-classes`，可以显式检查期望类别数。
- 第 146-153 行：解析参数、检查数据目录、用 `ImageFolder` 创建训练集和验证集。
- 第 155-159 行：如果用户指定了 `--num-classes`，这里会检查实际类别数是否匹配。
- 第 161-173 行：创建训练和验证 DataLoader。
- 第 175-184 行：选择设备，构建微调模型、损失函数和优化器。
- 第 186-189 行：打印运行信息，方便确认数据集和设备是否正确。
- 第 191-209 行：执行完整训练循环，并在验证准确率刷新时保存模型。
- 第 212-213 行：脚本入口。

教学重点：这个文件适合作为“完整工程版”展示。它把异常检查、类别数校验、transform 文件拆分都展示出来，更接近正式项目代码。

## `app/exceptions.py`

代码位置：`C:\Vscode\AI-Image-Recognition\app\exceptions.py` 第 1-25 行。

- 第 1 行：说明本文件集中定义项目异常。
- 第 4-13 行：定义模型加载相关异常。`ModelDownloadError` 会在预训练权重加载失败时使用。
- 第 16-17 行：定义图片加载异常，用于推理阶段图片读取失败。
- 第 20-21 行：定义 `DataDirectoryNotFoundError`，用于训练前发现数据目录缺失。
- 第 24-25 行：定义 `ClassCountMismatchError`，用于用户配置的类别数和实际数据集类别数不一致。

教学重点：异常类本身很短，但它让训练脚本能表达更清楚的错误含义。这对应 CPSC 210 的前置条件检查和异常处理。

---

# 建议课堂讲解顺序

1. 先讲 `data/train` 和 `data/val` 的目录结构，对应 Slide 2。
2. 再讲 `app/transforms.py`，解释训练集增强和验证集稳定性的区别。
3. 然后讲 `app/dataset.py`，说明 ImageFolder 如何自动生成标签。
4. 接着讲 `app/model_loader.py` 的 `build_finetune_model()`，这是微调的核心。
5. 最后讲 `app/train.py` 的训练循环，运行 1 个 epoch 看 loss 和 accuracy。
6. 如果要展示工程化版本，再补充 `app/finetune.py` 的异常检查和参数校验。

推荐演示命令：

```bash
conda run -n pytorch_env python app/train.py --data-dir data --epochs 1 --batch-size 16 --num-workers 0
```
