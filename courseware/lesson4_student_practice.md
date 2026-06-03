# 第四课时学生实操课件：数据准备与微调训练

> 适用对象：第一次接触 PyTorch 微调训练的同学  
> 实操目标：把一个预训练的 MobileNetV3 跑在自己的图片分类数据集上，并成功保存模型

---

## 一、这节课你要完成什么

这一节课你要完成 4 件事：

1. 把数据集整理成 `data/train` 和 `data/val` 的标准结构。
2. 运行预处理和数据加载代码，确认训练集和验证集能被正确读取。
3. 运行微调训练脚本，观察 loss 和 accuracy 的变化。
4. 保存训练好的模型文件 `finetuned_mobilenet.pth`。

如果你能做到上面 4 件事，就说明你已经完成了本节课的核心实操。

---

## 二、课前准备

### 1. 打开项目

请先确认你已经打开项目根目录：

```text
C:\Vscode\AI-Image-Recognition
```

### 2. 确认 Python 环境

建议使用已经安装好 PyTorch 的环境。  
如果你是跟课堂环境一致，一般直接使用项目自带的环境即可。

### 3. 确认依赖已安装

至少需要这些库：

- `torch`
- `torchvision`
- `tqdm`

如果你运行时提示缺少模块，先安装依赖再继续。

---

## 三、先看清楚数据目录结构

训练前，数据必须整理成下面这个样子：

```text
data/
  train/
    类别1/
    类别2/
  val/
    类别1/
    类别2/
```

### 这一步的目的

- `ImageFolder` 会把文件夹名当作类别名。
- 例如 `cat/`、`dog/` 会自动变成标签。
- 如果目录结构不对，训练脚本会直接报错。

### 你需要检查什么

- `data` 文件夹是否存在
- `data/train` 是否存在
- `data/val` 是否存在
- 每个 split 下是否真的有类别文件夹

### 对应代码位置

- 数据目录检查：`app/finetune.py` 第 42-58 行
- 数据集读取辅助函数：`app/dataset.py` 第 65-87 行
- 异常类：`app/exceptions.py` 第 20-25 行

---

## 四、先理解训练和验证为什么要用不同预处理

这一步你不需要记住所有代码，先记住一句话：

> 训练集要“多样一点”，验证集要“稳定一点”。

### 1. 训练集预处理做什么

训练集一般会做：

- 随机裁剪
- 随机水平翻转
- 转成 Tensor
- 归一化

### 目的

- 让模型看到更多变化
- 防止模型死记硬背训练图
- 提高泛化能力

### 2. 验证集预处理做什么

验证集一般会做：

- Resize
- CenterCrop
- 转成 Tensor
- 归一化

### 目的

- 让验证输入更稳定
- 方便比较每一轮训练后的效果

### 对应代码位置

- 策略接口：`app/dataset.py` 第 22-27 行
- 训练集策略：`app/dataset.py` 第 30-44 行
- 验证集策略：`app/dataset.py` 第 47-62 行
- 可复用 transform：`app/transforms.py` 第 1-29 行

---

## 五、实操步骤 1：先检查数据是否能被正确读取

### 方法 A：直接运行完整训练脚本做检查

在项目根目录打开终端，输入：

```bash
python app/finetune.py --data-dir data --epochs 1 --batch-size 16 --num-workers 0
```

### 这条命令是什么意思

- `python app/finetune.py`：运行完整微调脚本
- `--data-dir data`：告诉程序数据在 `data` 文件夹下
- `--epochs 1`：先只跑 1 轮，方便检查流程
- `--batch-size 16`：每次送 16 张图进模型
- `--num-workers 0`：Windows 课堂环境建议设为 0，避免多进程问题

### 运行后你应该看到什么

- 程序先检查数据目录
- 程序打印类别信息
- 程序开始训练
- 训练完后输出 loss 和 accuracy

### 如果报错怎么办

常见错误有：

- 数据目录不存在
- `train` 或 `val` 文件夹不存在
- 类别文件夹为空
- 传入的类别数和真实类别数不一致

这些都会在训练开始前报错，这样你更容易定位问题。

---

## 六、实操步骤 2：理解模型是怎么被改成适合你数据集的

训练前，预训练模型不会直接拿来用，而是要改最后一层。

### 你要记住的核心逻辑

1. 加载预训练 MobileNetV3。
2. 冻结前面的特征提取层。
3. 把最后的分类头换成你的类别数。

### 为什么要这样做

- 前面的层已经学会了通用图像特征，比如边缘、纹理、轮廓
- 你不想把这些通用能力重新学一遍
- 你只需要让最后一层学会“怎么分你的类别”

### 对应代码位置

- 模型构建函数：`app/model_loader.py` 第 66-95 行
- 冻结特征层：`app/model_loader.py` 第 89-91 行
- 替换分类头：`app/model_loader.py` 第 93-94 行

### 你可以在代码里重点看这三句

```python
model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
for param in model.features.parameters():
    param.requires_grad = False
model.classifier[3] = nn.Linear(in_features=1280, out_features=num_classes)
```

### 白话解释

- 第一行：拿来一个已经会看图的模型
- 第二行：前面特征层先别改
- 第三行：把最后一层改成适合你的类别数

---

## 七、实操步骤 3：真正开始训练

这一步才是本节课最核心的部分。

### 推荐命令

先用最稳妥的方式跑一轮：

```bash
python app/finetune.py --data-dir data --epochs 3 --batch-size 16 --num-workers 0
```

### 这一步会做什么

训练脚本会自动完成：

1. 读取训练集和验证集
2. 构建微调模型
3. 定义损失函数
4. 定义优化器
5. 训练一个 epoch
6. 验证一个 epoch
7. 比较当前验证准确率是否最好
8. 保存最好的模型

### 对应代码位置

- 训练一轮：`app/finetune.py` 第 61-89 行
- 验证一轮：`app/finetune.py` 第 92-117 行
- 主训练流程：`app/finetune.py` 第 146-209 行

---

## 八、这几行代码分别在干嘛

训练时最关键的两项配置是：**损失函数** 和 **优化器**。

### 1. 损失函数

代码位置：

- `app/train.py` 第 151 行左右
- `app/finetune.py` 第 180 行左右

代码形式：

```python
criterion = nn.CrossEntropyLoss()
```

### 它的作用

- 用来衡量“模型预测得有多错”
- 分类任务最常用
- 预测越接近真实标签，loss 越小

### 2. 优化器

代码位置：

- `app/train.py` 第 152-155 行左右
- `app/finetune.py` 第 181-184 行左右

代码形式：

```python
optimizer = Adam(
    (param for param in model.parameters() if param.requires_grad),
    lr=args.lr,
)
```

### 它的作用

- 根据 loss 的结果更新模型参数
- 这里用的是 Adam
- 只会更新 `requires_grad=True` 的参数

### 白话总结

- **损失函数**：告诉模型“你错了多少”
- **优化器**：告诉模型“你怎么改”

---

## 九、训练循环里每一步在做什么

你可以把一个 epoch 理解成“完整训练一轮”。

### 训练阶段

对应代码：

- `app/train.py` 第 47-77 行
- `app/finetune.py` 第 61-89 行

训练时固定顺序是：

1. `model.train()`：切换到训练模式
2. `optimizer.zero_grad()`：清空梯度
3. `outputs = model(images)`：模型做预测
4. `loss = criterion(outputs, labels)`：计算损失
5. `loss.backward()`：反向传播
6. `optimizer.step()`：更新参数

### 验证阶段

对应代码：

- `app/train.py` 第 80-106 行
- `app/finetune.py` 第 92-117 行

验证时固定顺序是：

1. `model.eval()`：切换到验证模式
2. `torch.no_grad()`：不计算梯度
3. `outputs = model(images)`：模型做预测
4. 统计 loss 和 accuracy

### 为什么验证时不更新参数

因为验证集的作用不是训练，而是检查模型有没有真正学会。

---

## 十、你在终端里应该怎么看训练结果

训练脚本一般会输出类似这样的信息：

```text
Device: cpu
Classes: {'cat': 0, 'dog': 1}
Training samples: 80
Validation samples: 20
Epoch 1/3: train loss 0.5421, acc 0.8125 | val loss 0.4132, acc 0.9000
Saved model to finetuned_mobilenet.pth
```

### 你要重点看什么

- `Device`：是在 CPU 还是 GPU 上跑
- `Classes`：类别是否正确
- `Training samples`：训练集样本数是否正常
- `Validation samples`：验证集样本数是否正常
- `train loss`：训练误差是否下降
- `val acc`：验证准确率是否上升
- `Saved model`：是否成功保存模型

### 实操判断标准

如果你看到：

- loss 在下降
- accuracy 在上升
- 最后出现模型保存提示

说明这次训练基本是成功的。

---

## 十一、实操步骤 4：确认模型已经保存成功

训练结束后，检查项目根目录是否生成了：

```text
finetuned_mobilenet.pth
```

### 这个文件是什么

这是训练好的模型参数文件。  
以后你做预测、部署、继续训练，都可以直接加载它。

### 对应代码位置

- `app/train.py` 第 177-180 行
- `app/finetune.py` 第 203-206 行

### 这一步的目的

- 保留当前最好的模型
- 避免下次启动后训练结果丢失

---

## 十二、进阶实操：修改参数再跑一次

如果你已经成功跑通一次，可以尝试改参数再试一遍。

### 你可以改哪些参数

- `--epochs`：训练轮数
- `--batch-size`：每次送入几张图
- `--lr`：学习率
- `--unfreeze-features`：是否解冻特征提取层
- `--num-classes`：类别数检查
- `--output`：模型保存路径

### 示例 1：多训练几轮

```bash
python app/finetune.py --data-dir data --epochs 5 --batch-size 16 --num-workers 0
```

### 示例 2：自己指定类别数

```bash
python app/finetune.py --data-dir data --num-classes 2 --epochs 3
```

### 示例 3：解冻特征层一起训练

```bash
python app/train.py --data-dir data --epochs 3 --unfreeze-features
```

### 注意

解冻特征层会让训练更“自由”，但也更容易过拟合。  
新手建议先冻结特征层，把流程跑通再尝试解冻。

---

## 十三、推荐你按照这个顺序做

### 第一步：检查数据目录

确认 `data/train` 和 `data/val` 都存在，并且每个目录下有类别文件夹。

### 第二步：先跑 1 轮

先执行：

```bash
python app/finetune.py --data-dir data --epochs 1 --batch-size 16 --num-workers 0
```

目的只有一个：确认代码能跑通。

### 第三步：看输出

检查：

- 类别是否正确
- loss 是否有数值
- accuracy 是否在正常范围
- 是否生成 `finetuned_mobilenet.pth`

### 第四步：再正式训练

确认没问题后，把 `--epochs` 调到 3 或 5，再正式训练一轮。

---

## 十四、常见问题

### 问题 1：提示数据目录不存在

检查：

- `data` 文件夹是否真的存在
- 当前终端是不是在项目根目录

### 问题 2：提示类别数不匹配

检查：

- `data/train` 下到底有几个类别文件夹
- 你传入的 `--num-classes` 是否和实际一致

### 问题 3：程序能运行，但准确率很低

检查：

- 数据量是不是太少
- 类别图片是否太混乱
- 训练轮数是不是太少

### 问题 4：Windows 上 DataLoader 报错

建议把：

```bash
--num-workers 0
```

保留为 0。

---

## 十五、本节课你最终应该交什么

完成后，请确认你能提交以下内容：

1. 一份整理好的 `data/train` 和 `data/val` 数据目录。
2. 一次成功运行的训练截图。
3. 生成的模型文件 `finetuned_mobilenet.pth`。
4. 你自己记录的训练结果，比如：
   - epochs
   - batch size
   - learning rate
   - 最终验证准确率

---

## 十六、最后总结

这节课你真正学到的是下面这条完整流程：

1. 准备数据目录
2. 读取数据集
3. 使用训练集和验证集不同的预处理
4. 加载预训练 MobileNetV3
5. 冻结特征层，替换分类头
6. 定义损失函数和优化器
7. 训练和验证
8. 保存最好的模型

如果你已经能独立跑通一次训练，说明你已经掌握了微调的最基本流程。

