"""
classifier.py - 图像分类推理
"""
import os
import torch
import cv2
from torchvision import transforms

try:
    from app.model_loader import MobileNetV3Loader
    from app.exceptions import ModelLoadException, ImageLoadException
except ModuleNotFoundError:  # Allows: python app/classifier.py
    from model_loader import MobileNetV3Loader
    from exceptions import ModelLoadException, ImageLoadException


def load_imagenet_classes():
    """Read labels bundled in TorchVision metadata without an import-time request."""

    from torchvision.models import MobileNet_V3_Large_Weights

    return MobileNet_V3_Large_Weights.IMAGENET1K_V1.meta["categories"]


def load_and_predict(image_path: str):
    """对一张图片进行识别，返回 Top-5 预测结果"""

    # 前置条件检查
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # 1. 加载模型
    loader = MobileNetV3Loader()
    try:
        model = loader.load_model()
    except ModelLoadException as e:
        print(f"ERROR: {e}")
        return None

    # 2. 用 OpenCV 读取图片
    img = cv2.imread(image_path)
    if img is None:
        raise ImageLoadException(f"Unable to read image: {image_path}")

    # 3. 图像预处理
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    img_tensor = preprocess(img).unsqueeze(0)

    # 4. 推理
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    # 5. 取 Top-5
    imagenet_classes = load_imagenet_classes()
    top5_prob, top5_idx = torch.topk(probabilities, min(5, len(imagenet_classes)))

    results = []
    print("\nClassification results (Top-5):")
    print("-" * 40)
    for i in range(len(top5_idx)):
        label = imagenet_classes[int(top5_idx[i])]
        score = top5_prob[i].item() * 100
        print(f"  {i+1}. {label:30s} {score:6.2f}%")
        results.append((label, score))
    print("-" * 40)
    return results


if __name__ == "__main__":
    load_and_predict("images/cat.jpg")
