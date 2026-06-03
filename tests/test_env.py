"""
test_env.py - 环境验证小脚本
运行方法：在终端（已激活虚拟环境）执行 python app/test_env.py
"""
import os
import sys

print("=" * 40)
print("🔍 正在检查核心库导入...")

try:
    import cv2
    print(f"✅ OpenCV 版本: {cv2.__version__}")
except Exception as e:
    print(f"❌ OpenCV 导入失败: {e}")
    sys.exit(1)

try:
    import tensorflow as tf
    print(f"✅ TensorFlow 版本: {tf.__version__}")
except Exception as e:
    print(f"❌ TensorFlow 导入失败: {e}")
    sys.exit(1)

try:
    import flask
    print(f"✅ Flask 版本: {flask.__version__}")
except Exception as e:
    print(f"❌ Flask 导入失败: {e}")
    sys.exit(1)

print("✅ 所有核心库导入成功！\n")

print("=" * 40)
print("🖼️  正在测试 OpenCV 图片读取...")

IMAGE_PATH = os.path.join("images", "sample.jpg")   # 请先在 images/ 里放一张 sample.jpg

if not os.path.exists(IMAGE_PATH):
    print(f"⚠️  未找到测试图片: {IMAGE_PATH}")
    print("   请在 images 文件夹里放一张 .jpg 图片并命名为 sample.jpg")
else:
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"❌ 图片读取失败，可能是文件损坏或格式不支持")
    else:
        h, w, c = img.shape
        print(f"📷 图片路径: {IMAGE_PATH}")
        print(f"📐 尺寸: 宽={w}px, 高={h}px, 通道数={c}")

        resized = cv2.resize(img, (224, 224))
        output = os.path.join("images", "test_resized.jpg")
        cv2.imwrite(output, resized)
        print(f"💾 已生成 224x224 缩放图: {output}")

print("\n" + "=" * 40)
print("🎉 环境验证完成！如果没有看到 ❌，你的 AI 开发环境一切就绪。")