# AI Image Recognition

This project demonstrates image classification with PyTorch, torchvision, MobileNetV3, transfer learning, fine-tuning, and inference comparison.

The current lesson focuses on:

- Preparing an `ImageFolder` dataset.
- Fine-tuning MobileNetV3 on custom classes.
- Comparing pretrained ImageNet predictions with fine-tuned custom predictions.
- Understanding the difference between a classroom training script and a more complete engineering script.

## Project Structure

```text
AI-Image-Recognition/
  app/
    train.py                # Classroom training script
    finetune.py             # More complete fine-tuning script
    model_loader.py         # Pretrained model loading and classifier replacement
    dataset.py              # ImageFolder and DataLoader helpers
    transforms.py           # Train/validation preprocessing
    compare_inference.py    # Pretrained vs fine-tuned inference comparison
    classifier.py           # Original ImageNet inference script
    exceptions.py           # Custom exceptions
  courseware/
    lesson4_latest_finetuning.md
    lesson4_student_practice.md
  data/
    train/
    val/
  images/
    cat.jpg
  requirements.txt
```

## Environment Setup

Create and activate a conda environment:

```bash
conda create -n pytorch_env python=3.11
conda activate pytorch_env
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If PyTorch installation fails or you need GPU-specific packages, install PyTorch from the official selector first, then run `pip install -r requirements.txt` again.

## Dataset Format

The dataset must follow the `ImageFolder` structure:

```text
data/
  train/
    cat/
    dog/
  val/
    cat/
    dog/
```

Folder names are used as class labels. For example, `cat` is one class and `dog` is another class.

## Train for One Epoch

Use one epoch first to verify that the environment, dataset, and model can run end to end:

```bash
python app/finetune.py --data-dir data --epochs 1 --batch-size 16 --num-workers 0
```

## Train for Three Epochs

After the one-epoch check passes, run a normal classroom fine-tuning job:

```bash
python app/finetune.py --data-dir data --epochs 3 --batch-size 16 --num-workers 0
```

The trained weights are saved as:

```text
finetuned_mobilenet.pth
```

This file is ignored by Git because it is a generated model artifact.

## Compare Pretrained and Fine-Tuned Inference

After training, compare the original ImageNet model with the fine-tuned custom model:

```bash
python app/compare_inference.py
```

The script uses:

- `images/cat.jpg` as the default test image.
- `finetuned_mobilenet.pth` as the fine-tuned model.
- `data/train` folder names as custom class names.

## Classroom Script vs Engineering Script

`app/train.py` is the classroom version. It is shorter and focuses on the core training loop.

`app/finetune.py` is the more complete version. It adds:

- Dataset directory validation.
- Optional class-count validation.
- Explicit `ImageFolder` and `DataLoader` construction.
- Clearer error handling.

For teaching the training loop, start with `app/train.py`. For hands-on runs, prefer `app/finetune.py`.

## Notes for macOS Users

The code can run on macOS after installing the dependencies. On Apple Silicon, PyTorch may support MPS acceleration, but the current code uses CUDA when available and otherwise falls back to CPU.

The code still runs on CPU; it may just be slower.
