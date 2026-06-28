# AI Image Recognition

This repository is the classroom project for image classification and object detection.

It contains the code and lightweight runtime assets needed for students to clone the repo and directly run the Lesson 6/7 Web demo:

- MobileNetV3 fine-tuned pet classifier.
- Flask Web UI for image upload and prediction.
- YOLOv8 object detection workflow.
- OpenCV post-processing for drawing detection boxes.
- Lesson 6/7 courseware, homework, and practice manuals.

## Quick Start for Students

Open a terminal in the project root, then run:

```bash
conda activate pytorch_env
pip install -r requirements.txt
python app/web_app.py
```

Open the browser:

```text
http://127.0.0.1:5000
```

The page provides two workflows:

- `图像分类`: calls `/predict` and returns Top-3 class scores.
- `目标检测`: calls `/detect`, runs YOLOv8, draws boxes with OpenCV, and returns a rendered result image.

The Flask warning below is normal for classroom use:

```text
WARNING: This is a development server. Do not use it in a production deployment.
```

It only means the built-in Flask server is for local development, not public deployment.

## Included Runtime Assets

This repo intentionally includes a lightweight runnable package:

```text
data/oxford_pet_split/train/
models/oxford_pet_mobilenet_epoch1.pth
yolov8n.pt
```

The included `data/oxford_pet_split/train/` keeps the 37 Oxford-IIIT Pet class folders with a small representative sample per class. This is enough for the Web app to load class names and run inference with the included classifier checkpoint.

The full Oxford-IIIT Pet image set is not committed because the local full dataset is about 1.5GB. If you need full training data, regenerate it locally with `scripts/split_oxford_pet.py`.

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
    object_detector.py      # ObjectDetector interface and YOLOv8Detector
    preprocess.py           # Detection result drawing helpers with OpenCV
    web_app.py              # Flask web interface for classification and detection
    exceptions.py           # Custom exceptions
  courseware/
    lesson7_detection_integration_detailed.md
    lesson7_code_walkthrough.md
    lesson7_homework.md
    lesson7_practice_manual.md
  data/
    oxford_pet_split/
      train/
        Abyssinian/
        Bengal/
        ...
  models/
    oxford_pet_mobilenet_epoch1.pth
  templates/
    index.html
  static/
    uploads/
  requirements.txt
  yolov8n.pt
```

## Lesson 6: Classification Web UI

Lesson 6 connects the fine-tuned MobileNetV3 classifier to a Flask page.

Main files:

- `app/web_app.py`
- `templates/index.html`
- `app/model_loader.py`

Classification flow:

```text
Upload image
-> Flask /predict
-> preprocess image
-> MobileNetV3 classifier
-> Top-3 JSON results
-> Web page renders scores
```

## Lesson 7: YOLOv8 Detection Integration

Lesson 7 adds object detection beside the existing classifier.

Main files:

- `app/object_detector.py`
- `app/preprocess.py`
- `app/web_app.py`
- `templates/index.html`

Detection flow:

```text
Upload image
-> Flask /detect
-> YOLOv8Detector detects objects
-> OpenCV draws boxes and labels
-> Flask returns JSON and rendered image URL
-> Web page displays detection results
```

YOLOv8 and OpenCV have different jobs:

- YOLOv8 identifies objects and returns labels, confidence scores, and box coordinates.
- OpenCV reads the image, draws boxes/text, and saves the rendered result image.

Lesson 7 does not fine-tune YOLO. It directly uses the pretrained `yolov8n.pt` model for inference.

## Dataset Format

The classifier uses `torchvision.datasets.ImageFolder`, so the data folder must follow this structure:

```text
data/oxford_pet_split/
  train/
    Abyssinian/
    Bengal/
    ...
```

Folder names become class labels. The included lightweight dataset preserves the same class-folder structure expected by the included model checkpoint.

To recreate a full split from the original Oxford-IIIT `images` folder:

```bash
python scripts/split_oxford_pet.py --images-dir data/images --output-dir data/oxford_pet_split --val-ratio 0.2 --seed 42 --overwrite
```

## Optional: Fine-Tune Again

If you have the full dataset locally, run:

```bash
python app/finetune.py --data-dir data/oxford_pet_split --epochs 1 --batch-size 16 --num-workers 0
```

The generated checkpoint can be saved as:

```text
models/oxford_pet_mobilenet_epoch1.pth
```

For the classroom Web demo, retraining is not required because a checkpoint is already included.

## Common Checks

Confirm the classifier checkpoint exists:

```bash
python -c "from pathlib import Path; print(Path('models/oxford_pet_mobilenet_epoch1.pth').exists())"
```

Confirm YOLOv8 weight exists:

```bash
python -c "from pathlib import Path; print(Path('yolov8n.pt').exists())"
```

Confirm the class folders exist:

```bash
python -c "from pathlib import Path; print(Path('data/oxford_pet_split/train').exists())"
```

## Courseware

Important Lesson 7 documents:

- `courseware/lesson7_detection_integration_detailed.md`
- `courseware/lesson7_code_walkthrough.md`
- `courseware/lesson7_homework.md`
- `courseware/lesson7_practice_manual.md`

Use `lesson7_practice_manual.md` when students need a step-by-step self-practice guide.

