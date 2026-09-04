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

Before starting the Web app, it is recommended to verify that the terminal is
using the same conda environment:

```bash
python tests/test_env.py
```

The check must end with `Environment check passed.`. If `python` cannot import
`torch`, the terminal is not using the environment where the project packages
were installed. Run `conda activate pytorch_env` again and repeat the check.

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

Uploaded and rendered images are temporary files under `static/uploads/`.
The app removes files older than 24 hours and keeps at most 100 recent files.
This prevents repeated classroom practice from filling the repository disk.

## Included Runtime Assets

This repo intentionally includes a lightweight runnable package:

```text
data/oxford_pet_split/train/
models/oxford_pet_mobilenet_epoch1.pth
yolov8n.pt
```

The included `data/oxford_pet_split/train/` keeps the 37 Oxford-IIIT Pet class folders with a small representative sample per class. This is enough for the Web app to load class names and run inference with the included classifier checkpoint.

The full Oxford-IIIT Pet image set is not committed because the local full dataset is about 1.5GB. If you need full training data, regenerate it locally with `scripts/split_oxford_pet.py`.

The included sample data and classifier checkpoint are enough to run the Web
demo. Full-dataset training is optional and is only needed when students want
to train a new classifier checkpoint.

The included checkpoint reaches **90.19% Top-1 accuracy** on the 1,478-image
validation split (1,333 correct predictions). See `MODEL_CARD.md` for the
training configuration, limitations, and reproducible evaluation command.
Fine-tuned inference loads the complete local checkpoint with `weights=None`,
so it does not download ImageNet weights at runtime.

Oxford-IIIT Pet provides breed labels, not coat-color labels. The current model
can classify 37 cat and dog breeds, but it cannot reliably distinguish golden
shaded, silver shaded, and blue British Shorthair variants as separate classes.

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

The detector model is loaded once when the first detection request arrives.
Changing the confidence threshold in the Web page changes only that request's
filtering threshold; it does not reload the YOLO weights.
The project also disables Ultralytics runtime auto-install. Install dependencies
with `pip install -r requirements.txt` before starting the app, rather than
letting a user request run `pip` in the background.

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

Evaluate the included checkpoint:

```bash
python scripts/evaluate_classifier.py
```

Continue full-model fine-tuning from the included checkpoint:

```bash
python app/finetune.py --data-dir data/oxford_pet_split --epochs 3 --batch-size 32 --lr 0.0001 --weight-decay 0.0001 --label-smoothing 0.1 --unfreeze-features --resume models/oxford_pet_mobilenet_epoch1.pth --output models/oxford_pet_mobilenet_candidate.pth
```

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

If the classifier assets are missing, opening `/` still works and the overview
area shows which data or checkpoint is missing. Classification requests return
a readable 503 error until the assets are restored. Invalid or oversized image
uploads return a 400/413 response instead of a generic server error.

## Courseware

Important Lesson 7 documents:

- `courseware/lesson7_detection_integration_detailed.md`
- `courseware/lesson7_code_walkthrough.md`
- `courseware/lesson7_homework.md`
- `courseware/lesson7_practice_manual.md`
- `courseware/project_portfolio_operation_manual.md`
- `courseware/project_interview_guide.md`
- `courseware/AI图像识别项目课堂讲义.docx`
- `scripts/build_project_class_handout_docx.py`

Use `lesson7_practice_manual.md` when students need a step-by-step self-practice guide.
Use `project_portfolio_operation_manual.md` for the full portfolio reproduction,
testing, troubleshooting, and interview demonstration procedure. Use
`project_interview_guide.md` to prepare architecture explanations and technical
follow-up questions. Use `AI图像识别项目课堂讲义.docx` when a concise Word handout
is needed for classroom distribution.
