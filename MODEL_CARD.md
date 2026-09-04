# MobileNetV3 Pet Classifier Model Card

## Purpose

The checkpoint at `models/oxford_pet_mobilenet_epoch1.pth` classifies one image
into one of the 37 Oxford-IIIT Pet breed classes. It powers the Flask `/predict`
endpoint and the classification tab in the Web interface.

## Training Data

- Dataset: Oxford-IIIT Pet
- Training images: 5,912
- Validation images: 1,478
- Classes: 37 cat and dog breeds
- Split seed: 42

The dataset labels animal breeds. It does not provide separate labels for coat
colors such as golden shaded, silver shaded, or blue. The checkpoint therefore
must not be presented as a coat-color classifier.

## Training Configuration

The previous checkpoint was continued for three full-model fine-tuning epochs:

- Architecture: MobileNetV3-Large
- Optimizer: AdamW
- Initial learning rate: 0.0001
- Weight decay: 0.0001
- Label smoothing: 0.1
- Batch size: 32
- Scheduler: cosine annealing
- Selection rule: highest validation Top-1 accuracy

Reproduction command:

```bash
python app/finetune.py --data-dir data/oxford_pet_split --epochs 3 --batch-size 32 --num-workers 0 --lr 0.0001 --weight-decay 0.0001 --label-smoothing 0.1 --unfreeze-features --resume models/oxford_pet_mobilenet_epoch1.pth --output models/oxford_pet_mobilenet_candidate.pth
```

## Evaluation

Run the checked-in evaluator:

```bash
python scripts/evaluate_classifier.py
```

Results on the 1,478-image validation split:

| Checkpoint | Top-1 accuracy |
| --- | ---: |
| Previous classroom checkpoint | 81.46% |
| Current checkpoint | 90.19% |

The current checkpoint correctly classified 1,333 of 1,478 validation images,
an improvement of 8.73 percentage points.

## Offline Behavior

Inference creates the MobileNetV3 structure with `weights=None` and then loads
the complete local checkpoint. Once project dependencies are installed, using
the fine-tuned classifier does not require downloading ImageNet weights.

## Limitations

- The model can still make confident mistakes on images unlike its training set.
- It predicts exactly one of the 37 known breeds, even for unrelated images.
- Confidence is a relative softmax score, not a guarantee of correctness.
- Golden, silver, and blue British Shorthair variants are not separate labels.
- A dedicated coat-color task requires a separately labeled dataset and model.
