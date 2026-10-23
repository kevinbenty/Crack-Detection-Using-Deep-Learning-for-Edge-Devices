# Crack Detection Using Deep Learning for Edge Devices

Code for the paper **"Image-based Non-invasive Bridge Crack Detection Model for Edge Devices"** (RAICS 2025)
by Athul Joe Palliparambil, Kevin Benty, Alan Cherian and Rajeev Rajan, Government Engineering College, Idukki.

The project compares three architectures for classifying concrete/bridge cracks into four severity levels
(**no crack, light crack, medium crack, heavy crack**) and deploys the most practical one, MobileNetV2,
on a Raspberry Pi 4B as an INT8 TensorFlow Lite model.

## Results reported in the paper

| Model | Accuracy | Macro F1 |
|-------|---------:|---------:|
| 3-layer CNN | 84.88% | 0.78 |
| MobileNetV2 | 86.99% | 0.79 |
| ViT-Base/16 | 90.03% | 0.77 |

MobileNetV2 gives the best balance of accuracy and efficiency. Quantized to INT8 TFLite (18.7 MB), it runs
on a Raspberry Pi 4B (4 GB). Per-class precision, recall and F1 are in Tables V–VII of the paper.

## Models

| Script | Model | Input | Details |
|--------|-------|-------|---------|
| `train_cnn.py` | 3-layer CNN | 256×256 | Conv2D 32/64/128 (3×3, ReLU) + 2×2 max pooling → Dense(512) → Dense(4). ~59M parameters, almost all in the Flatten→Dense(512) layer |
| `train_mobilenetv2.py` | MobileNetV2 | 224×224 | ImageNet-pretrained MobileNetV2 backbone → global average pooling → Dense(512) → Dense(4) |
| `train_vit.py` | ViT-Base/16 | 224×224 | 16×16 patches, 12 encoder blocks, 12 heads, embedding 768, MLP 3072; fine-tuned from `google/vit-base-patch16-224-in21k` |

All three use the training setup from the paper: Adam with initial learning rate 0.001 and batch size 32,
learning rate reduced on a validation plateau, up to 30 epochs, early stopping with patience 5, and the
checkpoint with the best validation accuracy kept. Training images are augmented with rotation, flips,
zoom and brightness changes.

## Dataset

The paper uses 20,020 training, 7,230 validation and 12,780 test images, starting from the
[Concrete Crack Images for Classification](https://www.kaggle.com/datasets/arnavr10880/concrete-crack-images-for-classification)
dataset plus real-world images. The dataset is not included in this repository. Arrange your images as:

```
data/
├── train/
│   ├── heavy_crack/
│   ├── light_crack/
│   ├── medium_crack/
│   └── no_crack/
├── val/      (same four folders)
└── test/     (same four folders)
```

## Installation

```bash
git clone https://github.com/kevinbenty/Crack-Detection-Using-Deep-Learning-for-Edge-Devices.git
cd Crack-Detection-Using-Deep-Learning-for-Edge-Devices
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.9–3.11 is recommended (TensorFlow 2.13–2.15). A GPU is strongly recommended for the ViT.

## Training and evaluation

```bash
python train_cnn.py         --data-dir data
python train_mobilenetv2.py --data-dir data
python train_vit.py         --data-dir data
```

Each script trains the model, then evaluates it on both the validation and the test set. Results are written
to `results/<model>/`:

- `test_report.txt`, `test_results.json`: accuracy, per-class precision/recall/F1 and macro averages
- `test_confusion_matrix.png`, `training_curves.png`
- the same files for the validation set (`val_*`)
- the trained model and `run_config.json`

Useful options: `--epochs`, `--batch-size`, `--learning-rate`, `--class-weights` (CNN and MobileNetV2),
`--train-backbone` (MobileNetV2). Run any script with `--help` for the full list.

## Raspberry Pi deployment

On the training machine, convert the trained MobileNetV2 to INT8 TensorFlow Lite:

```bash
python export_tflite.py --model results/mobilenetv2/mobilenetv2_model.keras --data-dir data
```

This writes `mobilenetv2_model_int8.tflite` and `labels.json`. Copy both and `rpi_inference.py` to the Pi, then:

```bash
pip3 install -r requirements_rpi.txt
python3 rpi_inference.py --model mobilenetv2_model_int8.tflite --labels labels.json image.jpg
python3 rpi_inference.py --model mobilenetv2_model_int8.tflite --labels labels.json --camera 0
```

## Repository structure

| File | Purpose |
|------|---------|
| `train_cnn.py`, `train_mobilenetv2.py`, `train_vit.py` | Train and evaluate each model |
| `keras_common.py` | Data loading, augmentation and training setup shared by the Keras models |
| `metrics.py` | Reports, confusion matrices and training curves shared by all models |
| `export_tflite.py` | INT8 TensorFlow Lite conversion |
| `rpi_inference.py` | Inference on a Raspberry Pi from images or a camera |

## Citation

```bibtex
@inproceedings{palliparambil2025crack,
  title     = {Image-based Non-invasive Bridge Crack Detection Model for Edge Devices},
  author    = {Palliparambil, Athul Joe and Benty, Kevin and Cherian, Alan and Rajan, Rajeev},
  booktitle = {RAICS 2025},
  year      = {2025}
}
```

## License

MIT. See [LICENSE](LICENSE).
