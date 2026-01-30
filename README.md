# Crack Detection Using Deep Learning for Edge Devices

A comprehensive study evaluating CNN, MobileNetV2, and Vision Transformer (ViT) architectures for automated multi-class bridge crack detection on edge devices.

**Paper**: Image-based Non-invasive Bridge Crack Detection Model for Edge Devices (RAICS 2025)

**Authors**: Athul Joe Palliparambil, Kevin Benty, Alan Cherian, Rajeev Rajan  
**Institution**: Government Engineering College, Idukki  
**University**: APJ Abdul Kalam Technological University, Thiruvananthapuram, India

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Models Implemented](#models-implemented)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Deployment on Edge Devices](#deployment-on-edge-devices)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## 🎯 Overview

This project addresses automated crack detection in bridge and concrete structures using deep learning. The study conducts a comprehensive evaluation of three distinct architectures:

1. **Custom 3-Layer CNN** - Lightweight baseline model
2. **MobileNetV2** - Efficient mobile-optimized architecture
3. **Vision Transformer (ViT)** - State-of-the-art attention-based model

The project focuses on real-world deployment on edge devices like Raspberry Pi while maintaining high accuracy.

---

## ✨ Key Features

- **Multi-class Classification**: Categorizes cracks into 4 severity levels:
  - No Crack
  - Light Crack
  - Medium Crack
  - Heavy Crack

- **Edge Device Optimization**: Successfully deployed on Raspberry Pi 4B with INT8 quantization
- **Balanced Dataset**: 20,020 training, 7,230 validation, 12,780 test images
- **Data Augmentation**: Rotation, flipping, zooming, brightness adjustment
- **Comprehensive Evaluation**: Metrics include accuracy, precision, recall, F1-score, and confusion matrices

---

## 🧠 Models Implemented

### 1. Custom 3-Layer CNN
- **Accuracy**: 84.88%
- **F1-Score**: 0.78
- **Parameters**: ~40K
- **Best For**: Resource-constrained environments

```
Input → Conv2D(32) → MaxPool → Conv2D(64) → MaxPool 
→ Conv2D(128) → MaxPool → Dense(512) → Dense(4)
```

### 2. MobileNetV2
- **Accuracy**: 86.99%
- **F1-Score**: 0.79
- **Quantized Size**: 18.7 MB (INT8)
- **Best For**: Optimal balance of accuracy and efficiency

Uses inverted residual bottleneck blocks and depthwise separable convolutions.

### 3. Vision Transformer (ViT)
- **Accuracy**: 90.03%
- **F1-Score**: 0.77
- **Parameters**: ~86M
- **Best For**: Maximum accuracy scenarios

Applies transformer architecture with multi-head self-attention for global context.

---

## 📊 Results

| Model | Accuracy | F1-Score | Model Size | Inference Time (CPU) | Best For |
|-------|----------|----------|------------|----------------------|----------|
| CNN | 84.88% | 0.78 | 1.2 MB | ~500 ms | Edge (minimal resources) |
| MobileNetV2 | 86.99% | 0.79 | 18.7 MB | ~200 ms | Edge (balanced) |
| ViT | 90.03% | 0.77 | 350+ MB | ~1-2 sec | Cloud/Server |

### Performance by Crack Severity (MobileNetV2)

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Heavy Crack | 0.91 | 0.87 | 0.90 |
| Light Crack | 0.63 | 0.52 | 0.58 |
| Medium Crack | 0.65 | 0.71 | 0.69 |
| No Crack | 0.99 | 0.99 | 0.99 |

---

## 📁 Project Structure

```
Crack_Detection/
│
├── README.md                          # Project documentation
├── PAPER.md                           # Paper summary and key findings
├── LICENSE                            # License file
├── requirements.txt                   # Python dependencies
│
├── data/
│   ├── README.md                      # Dataset documentation
│   ├── download_dataset.py            # Script to download Kaggle dataset
│   └── data_augmentation.py           # Data augmentation utilities
│
├── models/
│   ├── cnn_model.py                   # 3-Layer CNN implementation
│   ├── mobilenetv2_model.py           # MobileNetV2 implementation
│   ├── vit_model.py                   # Vision Transformer implementation
│   └── utils.py                       # Common utilities
│
├── training/
│   ├── train.py                       # Main training script
│   ├── hyperparameters.json           # Training configurations
│   ├── callbacks.py                   # Custom training callbacks
│   └── utils.py                       # Training utilities
│
├── evaluation/
│   ├── evaluate.py                    # Evaluation script
│   ├── metrics.py                     # Custom metrics
│   ├── visualization.py               # Confusion matrices, plots
│   └── results_analysis.py            # Detailed results analysis
│
├── deployment/
│   ├── README.md                      # Deployment guide
│   ├── quantize_model.py              # Model quantization script
│   ├── convert_to_tflite.py           # TensorFlow Lite conversion
│   ├── edge_inference.py              # Raspberry Pi inference script
│   └── requirements_rpi.txt           # RPi-specific dependencies
│
├── notebooks/
│   ├── 01_data_exploration.ipynb      # Dataset analysis
│   ├── 02_model_training.ipynb        # Training walkthrough
│   ├── 03_results_analysis.ipynb      # Comprehensive results analysis
│   └── 04_deployment_demo.ipynb       # Deployment example
│
├── configs/
│   ├── cnn_config.yaml                # CNN configuration
│   ├── mobilenetv2_config.yaml        # MobileNetV2 configuration
│   └── vit_config.yaml                # ViT configuration
│
├── scripts/
│   ├── download_and_prepare_data.sh   # Bash script for data prep
│   ├── train_all_models.sh            # Train all models script
│   └── run_evaluation.sh              # Run evaluation script
│
└── docs/
    ├── INSTALLATION.md                # Detailed installation guide
    ├── USAGE.md                       # Usage instructions
    ├── API.md                         # API documentation
    ├── RESULTS.md                     # Detailed results
    └── DEPLOYMENT.md                  # Edge deployment guide
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip or conda
- GPU recommended (CUDA 11.0+ for NVIDIA GPUs)

### Step 1: Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/Crack_Detection.git
cd Crack_Detection
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download Dataset
```bash
python data/download_dataset.py
```

For detailed installation instructions, see [INSTALLATION.md](docs/INSTALLATION.md)

---

## 💻 Usage

### Training a Model

#### Train Single Model
```bash
python training/train.py --model mobilenetv2 --epochs 30 --batch-size 32
```

#### Train All Models
```bash
bash scripts/train_all_models.sh
```

### Evaluate Models
```bash
python evaluation/evaluate.py --model mobilenetv2 --test-dir data/test
```

### Run Inference
```bash
python inference.py --model mobilenetv2 --image path/to/image.jpg
```

For detailed usage examples, see [USAGE.md](docs/USAGE.md)

---

## 📊 Dataset

**Source**: Kaggle Concrete Crack Images Dataset + Real-world images

**Composition**:
- Training: 20,020 images
- Validation: 7,230 images
- Testing: 12,780 images
- **Total**: 40,030 images

**Classes**:
- No Crack
- Light Crack
- Medium Crack
- Heavy Crack

**Image Specifications**:
- Original size: Various
- Resized to: 256×256 pixels
- Normalization: Applied

**Data Augmentation**:
- Rotation (0-45°)
- Flipping (horizontal, vertical)
- Zooming (0.8-1.2×)
- Brightness adjustment

To download the dataset:
```bash
python data/download_dataset.py
```

---

## 🔧 Deployment on Edge Devices

### Raspberry Pi 4B Deployment

#### Step 1: Prepare Model
```bash
python deployment/quantize_model.py --model mobilenetv2 --output quantized_model.tflite
```

#### Step 2: Install Raspberry Pi Dependencies
```bash
pip install -r deployment/requirements_rpi.txt
```

#### Step 3: Run Inference
```bash
python deployment/edge_inference.py --model quantized_model.tflite --image test_image.jpg
```

**Performance on Raspberry Pi 4B**:
- MobileNetV2: ~200-500 ms inference time
- Model Size: 18.7 MB (INT8 quantized)
- Real-time FPS: 3-5 FPS (with optimization)
- With Google Coral TPU: 30+ FPS

For detailed deployment guide, see [deployment/README.md](deployment/README.md)

---

## 📈 Results & Analysis

### Model Comparison

**Accuracy**: ViT > MobileNetV2 > CNN

**Efficiency**: CNN < MobileNetV2 ≪ ViT

**Recommendation**:
- **Best Overall Accuracy**: Vision Transformer (90.03%)
- **Best Edge Deployment**: MobileNetV2 (86.99% accuracy, 18.7 MB)
- **Minimal Resources**: 3-Layer CNN (84.88% accuracy, 1.2 MB)

### Detailed Results
See [RESULTS.md](docs/RESULTS.md) for:
- Confusion matrices
- Per-class performance analysis
- Training curves
- Comparative analysis

---

## 📚 Notebooks

Interactive Jupyter notebooks for exploration:

1. **01_data_exploration.ipynb** - Dataset statistics and visualization
2. **02_model_training.ipynb** - Step-by-step training walkthrough
3. **03_results_analysis.ipynb** - Comprehensive performance analysis
4. **04_deployment_demo.ipynb** - Edge device deployment example

Run notebooks:
```bash
jupyter notebook notebooks/
```

---

## 🔍 Key Findings

1. **Vision Transformer** achieves highest accuracy (90.03%) due to global context via self-attention
2. **MobileNetV2** offers best balance (86.99% accuracy, lightweight for edge)
3. **Data augmentation** crucial for handling class imbalance
4. **INT8 quantization** reduces model size by 95% with minimal accuracy loss
5. **Edge deployment** viable with optimizations on Raspberry Pi 4B

---

## 📖 Citation

If you use this project in your research, please cite the paper:

```bibtex
@inproceedings{palliparambil2025crack,
  title={Image-based Non-invasive Bridge Crack Detection Model for Edge Devices},
  author={Palliparambil, Athul Joe and Benty, Kevin and Cherian, Alan and Rajan, Rajeev},
  booktitle={RAICS 2025},
  year={2025},
  organization={APJ Abdul Kalam Technological University}
}
```

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📧 Contact

**Corresponding Author**: Rajeev Rajan  
**Email**: rajeev@gecidukki.ac.in  
**Institution**: Government Engineering College, Idukki

For questions or suggestions about this project, please open an issue on GitHub.

---

## 🙏 Acknowledgments

- Kaggle Concrete Crack Images Dataset
- TensorFlow and PyTorch communities
- APJ Abdul Kalam Technological University
- Government Engineering College, Idukki

---

## 📖 Additional Resources

- [Paper Summary](PAPER.md)
- [Installation Guide](docs/INSTALLATION.md)
- [Usage Guide](docs/USAGE.md)
- [Deployment Guide](deployment/README.md)
- [API Documentation](docs/API.md)
- [Results Analysis](docs/RESULTS.md)

---

**Last Updated**: January 2025  
**Status**: Active Development
