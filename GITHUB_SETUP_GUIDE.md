# GitHub Repository Setup Guide for Crack_Detection

## 📋 Complete Step-by-Step Instructions

This guide will help you create and manage your GitHub repository systematically from start to finish.

---

## PHASE 1: GitHub Repository Creation

### Step 1: Create Repository on GitHub

1. **Go to GitHub.com** and log in to your account
2. **Click "+" button** (top right) → **"New repository"**
3. **Fill in repository details**:
   - **Repository name**: `Crack_Detection`
   - **Description**: `Image-based Non-invasive Bridge Crack Detection Model for Edge Devices. Evaluates CNN, MobileNetV2, and Vision Transformer architectures for multi-class crack detection on edge devices.`
   - **Visibility**: Public (recommended for research)
   - **Add README**: ✓ (but we'll replace it)
   - **Add .gitignore**: Select `Python`
   - **Add license**: `MIT License`
4. **Click "Create repository"**

### Step 2: Clone Repository Locally

```bash
git clone https://github.com/YOUR_USERNAME/Crack_Detection.git
cd Crack_Detection
```

---

## PHASE 2: Directory Structure Setup

### Step 3: Create Directory Structure

Run these commands to create the project structure:

```bash
# Create main directories
mkdir -p data/raw data/processed
mkdir -p models/pretrained models/saved_models models/checkpoints
mkdir -p training
mkdir -p evaluation
mkdir -p deployment
mkdir -p notebooks
mkdir -p configs
mkdir -p scripts
mkdir -p docs
mkdir -p results

# Create subdirectories
mkdir -p evaluation/results
mkdir -p evaluation/visualizations
mkdir -p deployment/models_tflite
mkdir -p deployment/models_quantized
```

### Step 4: Add Key Files

Replace the default files created by GitHub:

```bash
# Remove default files (optional)
rm README.md  # We'll add our own

# Copy the files you've prepared
# (README.md, requirements.txt, LICENSE, PAPER.md, .gitignore)
```

---

## PHASE 3: Core Model Implementation

### Step 5: Create Model Files

Create `models/cnn_model.py`:
```python
import tensorflow as tf
from tensorflow.keras import layers, models

def create_cnn_model(input_shape=(256, 256, 3), num_classes=4):
    """
    Create a simple 3-layer CNN model for crack detection
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # Block 1
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 2
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Block 3
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model
```

Create `models/mobilenetv2_model.py`:
```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

def create_mobilenetv2_model(input_shape=(224, 224, 3), num_classes=4):
    """
    Create MobileNetV2 model with transfer learning
    """
    # Load pre-trained MobileNetV2
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model
    base_model.trainable = False
    
    # Add custom layers
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model
```

Create `models/vit_model.py`:
```python
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

class PatchEmbedding(layers.Layer):
    def __init__(self, patch_size, embedding_dim):
        super().__init__()
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
    
    def build(self, input_shape):
        self.projection = layers.Dense(self.embedding_dim)
    
    def call(self, x):
        B, H, W, C = x.shape
        patches = tf.image.extract_patches(
            x, 
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        patches = tf.reshape(patches, [B, -1, self.patch_size**2 * C])
        embeddings = self.projection(patches)
        return embeddings

def create_vit_model(input_shape=(224, 224, 3), num_classes=4, patch_size=16):
    """
    Create Vision Transformer model
    """
    model = models.Sequential([
        # Input
        layers.Input(shape=input_shape),
        PatchEmbedding(patch_size, 768),
        # Add your ViT layers here
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model
```

---

## PHASE 4: Training Script

Create `training/train.py`:
```python
import argparse
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn_model import create_cnn_model
from models.mobilenetv2_model import create_mobilenetv2_model

def train_model(model_name, epochs=30, batch_size=32):
    """
    Train the specified model
    """
    print(f"Training {model_name}...")
    
    # Create model
    if model_name == 'cnn':
        model = create_cnn_model()
    elif model_name == 'mobilenetv2':
        model = create_mobilenetv2_model()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(
            f'models/checkpoints/{model_name}_best.h5',
            monitor='val_accuracy',
            save_best_only=True
        )
    ]
    
    # Load and prepare data
    # (Implement your data loading logic)
    
    # Train
    history = model.fit(
        # X_train, y_train,
        # validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )
    
    return model, history

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train crack detection model')
    parser.add_argument('--model', choices=['cnn', 'mobilenetv2', 'vit'], 
                        default='mobilenetv2')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    
    args = parser.parse_args()
    
    train_model(args.model, args.epochs, args.batch_size)
```

---

## PHASE 5: Evaluation Script

Create `evaluation/evaluate.py`:
```python
import argparse
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model_path, test_data, test_labels, class_names):
    """
    Evaluate model on test data
    """
    # Load model
    model = tf.keras.models.load_model(model_path)
    
    # Predict
    predictions = model.predict(test_data)
    pred_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(test_labels, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average='weighted')
    recall = recall_score(true_labels, pred_labels, average='weighted')
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    
    # Print results
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('evaluation/results/confusion_matrix.png')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--test-dir', type=str, required=True)
    
    args = parser.parse_args()
    
    # Load test data (implement your logic)
    # results = evaluate_model(args.model, test_data, test_labels, class_names)
```

---

## PHASE 6: Data Management

Create `data/download_dataset.py`:
```python
import os
import zipfile
from pathlib import Path

def download_and_prepare_dataset():
    """
    Download Kaggle dataset and organize into train/val/test splits
    """
    print("Downloading Concrete Crack Images dataset from Kaggle...")
    
    # Using kaggle API (requires kaggle.json in ~/.kaggle/)
    os.system('kaggle datasets download -d arnavr10880/concrete-crack-images-for-classification')
    
    # Extract
    with zipfile.ZipFile('concrete-crack-images-for-classification.zip', 'r') as zip_ref:
        zip_ref.extractall('data/raw/')
    
    print("Dataset downloaded and extracted to data/raw/")
    print("Organize into train/val/test directories manually or use provided scripts")

if __name__ == '__main__':
    download_and_prepare_dataset()
```

---

## PHASE 7: Configuration Files

Create `configs/mobilenetv2_config.yaml`:
```yaml
# MobileNetV2 Configuration

model:
  name: MobileNetV2
  input_shape: [224, 224, 3]
  num_classes: 4
  weights: imagenet

training:
  optimizer: adam
  learning_rate: 0.001
  batch_size: 32
  epochs: 30
  validation_split: 0.2
  
callbacks:
  early_stopping:
    patience: 5
    monitor: val_loss
  model_checkpoint:
    monitor: val_accuracy
    save_best_only: true

data_augmentation:
  rotation_range: 45
  zoom_range: 0.2
  horizontal_flip: true
  vertical_flip: true
  brightness_range: [0.8, 1.2]
```

---

## PHASE 8: Documentation

Create comprehensive docs:

Create `docs/INSTALLATION.md`:
```markdown
# Installation Guide

## Requirements
- Python 3.8+
- pip or conda
- GPU (optional, but recommended)

## Steps

1. Clone repository
2. Create virtual environment
3. Install dependencies
4. Download dataset
5. Verify installation
```

Create `docs/USAGE.md`:
```markdown
# Usage Guide

## Training

```bash
python training/train.py --model mobilenetv2 --epochs 30
```

## Evaluation

```bash
python evaluation/evaluate.py --model models/mobilenetv2_best.h5 --test-dir data/test
```

## Inference

```bash
python inference.py --model mobilenetv2 --image test_image.jpg
```
```

Create `deployment/README.md`:
```markdown
# Edge Device Deployment

## Raspberry Pi 4B Setup

1. Convert model to TensorFlow Lite
2. Quantize model (INT8)
3. Deploy on Raspberry Pi

## Performance
- MobileNetV2: 3-5 FPS
- With Coral TPU: 30+ FPS
```

---

## PHASE 9: Git Workflow

### Step 6: Initialize and Commit

```bash
# Navigate to repository
cd Crack_Detection

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Project structure and documentation

- Add comprehensive README with project overview
- Add requirements.txt with all dependencies
- Add MIT License
- Add PAPER.md with detailed paper summary
- Create directory structure for models, training, evaluation
- Add gitignore for Python project"

# Push to GitHub
git push origin main
```

### Step 7: Organize with Branches

```bash
# Create feature branches for development
git checkout -b feature/models
git checkout -b feature/training-scripts
git checkout -b feature/evaluation
git checkout -b feature/deployment
git checkout -b feature/notebooks

# Work on each feature, then merge back to main
```

---

## PHASE 10: Adding Code Incrementally

### Step 8: Add Models

```bash
git checkout -b feature/models

# Add model files to models/ directory
# Commit
git add models/
git commit -m "Add model architectures

- Implement 3-layer CNN model
- Implement MobileNetV2 with transfer learning
- Implement Vision Transformer
- Add model utilities"

# Push and create PR
git push origin feature/models
```

### Step 9: Add Training Scripts

```bash
git checkout -b feature/training-scripts

# Add training files
git add training/
git commit -m "Add training pipeline

- Implement train.py with CLI interface
- Add training callbacks and utilities
- Add hyperparameter configurations
- Add data loading utilities"

git push origin feature/training-scripts
```

### Step 10: Add Evaluation

```bash
git checkout -b feature/evaluation

git add evaluation/
git commit -m "Add evaluation scripts

- Implement evaluation.py
- Add metric computation
- Add confusion matrix visualization
- Add results analysis tools"

git push origin feature/evaluation
```

---

## PHASE 11: Jupyter Notebooks

Create notebooks for different aspects:

Create `notebooks/01_data_exploration.ipynb`:
```
- Load dataset
- Visualize samples
- Class distribution analysis
- Image statistics
```

Create `notebooks/02_model_training.ipynb`:
```
- Train all three models
- Compare training curves
- Save trained models
```

Create `notebooks/03_results_analysis.ipynb`:
```
- Load test predictions
- Generate confusion matrices
- Per-class performance analysis
- Visualization
```

---

## PHASE 12: Final Polish

### Step 11: Update Repository Metadata

1. **Add Topics** on GitHub repo page:
   - deep-learning
   - crack-detection
   - edge-computing
   - computer-vision
   - mobiledev
   - tensorflow

2. **Add Repository Description**:
   ```
   Image-based Bridge Crack Detection using CNN, MobileNetV2, and Vision Transformer for edge devices
   ```

3. **Add Links** in About section:
   - Link to paper
   - Link to dataset
   - Contact email

### Step 12: Create Release

```bash
# Tag first release
git tag v1.0.0
git push origin v1.0.0

# On GitHub, create release from this tag with:
- Release notes
- Model weights (if size allows)
- Benchmark results
```

---

## PHASE 13: Maintenance & Updates

### Step 13: Ongoing Development

**For adding new code**:

```bash
# Pull latest
git pull origin main

# Create feature branch
git checkout -b feature/new-feature

# Make changes, test, commit
git add .
git commit -m "Add new feature description"

# Push and create pull request
git push origin feature/new-feature
```

**Create Pull Request Template** (`.github/pull_request_template.md`):
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update

## Testing
How have you tested this?

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review done
- [ ] Comments added
- [ ] Tests updated
```

---

## Complete File Checklist

```
Crack_Detection/
├── ✓ README.md (comprehensive overview)
├── ✓ PAPER.md (detailed paper summary)
├── ✓ requirements.txt (all dependencies)
├── ✓ LICENSE (MIT)
├── ✓ .gitignore (Python template)
│
├── data/
│   ├── README.md
│   ├── download_dataset.py
│   └── data_augmentation.py
│
├── models/
│   ├── __init__.py
│   ├── cnn_model.py
│   ├── mobilenetv2_model.py
│   ├── vit_model.py
│   └── utils.py
│
├── training/
│   ├── train.py
│   ├── callbacks.py
│   ├── hyperparameters.json
│   └── utils.py
│
├── evaluation/
│   ├── evaluate.py
│   ├── metrics.py
│   ├── visualization.py
│   └── results_analysis.py
│
├── deployment/
│   ├── README.md
│   ├── quantize_model.py
│   ├── convert_to_tflite.py
│   ├── edge_inference.py
│   └── requirements_rpi.txt
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_results_analysis.ipynb
│   └── 04_deployment_demo.ipynb
│
├── configs/
│   ├── cnn_config.yaml
│   ├── mobilenetv2_config.yaml
│   └── vit_config.yaml
│
├── docs/
│   ├── INSTALLATION.md
│   ├── USAGE.md
│   ├── API.md
│   ├── RESULTS.md
│   └── DEPLOYMENT.md
│
└── scripts/
    ├── download_and_prepare_data.sh
    ├── train_all_models.sh
    └── run_evaluation.sh
```

---

## Git Commit Messages Template

Follow this format for commits:

```
<type>: <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style
- `refactor`: Code refactor
- `test`: Tests
- `chore`: Maintenance

**Example**:
```
feat: Add MobileNetV2 model implementation

- Implement MobileNetV2 architecture with transfer learning
- Add model configuration yaml
- Include usage documentation

Closes #5
```

---

## README Badges

Add these to your README.md for visibility:

```markdown
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![TensorFlow 2.13+](https://img.shields.io/badge/TensorFlow-2.13+-FF6F00.svg)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper Status](https://img.shields.io/badge/Paper-RAICS%202025-success)]()
```

---

## Tips for GitHub Success

1. **Write clear commit messages** - helps others understand history
2. **Keep README updated** - first thing people read
3. **Add examples** - show how to use your code
4. **Include citations** - respect academic work
5. **Use branches** - keep main stable
6. **Write tests** - ensure code reliability
7. **Document API** - make it usable
8. **Add badges** - shows status at a glance
9. **Respond to issues** - engage with community
10. **Release versions** - track evolution

---

This comprehensive guide will help you systematically build and maintain your GitHub repository for the Crack_Detection project!
