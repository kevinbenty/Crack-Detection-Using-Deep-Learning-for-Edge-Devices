# train_vit.py
"""
Vision Transformer (ViT) Model for Crack Detection
From: Image-based Non-invasive Bridge Crack Detection Model for Edge Devices (RAICS 2025)

This script implements Vision Transformer for multi-class crack detection.
The model achieves 90.03% accuracy but requires significant computational resources.

Key Features:
- Vision Transformer architecture (google/vit-base-patch16-224-in21k)
- Transfer learning from ImageNet-21K pre-training
- Hugging Face Transformers library integration
- Comprehensive evaluation metrics
- Model quantization support

Usage:
    python train_vit.py
    
    Or with custom parameters:
    python train_vit.py --epochs 30 --batch-size 16 --learning-rate 2e-4
    
Note:
    - Requires GPU for reasonable training times
    - Model size: ~350+ MB (not suitable for edge without quantization)
    - Best for cloud/server deployment
"""

# -------------------------------------------------------------------
# 1. IMPORT LIBRARIES
# -------------------------------------------------------------------
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
from sklearn.metrics import (
    confusion_matrix, 
    ConfusionMatrixDisplay, 
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from datasets import load_dataset
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    TrainingArguments,
    Trainer
)
import evaluate

# -------------------------------------------------------------------
# 2. DEFINE CONFIGURATION
# -------------------------------------------------------------------
MODEL_NAME = "google/vit-base-patch16-224-in21k"
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 30  # Paper used 30
LEARNING_RATE = 2e-4
DATA_DIR = "data"

# -------------------------------------------------------------------
# 3. ARGUMENT PARSER
# -------------------------------------------------------------------
def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train Vision Transformer for crack detection'
    )
    parser.add_argument('--model', type=str, default=MODEL_NAME,
                        help=f'HuggingFace model name (default: {MODEL_NAME})')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help=f'Number of training epochs (default: {EPOCHS})')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help=f'Batch size for training (default: {BATCH_SIZE})')
    parser.add_argument('--learning-rate', type=float, default=LEARNING_RATE,
                        help=f'Learning rate (default: {LEARNING_RATE})')
    parser.add_argument('--data-dir', type=str, default=DATA_DIR,
                        help=f'Path to data directory (default: {DATA_DIR})')
    parser.add_argument('--output-dir', type=str, default='results/vit',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    return parser.parse_args()

# -------------------------------------------------------------------
# 4. LOAD AND PREPARE DATASET
# -------------------------------------------------------------------
def load_and_prepare_dataset(data_dir):
    """
    Load dataset from directory structure and prepare for training
    
    Expected directory structure:
    data/
    ├── train/
    │   ├── class1/
    │   └── class2/
    └── test/
        ├── class1/
        └── class2/
    
    Args:
        data_dir: Base directory containing data
    
    Returns:
        Tuple of (dataset, processor, class_names)
    """
    print("Loading dataset...")
    
    # Load dataset using imagefolder feature
    dataset = load_dataset("imagefolder", data_dir=data_dir)
    
    # Get class names
    class_names = dataset['train'].features['label'].names
    print(f"Classes found: {class_names}")
    print(f"Dataset splits: {dataset.column_names}")
    
    # Load processor
    print(f"Loading ViT processor from {MODEL_NAME}...")
    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    
    return dataset, processor, class_names

# -------------------------------------------------------------------
# 5. PREPARE DATASET FOR TRAINING
# -------------------------------------------------------------------
def prepare_dataset_for_training(dataset, processor):
    """
    Apply preprocessing to dataset
    
    Args:
        dataset: HuggingFace dataset
        processor: ViT image processor
    
    Returns:
        Processed dataset with batched tensors
    """
    print("Preparing dataset with preprocessing...")
    
    def transform(example_batch):
        """Transform function to preprocess images"""
        # Convert images to RGB if necessary
        inputs = processor(
            [x.convert("RGB") for x in example_batch['image']],
            return_tensors='pt'
        )
        inputs['labels'] = example_batch['label']
        return inputs
    
    # Apply transformation
    prepared_dataset = dataset.with_transform(transform)
    
    return prepared_dataset

# -------------------------------------------------------------------
# 6. CREATE DATA COLLATOR
# -------------------------------------------------------------------
def create_collate_fn():
    """
    Create a collate function for batching
    
    Returns:
        Collate function
    """
    def collate_fn(batch):
        """Collate function for batching preprocessed data"""
        return {
            'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
            'labels': torch.tensor([x['labels'] for x in batch])
        }
    
    return collate_fn

# -------------------------------------------------------------------
# 7. CREATE MODEL
# -------------------------------------------------------------------
def create_vit_model(model_name, num_labels, class_names):
    """
    Create Vision Transformer model
    
    Args:
        model_name: HuggingFace model identifier
        num_labels: Number of output classes
        class_names: List of class names
    
    Returns:
        ViT model for image classification
    """
    print(f"Loading Vision Transformer model: {model_name}")
    
    # Create label mappings
    id2label = {str(i): c for i, c in enumerate(class_names)}
    label2id = {c: str(i) for i, c in enumerate(class_names)}
    
    # Load model
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=False
    )
    
    print(f"Model loaded with {model.num_parameters()} parameters")
    
    return model

# -------------------------------------------------------------------
# 8. DEFINE METRICS
# -------------------------------------------------------------------
def create_compute_metrics_fn(num_labels):
    """
    Create metrics computation function
    
    Args:
        num_labels: Number of classes
    
    Returns:
        Function to compute metrics
    """
    # Load accuracy metric
    metric = evaluate.load("accuracy")
    
    def compute_metrics(p):
        """Compute accuracy metric"""
        predictions = np.argmax(p.predictions, axis=1)
        references = p.label_ids
        return metric.compute(predictions=predictions, references=references)
    
    return compute_metrics

# -------------------------------------------------------------------
# 9. CREATE TRAINING ARGUMENTS
# -------------------------------------------------------------------
def create_training_arguments(output_dir, epochs, batch_size, learning_rate):
    """
    Create training arguments for the Trainer
    
    Args:
        output_dir: Output directory
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
    
    Returns:
        TrainingArguments object
    """
    os.makedirs(output_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",  # Evaluate every epoch
        num_train_epochs=epochs,
        save_strategy="epoch",  # Save checkpoint every epoch
        save_total_limit=2,  # Keep only 2 best checkpoints
        logging_steps=10,
        learning_rate=learning_rate,
        warmup_ratio=0.1,  # Warmup 10% of steps
        weight_decay=0.01,  # L2 regularization
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        push_to_hub=False,
        seed=42,
    )
    
    return training_args

# -------------------------------------------------------------------
# 10. TRAIN MODEL
# -------------------------------------------------------------------
def train_vit_model(model, prepared_dataset, collate_fn, training_args, 
                    compute_metrics, num_labels):
    """
    Train Vision Transformer model using HuggingFace Trainer
    
    Args:
        model: ViT model
        prepared_dataset: Preprocessed dataset
        collate_fn: Collate function for batching
        training_args: Training arguments
        compute_metrics: Metrics computation function
        num_labels: Number of classes
    
    Returns:
        Trainer object with training history
    """
    print("Creating Trainer...")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=prepared_dataset["train"],
        eval_dataset=prepared_dataset.get("validation", prepared_dataset.get("test")),
    )
    
    print("Starting training...")
    trainer.train()
    
    return trainer

# -------------------------------------------------------------------
# 11. EVALUATE MODEL
# -------------------------------------------------------------------
def evaluate_vit_model(trainer, test_dataset, class_names, output_dir):
    """
    Evaluate trained ViT model on test set
    
    Args:
        trainer: Trainer object
        test_dataset: Test dataset
        class_names: List of class names
        output_dir: Output directory for results
    
    Returns:
        Dictionary with evaluation metrics
    """
    print("Evaluating on test data...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get predictions
    predictions = trainer.predict(test_dataset["test"])
    true_labels = predictions.label_ids
    predicted_labels = np.argmax(predictions.predictions, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    
    print(f"\nTest Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    # Classification report
    print("\n--- Classification Report ---")
    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names,
        zero_division=0
    )
    print(report)
    
    # Save report
    report_path = os.path.join(output_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Report saved to {report_path}")
    
    # Confusion matrix
    print("Generating confusion matrix...")
    cm = confusion_matrix(true_labels, predicted_labels)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    display.plot(xticks_rotation='vertical', cmap='Blues')
    plt.title("Vision Transformer Confusion Matrix", fontweight='bold')
    plt.tight_layout()
    
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved as {cm_path}")
    plt.close()
    
    return {
        'true_labels': true_labels,
        'predicted_labels': predicted_labels,
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': cm.tolist(),
        'report': report
    }

# -------------------------------------------------------------------
# 12. SAVE MODEL AND METADATA
# -------------------------------------------------------------------
def save_model_and_metadata(trainer, class_names, output_dir, metrics):
    """
    Save model and metadata
    
    Args:
        trainer: Trainer object
        class_names: List of class names
        output_dir: Output directory
        metrics: Dictionary with evaluation metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, 'vit_model')
    trainer.model.save_pretrained(model_path)
    print(f"Model saved to {model_path}")
    
    # Save processor
    processor_path = os.path.join(output_dir, 'processor')
    trainer.model.image_processor.save_pretrained(processor_path)
    print(f"Processor saved to {processor_path}")
    
    # Save metadata
    metadata = {
        'model_type': 'Vision Transformer (ViT)',
        'base_model': MODEL_NAME,
        'image_size': IMAGE_SIZE,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'epochs': EPOCHS,
        'classes': class_names,
        'num_classes': len(class_names),
        'timestamp': datetime.now().isoformat(),
        'input_shape': (IMAGE_SIZE, IMAGE_SIZE, 3),
        'output_shape': len(class_names),
        'total_parameters': trainer.model.num_parameters(),
        'paper_accuracy': '90.03%',
        'paper_f1_score': '0.77',
        'test_metrics': metrics
    }
    
    metadata_path = os.path.join(output_dir, 'model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_path}")

# -------------------------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------------------------
def main():
    """Main execution function"""
    args = parse_arguments()
    
    # Check device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("=" * 70)
    print("Vision Transformer for Crack Detection")
    print("From: Image-based Non-invasive Bridge Crack Detection")
    print("      Model for Edge Devices (RAICS 2025)")
    print("=" * 70)
    
    # Load dataset
    dataset, processor, class_names = load_and_prepare_dataset(args.data_dir)
    
    # Prepare dataset
    prepared_dataset = prepare_dataset_for_training(dataset, processor)
    
    # Create collate function
    collate_fn = create_collate_fn()
    
    # Create model
    num_labels = len(class_names)
    model = create_vit_model(args.model, num_labels, class_names)
    
    # Create metrics function
    compute_metrics = create_compute_metrics_fn(num_labels)
    
    # Create training arguments
    training_args = create_training_arguments(
        args.output_dir,
        args.epochs,
        args.batch_size,
        args.learning_rate
    )
    
    # Train model
    trainer = train_vit_model(
        model,
        prepared_dataset,
        collate_fn,
        training_args,
        compute_metrics,
        num_labels
    )
    
    # Evaluate model
    metrics = evaluate_vit_model(trainer, dataset, class_names, args.output_dir)
    
    # Save model and metadata
    save_model_and_metadata(trainer, class_names, args.output_dir, metrics)
    
    print("\n" + "=" * 70)
    print("Training and evaluation complete!")
    print(f"Results saved to {args.output_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()
