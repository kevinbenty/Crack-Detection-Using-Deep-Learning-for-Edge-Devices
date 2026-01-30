# train_mobilenetv2.py
"""
MobileNetV2 Model for Crack Detection with Improved Training
From: Image-based Non-invasive Bridge Crack Detection Model for Edge Devices (RAICS 2025)

This script implements MobileNetV2 with transfer learning for multi-class crack detection.
The model achieves 86.99% accuracy and is optimized for edge device deployment.

Key Features:
- Transfer learning from ImageNet weights
- Data augmentation to handle class imbalance
- Class weights to balance datasets
- Learning rate scheduling
- Early stopping and model checkpointing
- Comprehensive evaluation metrics

Usage:
    python train_mobilenetv2.py
    
    Or with custom parameters:
    python train_mobilenetv2.py --epochs 30 --batch-size 32
"""

# -------------------------------------------------------------------
# 1. IMPORT LIBRARIES
# -------------------------------------------------------------------
import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import class_weight
import json
from datetime import datetime

# -------------------------------------------------------------------
# 2. DEFINE CONFIGURATION
# -------------------------------------------------------------------
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30  # Paper used 30
LEARNING_RATE = 0.001
data_dir = "data"

# -------------------------------------------------------------------
# 3. ARGUMENT PARSER
# -------------------------------------------------------------------
def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train MobileNetV2 for crack detection with improved settings'
    )
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help=f'Number of training epochs (default: {EPOCHS})')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help=f'Batch size for training (default: {BATCH_SIZE})')
    parser.add_argument('--learning-rate', type=float, default=LEARNING_RATE,
                        help=f'Initial learning rate (default: {LEARNING_RATE})')
    parser.add_argument('--data-dir', type=str, default=data_dir,
                        help=f'Path to data directory (default: {data_dir})')
    parser.add_argument('--output-dir', type=str, default='results/mobilenetv2',
                        help='Output directory for results')
    parser.add_argument('--freeze-base', action='store_true', default=True,
                        help='Freeze base model layers (default: True)')
    
    return parser.parse_args()

# -------------------------------------------------------------------
# 4. LOAD AND PREPARE DATASET
# -------------------------------------------------------------------
def load_datasets(data_dir, image_size, batch_size):
    """
    Load training, validation, and test datasets
    
    Args:
        data_dir: Base directory containing data
        image_size: Tuple of (height, width)
        batch_size: Batch size for datasets
    
    Returns:
        Tuple of (train_dataset, validation_dataset, test_dataset, class_names)
    """
    print("Loading datasets...")
    
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "train"),
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=image_size,
        batch_size=batch_size
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "train"),
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=image_size,
        batch_size=batch_size
    )

    class_names = train_dataset.class_names
    print(f"Classes found: {class_names}")

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "test"),
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False
    )

    # Configure dataset for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    return train_dataset, validation_dataset, test_dataset, class_names

# -------------------------------------------------------------------
# 5. CALCULATE CLASS WEIGHTS
# -------------------------------------------------------------------
def calculate_class_weights(train_dataset):
    """
    Calculate class weights to handle dataset imbalance
    
    Args:
        train_dataset: Training dataset
    
    Returns:
        Dictionary mapping class index to weight
    """
    print("Calculating class weights to handle imbalance...")
    
    train_labels = []
    for _, labels in train_dataset:
        train_labels.extend(labels.numpy())

    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights_dict = dict(enumerate(class_weights))
    
    print("Calculated Class Weights:")
    for class_idx, weight in class_weights_dict.items():
        print(f"  Class {class_idx}: {weight:.4f}")
    
    return class_weights_dict

# -------------------------------------------------------------------
# 6. CREATE DATA AUGMENTATION
# -------------------------------------------------------------------
def create_data_augmentation():
    """
    Create data augmentation layer
    
    Data augmentation helps with:
    - Class imbalance
    - Model generalization
    - Robustness to variations
    
    Returns:
        Keras Sequential layer with augmentation
    """
    print("Creating data augmentation layer...")
    
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
    ], name="data_augmentation")
    
    return data_augmentation

# -------------------------------------------------------------------
# 7. CREATE THE MOBILENETV2 MODEL
# -------------------------------------------------------------------
def create_mobilenetv2_model(image_size, num_classes):
    """
    Create MobileNetV2 model with transfer learning
    
    Architecture:
    - Base model: MobileNetV2 (pre-trained on ImageNet)
    - Custom layers:
        * GlobalAveragePooling2D
        * Dropout(0.3)
        * Dense(512) with ReLU
        * Dense(num_classes) with Softmax
    
    Args:
        image_size: Tuple of (height, width)
        num_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    print("Building MobileNetV2 model with transfer learning...")
    
    # Preprocessing function
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    
    # Data augmentation
    data_augmentation = create_data_augmentation()
    
    # Load base model
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(image_size[0], image_size[1], 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze base model
    
    # Build custom model
    inputs = tf.keras.Input(shape=(image_size[0], image_size[1], 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    
    return model

# -------------------------------------------------------------------
# 8. COMPILE THE MODEL
# -------------------------------------------------------------------
def compile_model(model, learning_rate):
    """
    Compile the model with Adam optimizer
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for Adam optimizer
    """
    print("Compiling model...")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

# -------------------------------------------------------------------
# 9. CREATE CALLBACKS
# -------------------------------------------------------------------
def create_callbacks(output_dir):
    """
    Create training callbacks
    
    Callbacks:
    - ModelCheckpoint: Save best model
    - ReduceLROnPlateau: Reduce LR when loss plateaus
    - EarlyStopping: Stop when no improvement
    
    Args:
        output_dir: Directory to save checkpoints
    
    Returns:
        List of callbacks
    """
    os.makedirs(output_dir, exist_ok=True)
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, "best_model.keras"),
            save_best_only=True,
            monitor="val_loss",
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,  # Reduce LR by factor of 5
            patience=3,  # After 3 epochs with no improvement
            verbose=1,
            min_lr=1e-7
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,  # Stop if no improvement for 5 epochs
            verbose=1,
            restore_best_weights=True
        )
    ]
    
    return callbacks

# -------------------------------------------------------------------
# 10. TRAIN THE MODEL
# -------------------------------------------------------------------
def train_model(model, train_dataset, validation_dataset, epochs, 
                class_weights_dict, callbacks):
    """
    Train the model on the training dataset
    
    Args:
        model: Compiled Keras model
        train_dataset: Training dataset
        validation_dataset: Validation dataset
        epochs: Number of epochs to train
        class_weights_dict: Dictionary of class weights
        callbacks: List of training callbacks
    
    Returns:
        Training history
    """
    print("\nStarting training...")
    print(f"Training for {epochs} epochs with batch size {BATCH_SIZE}...\n")
    
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=validation_dataset,
        class_weight=class_weights_dict,
        callbacks=callbacks
    )
    
    print("\nTraining complete!")
    return history

# -------------------------------------------------------------------
# 11. EVALUATE AND VISUALIZE RESULTS
# -------------------------------------------------------------------
def evaluate_model(model, test_dataset, class_names, output_dir):
    """
    Evaluate model on test data and generate visualizations
    
    Args:
        model: Trained Keras model
        test_dataset: Test dataset
        class_names: List of class names
        output_dir: Directory to save outputs
    """
    print("\nEvaluating on test data...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get predictions
    true_labels = []
    predicted_labels = []
    
    y_pred = model.predict(test_dataset, verbose=1)
    predicted_labels = np.argmax(y_pred, axis=1)

    for _, labels in test_dataset:
        true_labels.extend(labels.numpy())

    # --- Classification Report ---
    print("\n--- Classification Report ---")
    report = classification_report(
        true_labels, 
        predicted_labels, 
        target_names=class_names
    )
    print(report)
    print("-" * 50)
    
    # Save report
    report_path = os.path.join(output_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Report saved to {report_path}")

    # --- Confusion Matrix ---
    print("\nGenerating confusion matrix...")
    cm = confusion_matrix(true_labels, predicted_labels)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    display.plot(xticks_rotation='vertical', cmap='Blues')
    plt.title("MobileNetV2 Confusion Matrix", fontweight='bold')
    plt.tight_layout()
    
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved as {cm_path}")
    plt.close()

    return {
        'true_labels': true_labels,
        'predicted_labels': predicted_labels,
        'confusion_matrix': cm.tolist(),
        'report': report
    }

# -------------------------------------------------------------------
# 12. PLOT TRAINING CURVES
# -------------------------------------------------------------------
def plot_training_curves(history, output_dir):
    """
    Plot training and validation accuracy/loss curves
    
    Args:
        history: Training history from model.fit()
        output_dir: Directory to save plots
    """
    print("Generating accuracy and loss plots...")
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy', linewidth=2, marker='o', markersize=3)
    plt.plot(val_acc, label='Validation Accuracy', linewidth=2, marker='s', markersize=3)
    plt.legend(loc='lower right', fontsize=10)
    plt.ylabel('Accuracy', fontsize=11)
    plt.xlabel('Epoch', fontsize=11)
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss', linewidth=2, marker='o', markersize=3)
    plt.plot(val_loss, label='Validation Loss', linewidth=2, marker='s', markersize=3)
    plt.legend(loc='upper right', fontsize=10)
    plt.ylabel('Loss', fontsize=11)
    plt.xlabel('Epoch', fontsize=11)
    plt.title('Training and Validation Loss', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    
    curves_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(curves_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved as {curves_path}")
    plt.close()

# -------------------------------------------------------------------
# 13. SAVE MODEL AND METADATA
# -------------------------------------------------------------------
def save_model(model, output_dir, class_names):
    """
    Save the trained model and metadata
    
    Args:
        model: Trained Keras model
        output_dir: Directory to save model
        class_names: List of class names
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, 'mobilenetv2_model.keras')
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Save metadata
    metadata = {
        'model_type': 'MobileNetV2 with Transfer Learning',
        'image_size': IMAGE_SIZE,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'classes': class_names,
        'num_classes': len(class_names),
        'timestamp': datetime.now().isoformat(),
        'input_shape': (IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        'output_shape': len(class_names),
        'total_parameters': model.count_params(),
        'trainable_parameters': sum([tf.size(w).numpy() for w in model.trainable_weights]),
        'paper_accuracy': '86.99%',
        'paper_f1_score': '0.79'
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
    
    print("=" * 70)
    print("MobileNetV2 for Crack Detection with Transfer Learning")
    print("From: Image-based Non-invasive Bridge Crack Detection")
    print("      Model for Edge Devices (RAICS 2025)")
    print("=" * 70)
    
    # Load datasets
    train_dataset, validation_dataset, test_dataset, class_names = load_datasets(
        args.data_dir,
        IMAGE_SIZE,
        args.batch_size
    )
    
    # Calculate class weights
    class_weights_dict = calculate_class_weights(train_dataset)
    
    # Create model
    num_classes = len(class_names)
    model = create_mobilenetv2_model(IMAGE_SIZE, num_classes)
    compile_model(model, args.learning_rate)
    model.summary()
    
    # Create callbacks
    callbacks = create_callbacks(args.output_dir)
    
    # Train model
    history = train_model(
        model,
        train_dataset,
        validation_dataset,
        args.epochs,
        class_weights_dict,
        callbacks
    )
    
    # Load best model
    best_model_path = os.path.join(args.output_dir, 'best_model.keras')
    if os.path.exists(best_model_path):
        print(f"\nLoading best model from {best_model_path}")
        model = tf.keras.models.load_model(best_model_path)
    
    # Evaluate model
    metrics = evaluate_model(model, test_dataset, class_names, args.output_dir)
    
    # Visualize results
    plot_training_curves(history, args.output_dir)
    
    # Save model
    save_model(model, args.output_dir, class_names)
    
    print("\n" + "=" * 70)
    print("Training and evaluation complete!")
    print(f"Results saved to {args.output_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()
