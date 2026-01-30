# train_cnn.py
"""
3-Layer CNN Model for Crack Detection
From: Image-based Non-invasive Bridge Crack Detection Model for Edge Devices (RAICS 2025)

This script implements a lightweight 3-layer CNN model for multi-class crack detection.
The model is designed for edge device compatibility with minimal parameters (~40K).

Architecture: Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Conv2D(128) → MaxPool → Dense(512) → Dense(4)

Usage:
    python train_cnn.py
    
    Or with custom parameters:
    python train_cnn.py --epochs 30 --batch-size 32
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
import json
from datetime import datetime

# -------------------------------------------------------------------
# 2. DEFINE CONFIGURATION
# -------------------------------------------------------------------
# The paper specifies using 256x256 images for the basic CNN model
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32
EPOCHS = 5  # Default: 5 (paper used 30)
data_dir = "data"

# -------------------------------------------------------------------
# 3. ARGUMENT PARSER
# -------------------------------------------------------------------
def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Train 3-Layer CNN for crack detection'
    )
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help=f'Number of training epochs (default: {EPOCHS})')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help=f'Batch size for training (default: {BATCH_SIZE})')
    parser.add_argument('--data-dir', type=str, default=data_dir,
                        help=f'Path to data directory (default: {data_dir})')
    parser.add_argument('--output-dir', type=str, default='results/cnn',
                        help='Output directory for results')
    
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
    
    # Load training dataset with 20% validation split
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

    # Create the test dataset
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "test"),
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False  # Keep test data in order for evaluation
    )

    # Configure dataset for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    return train_dataset, validation_dataset, test_dataset, class_names

# -------------------------------------------------------------------
# 5. CREATE THE 3-LAYER CNN MODEL
# -------------------------------------------------------------------
def create_cnn_model(image_size, num_classes):
    """
    Create a 3-layer CNN model for crack detection
    
    Architecture (from Table I in paper):
    - Input: Rescaling layer
    - Conv Block 1: Conv2D(32) → ReLU → MaxPool(2,2)
    - Conv Block 2: Conv2D(64) → ReLU → MaxPool(2,2)
    - Conv Block 3: Conv2D(128) → ReLU → MaxPool(2,2)
    - Dense: Flatten → Dense(512) → ReLU
    - Output: Dense(num_classes) → Softmax
    
    Args:
        image_size: Tuple of (height, width)
        num_classes: Number of output classes
    
    Returns:
        Compiled Keras model
    """
    print("Building 3-Layer CNN model...")
    
    model = tf.keras.Sequential([
        # Input layer: Rescale pixel values from [0, 255] to [0, 1]
        tf.keras.layers.Rescaling(
            1./255, 
            input_shape=(image_size[0], image_size[1], 3)
        ),

        # First Convolutional Block
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Second Convolutional Block
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Third Convolutional Block
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Flatten the results to feed into a dense layer
        tf.keras.layers.Flatten(),

        # Dense Layer
        tf.keras.layers.Dense(512, activation='relu'),

        # Output Layer
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    return model

# -------------------------------------------------------------------
# 6. COMPILE THE MODEL
# -------------------------------------------------------------------
def compile_model(model):
    """
    Compile the model with Adam optimizer and categorical cross-entropy loss
    
    Args:
        model: Keras model to compile
    """
    print("Compiling model...")
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

# -------------------------------------------------------------------
# 7. TRAIN THE MODEL
# -------------------------------------------------------------------
def train_model(model, train_dataset, validation_dataset, epochs):
    """
    Train the model on the training dataset
    
    Args:
        model: Compiled Keras model
        train_dataset: Training dataset
        validation_dataset: Validation dataset
        epochs: Number of epochs to train
    
    Returns:
        Training history
    """
    print("\nStarting training...")
    print(f"Training for {epochs} epochs with batch size {BATCH_SIZE}...\n")
    
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=validation_dataset
    )
    
    print("\nTraining complete!")
    return history

# -------------------------------------------------------------------
# 8. EVALUATE AND VISUALIZE RESULTS
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
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get predictions on the test set
    true_labels = []
    predicted_labels = []

    for images, labels in test_dataset:
        true_labels.extend(labels.numpy())
        preds = model.predict(images, verbose=0)
        predicted_labels.extend(np.argmax(preds, axis=1))

    # --- Classification Report ---
    print("\n--- Classification Report ---")
    report = classification_report(
        true_labels, 
        predicted_labels, 
        target_names=class_names
    )
    print(report)
    
    # Save classification report
    report_path = os.path.join(output_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Report saved to {report_path}")

    # --- Confusion Matrix ---
    print("\nGenerating confusion matrix...")
    cm = confusion_matrix(true_labels, predicted_labels)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    display.plot(xticks_rotation='vertical')
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
# 9. PLOT TRAINING CURVES
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

    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy', linewidth=2)
    plt.plot(val_acc, label='Validation Accuracy', linewidth=2)
    plt.legend(loc='lower right', fontsize=10)
    plt.ylabel('Accuracy', fontsize=11)
    plt.xlabel('Epoch', fontsize=11)
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss', linewidth=2)
    plt.plot(val_loss, label='Validation Loss', linewidth=2)
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
# 10. SAVE MODEL AND METADATA
# -------------------------------------------------------------------
def save_model(model, output_dir, class_names, metrics):
    """
    Save the trained model and metadata
    
    Args:
        model: Trained Keras model
        output_dir: Directory to save model
        class_names: List of class names
        metrics: Dictionary of evaluation metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, 'cnn_model.keras')
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Save metadata
    metadata = {
        'model_type': '3-Layer CNN',
        'image_size': IMAGE_SIZE,
        'batch_size': BATCH_SIZE,
        'classes': class_names,
        'num_classes': len(class_names),
        'timestamp': datetime.now().isoformat(),
        'input_shape': (IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        'output_shape': len(class_names),
        'total_parameters': model.count_params()
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
    print("3-Layer CNN for Crack Detection")
    print("From: Image-based Non-invasive Bridge Crack Detection")
    print("      Model for Edge Devices (RAICS 2025)")
    print("=" * 70)
    
    # Load datasets
    train_dataset, validation_dataset, test_dataset, class_names = load_datasets(
        args.data_dir, 
        IMAGE_SIZE, 
        args.batch_size
    )
    
    # Create model
    num_classes = len(class_names)
    model = create_cnn_model(IMAGE_SIZE, num_classes)
    compile_model(model)
    model.summary()
    
    # Train model
    history = train_model(
        model, 
        train_dataset, 
        validation_dataset, 
        args.epochs
    )
    
    # Evaluate model
    metrics = evaluate_model(model, test_dataset, class_names, args.output_dir)
    
    # Visualize results
    plot_training_curves(history, args.output_dir)
    
    # Save model
    save_model(model, args.output_dir, class_names, metrics)
    
    print("\n" + "=" * 70)
    print("Training and evaluation complete!")
    print(f"Results saved to {args.output_dir}")
    print("=" * 70)

if __name__ == "__main__":
    main()
