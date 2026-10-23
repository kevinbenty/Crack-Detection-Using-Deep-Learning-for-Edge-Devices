"""Data pipeline and training setup shared by the Keras models (CNN, MobileNetV2).

Implements the experimental framework of Section III of the paper:
  * separate train / val / test directories with no overlap
  * augmentation: rotation, flipping, zooming, brightness adjustment
  * Adam optimizer, initial learning rate 0.001, batch size 32
  * learning rate reduced by a validation-based scheduler
  * up to 30 epochs, early stopping with patience 5
  * checkpoint keeps the model with the best validation accuracy
"""
import os

import numpy as np
import tensorflow as tf

BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 5
SEED = 123


def build_augmentation():
    """Augmentations listed in Section III-A. Applied to training data only."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.125, seed=SEED),  # up to +/-45 degrees
            tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=SEED),
            tf.keras.layers.RandomZoom(0.2, seed=SEED),  # 0.8x - 1.2x
            tf.keras.layers.RandomBrightness(0.2, value_range=(0, 255), seed=SEED),
        ],
        name="augmentation",
    )


def _load_split(data_dir, split, image_size, batch_size, shuffle):
    path = os.path.join(data_dir, split)
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"Missing '{path}'. Expected {data_dir}/train, {data_dir}/val and "
            f"{data_dir}/test, each with one folder per class."
        )
    return tf.keras.utils.image_dataset_from_directory(
        path,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=SEED,
        label_mode="int",
    )


def load_datasets(data_dir, image_size, batch_size=BATCH_SIZE, augment=True):
    """Return (train, val, test, class_names). Images stay in [0, 255];
    each model does its own rescaling so the saved model works on raw images."""
    train = _load_split(data_dir, "train", image_size, batch_size, shuffle=True)
    val = _load_split(data_dir, "val", image_size, batch_size, shuffle=False)
    test = _load_split(data_dir, "test", image_size, batch_size, shuffle=False)

    class_names = train.class_names
    for name, ds in (("val", val), ("test", test)):
        if ds.class_names != class_names:
            raise ValueError(
                f"Class folders in '{name}' {ds.class_names} differ from 'train' {class_names}"
            )
    print(f"Classes: {class_names}")

    autotune = tf.data.AUTOTUNE
    if augment:
        aug = build_augmentation()
        train = train.map(
            lambda x, y: (aug(x, training=True), y), num_parallel_calls=autotune
        )
    return train.prefetch(autotune), val.prefetch(autotune), test.prefetch(autotune), class_names


def compute_class_weights(train_ds, num_classes):
    """Optional 'balanced' class weights (n_samples / (n_classes * n_class_c))."""
    counts = np.zeros(num_classes, dtype=np.int64)
    for _, labels in train_ds:
        counts += np.bincount(labels.numpy(), minlength=num_classes)
    total = counts.sum()
    weights = {i: float(total / (num_classes * c)) for i, c in enumerate(counts) if c > 0}
    print(f"Class counts: {counts.tolist()}  ->  weights: {weights}")
    return weights


def build_callbacks(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, "best_model.keras"),
            monitor="val_accuracy", mode="max", save_best_only=True, verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", mode="max", patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=3, min_lr=1e-7, verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(os.path.join(output_dir, "history.csv")),
    ]


def compile_and_fit(model, train, val, output_dir, epochs, learning_rate, class_weight=None):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        train,
        validation_data=val,
        epochs=epochs,
        callbacks=build_callbacks(output_dir),
        class_weight=class_weight,
    )
    return history.history


def predict_labels(model, ds):
    """True and predicted labels for an unshuffled dataset."""
    y_true = np.concatenate([y.numpy() for _, y in ds])
    y_pred = model.predict(ds, verbose=1).argmax(axis=1)
    return y_true, y_pred


def add_common_args(parser, output_dir):
    parser.add_argument("--data-dir", default="data",
                        help="Folder with train/, val/ and test/ subfolders")
    parser.add_argument("--output-dir", default=output_dir)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--class-weights", action="store_true",
                        help="Also weight the loss by inverse class frequency")
    return parser
