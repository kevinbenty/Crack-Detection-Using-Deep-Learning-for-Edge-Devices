"""Train and evaluate the 3-layer CNN (paper Section II-1, Table I).

Architecture: three 3x3 Conv2D layers (32, 64, 128 filters, ReLU), each
followed by 2x2 max pooling, then Flatten -> Dense(512, ReLU) ->
Dense(4, softmax). Input images are 256x256.

    python train_cnn.py --data-dir data
"""
import argparse

import tensorflow as tf

import keras_common as kc
from metrics import plot_history, save_evaluation, save_metadata

IMAGE_SIZE = (256, 256)


def build_cnn(num_classes, image_size=IMAGE_SIZE):
    inputs = tf.keras.Input(shape=(*image_size, 3))
    x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)
    for filters in (32, 64, 128):
        x = tf.keras.layers.Conv2D(filters, 3, activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="cnn_3layer")


def main():
    parser = kc.add_common_args(
        argparse.ArgumentParser(description=__doc__), output_dir="results/cnn"
    )
    args = parser.parse_args()

    train, val, test, class_names = kc.load_datasets(
        args.data_dir, IMAGE_SIZE, args.batch_size
    )
    model = build_cnn(len(class_names))
    model.summary()

    class_weight = kc.compute_class_weights(train, len(class_names)) if args.class_weights else None
    history = kc.compile_and_fit(
        model, train, val, args.output_dir, args.epochs, args.learning_rate, class_weight
    )
    plot_history(history, args.output_dir, "3-layer CNN")

    # EarlyStopping restores the best-validation-accuracy weights.
    for split, ds in (("val", val), ("test", test)):
        y_true, y_pred = kc.predict_labels(model, ds)
        save_evaluation(y_true, y_pred, class_names, args.output_dir, "3-layer CNN", split)

    model.save(f"{args.output_dir}/cnn_model.keras")
    save_metadata(
        args.output_dir, model="3-layer CNN", image_size=IMAGE_SIZE,
        classes=class_names, parameters=model.count_params(), **vars(args),
    )


if __name__ == "__main__":
    main()
