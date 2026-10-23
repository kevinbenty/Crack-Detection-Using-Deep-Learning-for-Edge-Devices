"""Train and evaluate the MobileNetV2-based model (paper Section II-2, Table II).

Architecture: MobileNetV2 feature extractor (inverted residual bottleneck
blocks, expansion 6, ending in a 7x7x1280 feature map), global average
pooling, Dense(512, ReLU) and Dense(4, softmax). Input images are 224x224.
The backbone starts from ImageNet weights and is frozen by default.

    python train_mobilenetv2.py --data-dir data
"""
import argparse

import tensorflow as tf

import keras_common as kc
from metrics import plot_history, save_evaluation, save_metadata

IMAGE_SIZE = (224, 224)


def build_mobilenetv2(num_classes, image_size=IMAGE_SIZE, pretrained=True, train_backbone=False):
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(*image_size, 3),
        include_top=False,
        weights="imagenet" if pretrained else None,
    )
    backbone.trainable = train_backbone

    inputs = tf.keras.Input(shape=(*image_size, 3))
    # Same scaling as mobilenet_v2.preprocess_input: [0, 255] -> [-1, 1]
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0)(inputs)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="mobilenetv2_crack")


def main():
    parser = kc.add_common_args(
        argparse.ArgumentParser(description=__doc__), output_dir="results/mobilenetv2"
    )
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Train the backbone from scratch instead of ImageNet weights")
    parser.add_argument("--train-backbone", action="store_true",
                        help="Fine-tune the MobileNetV2 backbone instead of freezing it")
    args = parser.parse_args()

    train, val, test, class_names = kc.load_datasets(
        args.data_dir, IMAGE_SIZE, args.batch_size
    )
    model = build_mobilenetv2(
        len(class_names),
        pretrained=not args.no_pretrained,
        train_backbone=args.train_backbone or args.no_pretrained,
    )
    model.summary()

    class_weight = kc.compute_class_weights(train, len(class_names)) if args.class_weights else None
    history = kc.compile_and_fit(
        model, train, val, args.output_dir, args.epochs, args.learning_rate, class_weight
    )
    plot_history(history, args.output_dir, "MobileNetV2")

    for split, ds in (("val", val), ("test", test)):
        y_true, y_pred = kc.predict_labels(model, ds)
        save_evaluation(y_true, y_pred, class_names, args.output_dir, "MobileNetV2", split)

    model.save(f"{args.output_dir}/mobilenetv2_model.keras")
    save_metadata(
        args.output_dir, model="MobileNetV2", image_size=IMAGE_SIZE,
        classes=class_names, parameters=model.count_params(), **vars(args),
    )
    print("\nNext: python export_tflite.py --model "
          f"{args.output_dir}/mobilenetv2_model.keras --data-dir {args.data_dir}")


if __name__ == "__main__":
    main()
