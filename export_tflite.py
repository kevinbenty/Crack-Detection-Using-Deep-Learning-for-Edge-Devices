"""Convert a trained Keras model to an INT8-quantized TensorFlow Lite model
for the Raspberry Pi (paper Section IV: MobileNetV2, TFLite, INT8).

Full-integer quantization needs a representative dataset; a few hundred
training images are used to calibrate activation ranges. Input and output
stay float32, so the Pi script can feed raw [0, 255] pixels.

    python export_tflite.py --model results/mobilenetv2/mobilenetv2_model.keras --data-dir data
"""
import argparse
import json
import os

import tensorflow as tf


def representative_dataset(data_dir, image_size, num_images):
    ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_dir, "train"),
        image_size=image_size, batch_size=1, shuffle=True, seed=123,
    )
    for image, _ in ds.take(num_images):
        yield [tf.cast(image, tf.float32)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to a trained .keras model")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default=None,
                        help="Output .tflite path (default: next to the model)")
    parser.add_argument("--num-calibration-images", type=int, default=300)
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    image_size = tuple(model.input_shape[1:3])
    class_names = sorted(
        d for d in os.listdir(os.path.join(args.data_dir, "train"))
        if os.path.isdir(os.path.join(args.data_dir, "train", d))
    )

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(
        args.data_dir, image_size, args.num_calibration_images
    )
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    tflite_model = converter.convert()

    output = args.output or os.path.splitext(args.model)[0] + "_int8.tflite"
    with open(output, "wb") as f:
        f.write(tflite_model)
    labels_path = os.path.join(os.path.dirname(output), "labels.json")
    with open(labels_path, "w") as f:
        json.dump(class_names, f, indent=2)

    print(f"Saved {output} ({len(tflite_model) / 1e6:.1f} MB) and {labels_path}")


if __name__ == "__main__":
    main()
