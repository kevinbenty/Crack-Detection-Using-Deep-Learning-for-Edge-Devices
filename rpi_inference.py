"""Run the INT8 TFLite crack classifier on a Raspberry Pi.

Classify image files:
    python3 rpi_inference.py --model mobilenetv2_model_int8.tflite --labels labels.json img1.jpg img2.jpg

Classify a live camera feed (needs OpenCV), printing the class and FPS:
    python3 rpi_inference.py --model mobilenetv2_model_int8.tflite --labels labels.json --camera 0
"""
import argparse
import json
import time

import numpy as np
from PIL import Image

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:  # full TensorFlow installed instead of tflite-runtime
    from tensorflow.lite.python.interpreter import Interpreter


class CrackClassifier:
    def __init__(self, model_path, labels_path, num_threads=4):
        self.interpreter = Interpreter(model_path=model_path, num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.input = self.interpreter.get_input_details()[0]
        self.output = self.interpreter.get_output_details()[0]
        self.height, self.width = self.input["shape"][1:3]
        with open(labels_path) as f:
            self.labels = json.load(f)

    def _prepare(self, image):
        image = image.convert("RGB").resize((self.width, self.height))
        x = np.asarray(image, dtype=np.float32)[None, ...]
        if self.input["dtype"] != np.float32:  # model exported with integer input
            scale, zero_point = self.input["quantization"]
            x = np.clip(np.round(x / scale + zero_point), 0, 255).astype(self.input["dtype"])
        return x

    def predict(self, image):
        self.interpreter.set_tensor(self.input["index"], self._prepare(image))
        self.interpreter.invoke()
        probs = self.interpreter.get_tensor(self.output["index"])[0].astype(np.float32)
        if self.output["dtype"] != np.float32:
            scale, zero_point = self.output["quantization"]
            probs = (probs - zero_point) * scale
        idx = int(np.argmax(probs))
        return self.labels[idx], float(probs[idx])


def run_camera(classifier, camera_index):
    import cv2

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            start = time.perf_counter()
            label, conf = classifier.predict(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            fps = 1.0 / (time.perf_counter() - start)
            print(f"{label:<14} {conf:6.1%}   {fps:5.1f} FPS", end="\r", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        print()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--camera", type=int, default=None, help="Camera index for live mode")
    parser.add_argument("images", nargs="*")
    args = parser.parse_args()

    classifier = CrackClassifier(args.model, args.labels, args.threads)
    if args.camera is not None:
        run_camera(classifier, args.camera)
        return
    if not args.images:
        parser.error("give image paths or --camera")

    for path in args.images:
        start = time.perf_counter()
        label, conf = classifier.predict(Image.open(path))
        ms = (time.perf_counter() - start) * 1000
        print(f"{path}: {label} ({conf:.1%}) in {ms:.0f} ms")


if __name__ == "__main__":
    main()
