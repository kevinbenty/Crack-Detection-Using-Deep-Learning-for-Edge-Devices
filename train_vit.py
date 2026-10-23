"""Train and evaluate the Vision Transformer (paper Section II-3, Table III).

ViT-Base/16: 224x224 input, 16x16 patches, 12 encoder blocks, 12 attention
heads, embedding dimension 768, MLP size 3072, [CLS] token -> MLP head.
Starts from google/vit-base-patch16-224-in21k.

Training follows the same framework as the other models (Section III-B):
Adam, initial learning rate 0.001, batch size 32, learning rate reduced on
validation plateau, up to 30 epochs, early stopping with patience 5, best
checkpoint chosen by validation accuracy.

    python train_vit.py --data-dir data
"""
import argparse
import os

import numpy as np
import torch
from torchvision import datasets, transforms
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    ViTForImageClassification,
    ViTImageProcessor,
)

from metrics import plot_history, save_evaluation, save_metadata

MODEL_NAME = "google/vit-base-patch16-224-in21k"
IMAGE_SIZE = 224


class CrackFolder(torch.utils.data.Dataset):
    """ImageFolder wrapper returning the dict format Trainer expects."""

    def __init__(self, root, transform):
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Missing '{root}' (one folder per class expected)")
        self.folder = datasets.ImageFolder(root)
        self.transform = transform

    @property
    def classes(self):
        return self.folder.classes

    @property
    def labels(self):
        return [label for _, label in self.folder.samples]

    def __len__(self):
        return len(self.folder)

    def __getitem__(self, idx):
        image, label = self.folder[idx]
        return {"pixel_values": self.transform(image.convert("RGB")), "labels": label}


def build_transforms(processor):
    normalize = transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
    train_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        # Augmentations from Section III-A: rotation, flipping, zooming, brightness
        transforms.RandomRotation(45),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=0, scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, eval_tf


def compute_accuracy(eval_pred):
    logits, labels = eval_pred
    return {"accuracy": float((np.argmax(logits, axis=1) == labels).mean())}


def history_from_logs(log_history):
    """Per-epoch train/val accuracy and loss from Trainer logs (for Fig. 4)."""
    history = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}
    train_loss_by_epoch = {}
    for entry in log_history:
        if "loss" in entry and "epoch" in entry:
            train_loss_by_epoch[round(entry["epoch"])] = entry["loss"]
        if "eval_accuracy" in entry:
            epoch = round(entry["epoch"])
            history["val_accuracy"].append(entry["eval_accuracy"])
            history["val_loss"].append(entry["eval_loss"])
            history["loss"].append(train_loss_by_epoch.get(epoch, np.nan))
            history["accuracy"].append(entry.get("train_accuracy", np.nan))
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data",
                        help="Folder with train/, val/ and test/ subfolders")
    parser.add_argument("--output-dir", default="results/vit")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Paper value. If training is unstable, try 2e-4 or lower.")
    args = parser.parse_args()

    processor = ViTImageProcessor.from_pretrained(args.model_name)
    train_tf, eval_tf = build_transforms(processor)
    train_ds = CrackFolder(os.path.join(args.data_dir, "train"), train_tf)
    val_ds = CrackFolder(os.path.join(args.data_dir, "val"), eval_tf)
    test_ds = CrackFolder(os.path.join(args.data_dir, "test"), eval_tf)

    class_names = train_ds.classes
    for name, ds in (("val", val_ds), ("test", test_ds)):
        if ds.classes != class_names:
            raise ValueError(f"Class folders in '{name}' {ds.classes} differ from 'train' {class_names}")
    print(f"Classes: {class_names}")

    model = ViTForImageClassification.from_pretrained(
        args.model_name,
        num_labels=len(class_names),
        id2label=dict(enumerate(class_names)),
        label2id={c: i for i, c in enumerate(class_names)},
    )

    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        optim="adamw_torch",
        weight_decay=0.0,                          # plain Adam
        lr_scheduler_type="reduce_lr_on_plateau",  # validation-based scheduler
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_strategy="epoch",
        remove_unused_columns=False,
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_accuracy,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )
    trainer.train()

    history = history_from_logs(trainer.state.log_history)
    if history["val_loss"]:
        plot_history(history, args.output_dir, "ViT")

    for split, ds in (("val", val_ds), ("test", test_ds)):
        pred = trainer.predict(ds)
        save_evaluation(
            pred.label_ids, pred.predictions.argmax(axis=1),
            class_names, args.output_dir, "ViT", split,
        )

    model_dir = os.path.join(args.output_dir, "vit_model")
    trainer.save_model(model_dir)
    processor.save_pretrained(model_dir)
    save_metadata(
        args.output_dir, model="ViT-Base/16", image_size=IMAGE_SIZE,
        classes=class_names, parameters=model.num_parameters(), **vars(args),
    )


if __name__ == "__main__":
    main()
