"""
Evaluates a trained checkpoint on the test split and reports every
metric the project brief asks for:
  - top-1 and top-5 accuracy
  - confusion matrix (saved as an image)
  - precision / recall / F1 per class
  - average inference time per clip
  - most-confused class PAIRS (not just the raw matrix — see
    analyze_confused_pairs), directly answering the brief's "confusion
    matrix analysis to identify class-wise performance and common
    misclassifications"

Usage:
    python src/evaluate.py --checkpoint checkpoints/best_model.pt
"""

import argparse
import json
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (  # noqa: E402
    CACHE_FRAMES_PER_CLIP, DATA_DIR, FRAME_CACHE_DIR, FRAME_SIZE,
    NUM_FRAMES, OUTPUT_DIR, USE_FRAME_CACHE, get_classes,
)
from src.dataset import VideoClipDataset  # noqa: E402
from src.model import build_model  # noqa: E402
from src.utils import get_device  # noqa: E402


def top_k_correct(outputs: torch.Tensor, labels: torch.Tensor, k: int) -> int:
    topk = outputs.topk(k, dim=1).indices
    return (topk == labels.unsqueeze(1)).any(dim=1).sum().item()


def analyze_confused_pairs(cm: np.ndarray, classes: list, top_n: int = 15) -> list:
    """
    Extracts the most-confused (true, predicted) class pairs from the
    confusion matrix — the actual analysis the brief asks for, rather
    than just handing over the raw matrix image and leaving the reader
    to spot patterns themselves.
    """
    pairs = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i, j] > 0:
                pairs.append({
                    "true_class": classes[i],
                    "predicted_as": classes[j],
                    "count": int(cm[i, j]),
                    "pct_of_true_class": round(100 * cm[i, j] / max(cm[i].sum(), 1), 1),
                })
    pairs.sort(key=lambda p: p["count"], reverse=True)
    return pairs[:top_n]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.path.join("checkpoints", "best_model.pt"))
    args = parser.parse_args()

    device = get_device()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=device)
    classes = ckpt.get("classes", get_classes())

    model = build_model(num_classes=len(classes), freeze_backbone=False, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    test_ds = VideoClipDataset(
        os.path.join(DATA_DIR, "test"), classes, NUM_FRAMES, FRAME_SIZE, train=False,
        use_cache=USE_FRAME_CACHE, cache_dir=FRAME_CACHE_DIR,
        cache_frames_per_clip=CACHE_FRAMES_PER_CLIP, data_root=DATA_DIR,
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    all_preds, all_labels = [], []
    top1_correct, top5_correct, total = 0, 0, 0
    inference_times = []

    with torch.no_grad():
        for clip, label in tqdm(test_loader, desc="evaluating"):
            clip, label = clip.to(device), label.to(device)

            start = time.time()
            outputs = model(clip)
            if device.type == "cuda":
                torch.cuda.synchronize()
            inference_times.append(time.time() - start)

            top1_correct += top_k_correct(outputs, label, k=1)
            top5_correct += top_k_correct(outputs, label, k=min(5, len(classes)))
            total += clip.size(0)

            all_preds.append(outputs.argmax(dim=1).item())
            all_labels.append(label.item())

    top1_acc = top1_correct / total
    top5_acc = top5_correct / total
    avg_inference_ms = np.mean(inference_times) * 1000

    print(f"\nTop-1 accuracy: {top1_acc:.4f}")
    print(f"Top-5 accuracy: {top5_acc:.4f}")
    print(f"Avg inference time per clip: {avg_inference_ms:.2f} ms")

    print("\nClassification report:")
    report = classification_report(all_labels, all_preds, target_names=classes, digits=3)
    print(report)

    cm = confusion_matrix(all_labels, all_preds)
    confused_pairs = analyze_confused_pairs(cm, classes)

    print("\nMost-confused class pairs (true → predicted):")
    for p in confused_pairs:
        print(f"  {p['true_class']} → {p['predicted_as']}: "
              f"{p['count']} clips ({p['pct_of_true_class']}% of that class's test clips)")

    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
        f.write(f"Top-1 accuracy: {top1_acc:.4f}\n")
        f.write(f"Top-5 accuracy: {top5_acc:.4f}\n")
        f.write(f"Avg inference time per clip: {avg_inference_ms:.2f} ms\n\n")
        f.write(report)
        f.write("\n\nMost-confused class pairs (true -> predicted):\n")
        for p in confused_pairs:
            f.write(f"  {p['true_class']} -> {p['predicted_as']}: "
                    f"{p['count']} clips ({p['pct_of_true_class']}% of that class's test clips)\n")

    with open(os.path.join(OUTPUT_DIR, "confused_pairs.json"), "w") as f:
        json.dump(confused_pairs, f, indent=2)

    # Raw confusion matrix + per-class metrics, saved as data (not just a
    # static image) — lets the frontend render its own interactive heatmap
    # instead of a fixed-size PNG that becomes unreadable at 101 classes.
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(len(classes))), zero_division=0
    )
    per_class_metrics = [
        {"class": classes[i], "precision": float(precision[i]), "recall": float(recall[i]),
         "f1": float(f1[i]), "support": int(support[i])}
        for i in range(len(classes))
    ]
    with open(os.path.join(OUTPUT_DIR, "confusion_matrix_data.json"), "w") as f:
        json.dump({"classes": classes, "matrix": cm.tolist(), "per_class_metrics": per_class_metrics}, f)

    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes, yticklabels=classes, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    print(f"\nConfusion matrix saved to {cm_path}")
    print(f"Raw confusion matrix + per-class metrics saved to "
          f"{os.path.join(OUTPUT_DIR, 'confusion_matrix_data.json')}")
    print(f"Full report saved to {os.path.join(OUTPUT_DIR, 'classification_report.txt')}")
    print(f"Confused-pairs data saved to {os.path.join(OUTPUT_DIR, 'confused_pairs.json')}")


if __name__ == "__main__":
    main()

