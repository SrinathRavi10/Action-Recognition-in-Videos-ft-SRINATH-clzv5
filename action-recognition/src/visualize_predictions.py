"""
Grabs a few random test clips, runs the model, and saves a grid image
showing a middle frame of each clip with its true vs predicted label.
This satisfies the brief's "visualization of model predictions" deliverable.

Usage:
    python src/visualize_predictions.py --checkpoint checkpoints/best_model.pt --num_samples 8
"""

import argparse
import os
import random
import sys

import matplotlib.pyplot as plt
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (  # noqa: E402
    CACHE_FRAMES_PER_CLIP, DATA_DIR, FRAME_CACHE_DIR, FRAME_SIZE,
    NUM_FRAMES, OUTPUT_DIR, SEED, USE_FRAME_CACHE, get_classes,
)
from src.dataset import VideoClipDataset  # noqa: E402
from src.model import build_model  # noqa: E402
from src.utils import get_device  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.path.join("checkpoints", "best_model.pt"))
    parser.add_argument("--num_samples", type=int, default=8)
    args = parser.parse_args()

    random.seed(SEED)
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
    indices = random.sample(range(len(test_ds)), min(args.num_samples, len(test_ds)))

    cols = 4
    rows = (len(indices) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten() if rows > 1 else axes

    with torch.no_grad():
        for ax, idx in zip(axes, indices):
            clip, label = test_ds[idx]
            pred = model(clip.unsqueeze(0).to(device)).argmax(dim=1).item()

            # de-normalize the middle frame for display
            mid = clip[:, clip.shape[1] // 2, :, :].permute(1, 2, 0).numpy()
            mid = (mid * [0.22803, 0.22145, 0.216989]) + [0.43216, 0.394666, 0.37645]
            mid = mid.clip(0, 1)

            ax.imshow(mid)
            ax.axis("off")
            color = "green" if pred == label else "red"
            ax.set_title(f"true: {classes[label]}\npred: {classes[pred]}", color=color, fontsize=10)

    for ax in axes[len(indices):]:
        ax.axis("off")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "sample_predictions.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved prediction visualization to {out_path}")


if __name__ == "__main__":
    main()
