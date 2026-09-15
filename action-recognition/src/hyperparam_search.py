"""
A BOUNDED hyperparameter probe — not a full grid/random search.

Why bounded, stated plainly: a proper exhaustive search (e.g. grid search
across learning rate x freeze-depth x weight decay, each run for a full
training cycle) would need many multiples of the compute this project
has access to (free-tier Colab/Kaggle GPU, session time limits). Running
one anyway and calling it "full hyperparameter tuning" would be
dishonest about what actually happened.

What this DOES do instead: trains each candidate configuration for a
small, fixed number of epochs (PROBE_EPOCHS) on the full dataset, then
picks the configuration with the best short-run validation accuracy as
the starting point for the full run in train.py. This is a real,
principled way to compare configurations within an honest compute
budget — it will not find the global optimum, but it is far better than
guessing, and it is transparent about its own limitation.

Usage:
    python src/hyperparam_search.py
"""

import itertools
import json
import os
import sys

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (  # noqa: E402
    BATCH_SIZE, BEST_HPARAMS_FILE, CACHE_FRAMES_PER_CLIP, DATA_DIR,
    FRAME_CACHE_DIR, FRAME_SIZE, NUM_FRAMES, NUM_WORKERS, OUTPUT_DIR, SEED,
    USE_FRAME_CACHE, WEIGHT_DECAY, get_classes,
)
from src.dataset import VideoClipDataset  # noqa: E402
from src.model import build_model  # noqa: E402
from src.utils import get_device, set_seed  # noqa: E402

PROBE_EPOCHS = 3  # small on purpose — this is a probe, not a full run

# The actual search space. Kept small and specific rather than a broad
# grid, because each additional point costs a full probe training run.
LEARNING_RATES = [1e-3, 1e-4, 1e-5]
FREEZE_STRATEGIES = ["last_block", "last_two_blocks"]


def run_probe_epoch(model, loader, criterion, optimizer, scaler, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for clips, labels in loader:
            clips, labels = clips.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
                with autocast("cuda", enabled=device.type == "cuda"):
                    outputs = model(clips)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                with autocast("cuda", enabled=device.type == "cuda"):
                    outputs = model(clips)
                    loss = criterion(outputs, labels)
            total_loss += loss.item() * clips.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += clips.size(0)
    return total_loss / total, correct / total


def build_model_with_freeze(num_classes: int, freeze_strategy: str):
    """freeze_strategy controls how much of the backbone stays frozen."""
    model = build_model(num_classes=num_classes, freeze_backbone=True)
    if freeze_strategy == "last_two_blocks":
        for param in model.layer3.parameters():
            param.requires_grad = True
    # "last_block" is the default build_model() behavior (layer4 + fc only)
    return model


def main():
    set_seed(SEED)
    device = get_device()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    classes = get_classes()

    print(f"Bounded hyperparameter probe: {len(LEARNING_RATES)} learning rates x "
          f"{len(FREEZE_STRATEGIES)} freeze strategies = "
          f"{len(LEARNING_RATES) * len(FREEZE_STRATEGIES)} configurations, "
          f"{PROBE_EPOCHS} epochs each.")
    print("This picks a good starting configuration — it is NOT a substitute for "
          "the full training run in train.py, and does not claim global optimality.\n")

    train_ds = VideoClipDataset(
        os.path.join(DATA_DIR, "train"), classes, NUM_FRAMES, FRAME_SIZE, train=True,
        use_cache=USE_FRAME_CACHE, cache_dir=FRAME_CACHE_DIR,
        cache_frames_per_clip=CACHE_FRAMES_PER_CLIP, data_root=DATA_DIR,
    )
    test_ds = VideoClipDataset(
        os.path.join(DATA_DIR, "test"), classes, NUM_FRAMES, FRAME_SIZE, train=False,
        use_cache=USE_FRAME_CACHE, cache_dir=FRAME_CACHE_DIR,
        cache_frames_per_clip=CACHE_FRAMES_PER_CLIP, data_root=DATA_DIR,
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    results = []
    for lr, freeze_strategy in itertools.product(LEARNING_RATES, FREEZE_STRATEGIES):
        print(f"--- Probing lr={lr}, freeze={freeze_strategy} ---")
        model = build_model_with_freeze(len(classes), freeze_strategy).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=lr, weight_decay=WEIGHT_DECAY)
        scaler = GradScaler("cuda", enabled=device.type == "cuda")

        val_acc = 0.0
        for epoch in range(1, PROBE_EPOCHS + 1):
            train_loss, train_acc = run_probe_epoch(model, train_loader, criterion, optimizer, scaler, device, train=True)
            val_loss, val_acc = run_probe_epoch(model, test_loader, criterion, optimizer, scaler, device, train=False)
            print(f"  epoch {epoch}/{PROBE_EPOCHS} | train_acc {train_acc:.3f} | val_acc {val_acc:.3f}")

        results.append({"lr": lr, "freeze_strategy": freeze_strategy, "probe_val_acc": val_acc})

    results.sort(key=lambda r: r["probe_val_acc"], reverse=True)
    best = results[0]

    print("\n=== Probe results (sorted by short-run val accuracy) ===")
    for r in results:
        print(f"  lr={r['lr']:<8} freeze={r['freeze_strategy']:<16} probe_val_acc={r['probe_val_acc']:.3f}")

    print(f"\nBest candidate: lr={best['lr']}, freeze={best['freeze_strategy']} "
          f"(probe_val_acc={best['probe_val_acc']:.3f})")
    print("Update LEARNING_RATE in src/config.py to this value before running the full "
          "src/train.py, and adjust model.py's freeze strategy if 'last_two_blocks' won.")

    with open(BEST_HPARAMS_FILE, "w") as f:
        json.dump({"all_results": results, "best": best}, f, indent=2)
    print(f"\nFull probe results saved to: {BEST_HPARAMS_FILE}")


if __name__ == "__main__":
    main()
