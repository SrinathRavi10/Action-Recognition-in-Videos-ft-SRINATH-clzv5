"""
Downloads UCF101 (via the Hugging Face mirror) and keeps only the class
subset defined in src/config.py, then creates a train/test split.

Usage:
    python data/download_ucf101.py

Result:
    data/UCF101_subset/train/<ClassName>/*.avi
    data/UCF101_subset/test/<ClassName>/*.avi
"""

import json
import os
import random
import re
import shutil
import sys
import zipfile

from huggingface_hub import snapshot_download

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (  # noqa: E402
    CLASS_DIST_FILE, CLASSES, CLASSES_FILE, DATA_DIR, TRAIN_SPLIT, SEED, USE_ALL_CLASSES,
)

HF_REPO = "quchenyuan/UCF101-ZIP"
RAW_DIR = os.path.join(os.path.dirname(__file__), "_raw_ucf101")


def download_full_dataset() -> str:
    """Downloads the full UCF101 zip mirror from Hugging Face (one-time, ~7GB)."""
    print(f"Downloading {HF_REPO} from Hugging Face (this can take a while)...")
    local_dir = snapshot_download(repo_id=HF_REPO, repo_type="dataset", local_dir=RAW_DIR)
    return local_dir


def extract_if_needed(local_dir: str) -> str:
    """Extracts the dataset zip(s) if not already extracted, then deletes
    the zip files — once extracted, they're just dead weight taking up
    disk space we don't have room for at full 101-class scale."""
    extracted_marker = os.path.join(local_dir, "_extracted")
    if os.path.exists(extracted_marker):
        return local_dir

    for name in os.listdir(local_dir):
        if name.lower().endswith(".zip"):
            zip_path = os.path.join(local_dir, name)
            print(f"Extracting {name} ...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(local_dir)
            os.remove(zip_path)  # reclaim disk space immediately
            print(f"  Deleted {name} after extraction to free disk space.")

    open(extracted_marker, "w").close()
    return local_dir


def find_class_folder(root: str, class_name: str):
    """UCF101 zip mirrors sometimes nest videos under an extra folder level."""
    for dirpath, dirnames, _ in os.walk(root):
        if class_name in dirnames:
            return os.path.join(dirpath, class_name)
    return None


def discover_all_classes(root: str) -> list:
    """
    Finds the directory that actually holds the 101 class folders (mirrors
    nest it at varying depths) by picking the directory with the most
    subdirectories, then returns those subdirectory names sorted.
    """
    best_dir, best_count = None, 0
    for dirpath, dirnames, _ in os.walk(root):
        if len(dirnames) > best_count:
            best_dir, best_count = dirpath, len(dirnames)

    if best_dir is None or best_count < 2:
        raise RuntimeError("Could not locate UCF101 class folders under the downloaded dataset.")

    print(f"Discovered {best_count} class folders under: {best_dir}")
    return sorted(os.listdir(best_dir))


GROUP_PATTERN = re.compile(r"_g(\d+)_c\d+", re.IGNORECASE)


def group_id(filename: str) -> str:
    """
    UCF101 filenames look like v_JugglingBalls_g01_c01.avi — clips sharing
    the same g## come from the same source recording (same actor/background).
    We must keep every clip from a group together in one split, or the
    model can "recognize the scene" instead of the action, inflating
    accuracy through leakage.
    """
    match = GROUP_PATTERN.search(filename)
    return match.group(1) if match else filename  # fallback: treat as its own group


def build_subset(root: str, classes: list) -> None:
    random.seed(SEED)
    class_distribution = {}  # for the frontend's imbalance chart + train.py's class weights

    for class_name in classes:
        src_folder = find_class_folder(root, class_name)
        if src_folder is None:
            print(f"[WARN] Could not find class folder for '{class_name}' — skipping.")
            class_distribution[class_name] = {"train": 0, "test": 0}
            continue

        videos = [f for f in os.listdir(src_folder) if f.lower().endswith((".avi", ".mp4"))]

        # Bucket clips by source group, then split at the GROUP level so no
        # group's clips end up split across train and test.
        groups = {}
        for v in videos:
            groups.setdefault(group_id(v), []).append(v)

        group_ids = list(groups.keys())
        random.shuffle(group_ids)
        split_idx = max(1, int(len(group_ids) * TRAIN_SPLIT))
        train_group_ids, test_group_ids = group_ids[:split_idx], group_ids[split_idx:]

        train_videos = [v for g in train_group_ids for v in groups[g]]
        test_videos = [v for g in test_group_ids for v in groups[g]]

        for split_name, split_videos in [("train", train_videos), ("test", test_videos)]:
            dst_folder = os.path.join(DATA_DIR, split_name, class_name)
            os.makedirs(dst_folder, exist_ok=True)
            for v in split_videos:
                src_path = os.path.join(src_folder, v)
                dst_path = os.path.join(dst_folder, v)
                if os.path.lexists(dst_path):
                    os.remove(dst_path)
                # Symlink instead of copy: at full 101-class scale, copying
                # duplicates nearly the entire dataset's disk footprint a
                # second time, which is what caused "No space left on
                # device" on Kaggle/Colab. A symlink costs ~0 bytes and
                # cv2.VideoCapture reads through it transparently.
                try:
                    os.symlink(os.path.abspath(src_path), dst_path)
                except OSError:
                    # Some filesystems (rare, e.g. certain network mounts)
                    # don't support symlinks — fall back to a real copy.
                    shutil.copy(src_path, dst_path)

        class_distribution[class_name] = {"train": len(train_videos), "test": len(test_videos)}
        print(f"{class_name}: {len(train_videos)} train / {len(test_videos)} test clips")

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(CLASSES_FILE, "w") as f:
        f.write("\n".join(classes) + "\n")

    with open(CLASS_DIST_FILE, "w") as f:
        json.dump(class_distribution, f, indent=2)

    # Flag imbalance explicitly rather than silently accepting it — this is
    # exactly why train.py applies class-weighted loss (see src/config.py).
    train_counts = [v["train"] for v in class_distribution.values() if v["train"] > 0]
    if train_counts:
        imbalance_ratio = max(train_counts) / min(train_counts)
        print(f"\nClass imbalance ratio (largest/smallest train class): {imbalance_ratio:.2f}x")
        if imbalance_ratio > 2.0:
            print("  -> Meaningful imbalance detected. Class-weighted loss will be used in training "
                  "(USE_CLASS_WEIGHTS in src/config.py) to prevent common classes from dominating.")

    print(f"\nDone. Subset written to: {DATA_DIR}")
    print(f"Class list written to: {CLASSES_FILE}")
    print(f"Class distribution written to: {CLASS_DIST_FILE}")


if __name__ == "__main__":
    local_dir = download_full_dataset()
    extracted_root = extract_if_needed(local_dir)

    if USE_ALL_CLASSES:
        classes_to_use = discover_all_classes(extracted_root)
        print(f"USE_ALL_CLASSES=True — training on all {len(classes_to_use)} discovered classes.")
    else:
        classes_to_use = CLASSES
        print(f"Using fixed subset of {len(classes_to_use)} classes from src/config.py")

    build_subset(extracted_root, classes_to_use)
