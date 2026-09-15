"""
Builds a frame cache: decodes CACHE_FRAMES_PER_CLIP frames per video ONCE
and saves them to disk, so training doesn't need to re-decode every video
from scratch on every epoch.

WHY THIS IS SAFE FOR ACCURACY: this caches decoded, resized frames only —
never the final augmented tensor. Random flip, random crop, and temporal
subsampling still happen fresh every time a clip is loaded during training
(see src/dataset.py's VideoClipDataset docstring for the full reasoning).
Only the expensive video-decode step is done once instead of every epoch.

DISK SPACE — read before running:
  The cache needs the RAW EXTRACTED dataset to stay on disk at the same
  time (data/UCF101_subset/{train,test} are symlinks into
  data/_raw_ucf101/ — deleting the raw folder would break them). So the
  real peak disk usage is (raw dataset, ~7GB) + (cache, ~15GB at default
  settings) = ~22GB, not just the cache size alone. This exceeded
  Kaggle's disk quota in practice.

  This script now handles running out of space GRACEFULLY instead of
  failing file-by-file for the remaining thousands of clips:
    - Checks free disk space before each write and stops cleanly with a
      clear summary once it gets close to full, rather than continuing
      to attempt (and fail) every remaining file one at a time.
    - Any clip that doesn't get cached will simply be decoded live
      during training instead — src/dataset.py already falls back to
      live decoding on a cache miss. This means an incomplete cache
      costs some training speed (only for the uncached clips), but has
      ZERO effect on accuracy or correctness.
    - Deletes any partially-written file from an interrupted save
      instead of leaving a corrupted .npy behind — a truncated cache
      file left on disk would otherwise be mistaken for a valid cached
      clip on the next run and crash (or silently return garbage) when
      loaded during training. This was an actual bug found from a real
      run's "No space left on device" mid-write failure.

  If you want a smaller cache to begin with: lower CACHE_FRAMES_PER_CLIP
  in src/config.py (linearly reduces cache size), or delete
  data/_raw_ucf101/*.zip leftovers (already deleted automatically) /
  free up other Kaggle working-directory space first.

Usage:
    python data/build_frame_cache.py
"""

import os
import shutil
import sys
import time

import numpy as np
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (  # noqa: E402
    CACHE_FRAMES_PER_CLIP, DATA_DIR, FRAME_CACHE_DIR, FRAME_SIZE, get_classes,
)
from src.dataset import build_cached_clip, cache_path_for  # noqa: E402

# Stop BEFORE hitting zero, not after — leaves headroom for the OS/other
# processes and avoids the failure-storm of trying hundreds of writes
# that are all doomed to fail once space is nearly gone.
MIN_FREE_SPACE_GB = 1.0


def estimate_size_gb(num_clips: int) -> float:
    bytes_per_clip = CACHE_FRAMES_PER_CLIP * FRAME_SIZE * FRAME_SIZE * 3
    return (num_clips * bytes_per_clip) / (1024 ** 3)


def free_space_gb(path: str) -> float:
    return shutil.disk_usage(path).free / (1024 ** 3)


def safe_remove(path: str) -> None:
    """Deletes a file if it exists, swallowing errors — used for cleaning
    up partial writes, where the delete itself could theoretically also
    fail if disk is in a bad state, but we don't want that to crash the
    cleanup path."""
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def main():
    classes = get_classes()
    data_root = DATA_DIR  # parent of train/ and test/, matches VideoClipDataset's default

    all_videos = []
    for split in ["train", "test"]:
        split_root = os.path.join(DATA_DIR, split)
        if not os.path.isdir(split_root):
            continue
        for class_name in classes:
            class_dir = os.path.join(split_root, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".avi", ".mp4")):
                    all_videos.append(os.path.join(class_dir, fname))

    if not all_videos:
        raise RuntimeError(f"No videos found under {DATA_DIR}. Run data/download_ucf101.py first.")

    estimated_gb = estimate_size_gb(len(all_videos))
    current_free_gb = free_space_gb(DATA_DIR)
    print(f"Found {len(all_videos)} clips to cache.")
    print(f"Estimated full cache size: {estimated_gb:.1f} GB "
          f"({CACHE_FRAMES_PER_CLIP} frames x {FRAME_SIZE}x{FRAME_SIZE} per clip).")
    print(f"Currently free disk space: {current_free_gb:.1f} GB")
    if current_free_gb < estimated_gb + MIN_FREE_SPACE_GB:
        print(f"[WARNING] Free space is less than the estimated full cache size — "
              f"this run will likely stop early once space runs low. That's OK: any "
              f"clip left uncached just decodes live during training instead (no "
              f"accuracy impact, just somewhat slower for those specific clips).")
    print(f"Cache directory: {FRAME_CACHE_DIR}\n")

    os.makedirs(FRAME_CACHE_DIR, exist_ok=True)

    already_cached, newly_cached, failed, stopped_early = 0, 0, 0, False
    start = time.time()

    pbar = tqdm(all_videos, desc="Building frame cache")
    for video_path in pbar:
        cpath = cache_path_for(video_path, data_root, FRAME_CACHE_DIR)
        if os.path.exists(cpath):
            already_cached += 1
            continue

        if free_space_gb(FRAME_CACHE_DIR) < MIN_FREE_SPACE_GB:
            stopped_early = True
            break

        try:
            os.makedirs(os.path.dirname(cpath), exist_ok=True)
            clip = build_cached_clip(video_path, CACHE_FRAMES_PER_CLIP, FRAME_SIZE)
            np.save(cpath, clip)
            newly_cached += 1
        except Exception as e:
            # Whatever failed mid-write (disk full, IO error, etc.) may have
            # left a truncated, corrupted .npy behind. Remove it so a future
            # run's "already cached" check doesn't trust a broken file.
            safe_remove(cpath if cpath.endswith(".npy") else cpath + ".npy")
            print(f"[WARN] Failed to cache {os.path.basename(video_path)}: {e}")
            failed += 1

            if isinstance(e, OSError) and getattr(e, "errno", None) == 28:  # ENOSPC
                stopped_early = True
                break

    elapsed = time.time() - start
    remaining = len(all_videos) - (already_cached + newly_cached + failed)

    print(f"\nDone in {elapsed / 60:.1f} minutes.")
    print(f"  Already cached (skipped, resumable): {already_cached}")
    print(f"  Newly cached this run: {newly_cached}")
    print(f"  Failed (cleaned up, not left corrupted): {failed}")
    if stopped_early:
        print(f"  Stopped early: ran out of disk space with {remaining} clips left uncached.")
        print(f"  This is safe — uncached clips decode live during training (no accuracy impact, "
              f"just slightly slower for those specific clips). Free up disk space and re-run this "
              f"script any time to cache more; it resumes from where it left off.")
    print(f"\nSet USE_FRAME_CACHE = True in src/config.py, then run src/train.py — "
          f"training will load cached clips where available and decode the rest live.")


if __name__ == "__main__":
    main()
