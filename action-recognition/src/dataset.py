"""
VideoClipDataset: reads a video file, samples frames, and returns a
(C, T, H, W) tensor ready for torchvision.models.video models.

FRAME CACHING — what is and isn't cached, and why:
  Video decoding (cv2.VideoCapture) is the expensive, repeated cost: every
  epoch re-opens and re-decodes every clip from scratch. Caching the FINAL
  augmented tensor would eliminate that cost but would also freeze random
  flip/crop/normalization to one fixed version forever — killing
  augmentation diversity and measurably hurting generalization.

  Instead, this caches the DECODED, RESIZED frames only (before any
  augmentation), at a denser sampling (CACHE_FRAMES_PER_CLIP, default 32)
  than the model actually uses (NUM_FRAMES, default 16). Each epoch still
  independently: (1) randomly subsamples 16 of the 32 cached frames with
  temporal jitter, (2) applies a fresh random flip/crop, (3) normalizes.
  So augmentation stays exactly as random as before — only the expensive
  decode step is done once instead of every epoch.

  Build the cache with: python data/build_frame_cache.py
  Caching is OFF by default (use_cache=False) — enable it explicitly once
  you've confirmed you have disk space for it (see that script's docstring
  for the size estimate).
"""

import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# ImageNet/Kinetics-style normalization used by torchvision's video models
MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)


def sample_frame_indices(total_frames: int, num_frames: int, train: bool) -> np.ndarray:
    """Evenly spaced frame indices, with a little random jitter during training."""
    if total_frames <= num_frames:
        # repeat frames if the clip is shorter than num_frames
        return np.linspace(0, max(total_frames - 1, 0), num_frames).astype(int)

    if train:
        # temporal jitter: shift the evenly spaced window slightly at random
        max_offset = total_frames // num_frames
        base = np.linspace(0, total_frames - max_offset - 1, num_frames)
        jitter = np.random.randint(0, max(max_offset, 1), size=num_frames)
        indices = (base + jitter).astype(int)
    else:
        indices = np.linspace(0, total_frames - 1, num_frames).astype(int)

    return np.clip(indices, 0, total_frames - 1)


def decode_frames(video_path: str, indices: set, frame_size: int) -> dict:
    """Decodes exactly the requested frame indices from a video file. This
    is the expensive operation frame caching exists to avoid repeating."""
    cap = cv2.VideoCapture(video_path)
    frames = {}
    idx = 0
    while cap.isOpened() and len(frames) < len(indices):
        ret, frame = cap.read()
        if not ret:
            break
        if idx in indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (frame_size, frame_size))
            frames[idx] = frame
        idx += 1
    cap.release()
    return frames


def load_clip(video_path: str, num_frames: int, frame_size: int, train: bool) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    indices = set(sample_frame_indices(total_frames, num_frames, train).tolist())

    frames = decode_frames(video_path, indices, frame_size)

    if not frames:
        # corrupt/unreadable video — return a black clip rather than crashing a batch
        return np.zeros((num_frames, frame_size, frame_size, 3), dtype=np.uint8)

    ordered = sorted(frames.keys())
    clip = np.stack([frames[i] for i in ordered])
    if clip.shape[0] < num_frames:
        pad = np.repeat(clip[-1:], num_frames - clip.shape[0], axis=0)
        clip = np.concatenate([clip, pad], axis=0)
    return clip


def cache_path_for(video_path: str, data_root: str, cache_dir: str) -> str:
    """Mirrors a video's path under data_root into cache_dir, with a .npy extension."""
    rel_path = os.path.relpath(video_path, data_root)
    rel_path = os.path.splitext(rel_path)[0] + ".npy"
    return os.path.join(cache_dir, rel_path)


def build_cached_clip(video_path: str, cache_frames_per_clip: int, frame_size: int) -> np.ndarray:
    """Decodes a dense, fixed (non-jittered) set of frames spanning the
    whole clip — used once at cache-build time. More frames than the model
    needs (CACHE_FRAMES_PER_CLIP > NUM_FRAMES) so later epochs still have
    room to pick a temporally-varied subset."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    indices = set(sample_frame_indices(total_frames, cache_frames_per_clip, train=False).tolist())
    frames = decode_frames(video_path, indices, frame_size)

    if not frames:
        return np.zeros((cache_frames_per_clip, frame_size, frame_size, 3), dtype=np.uint8)

    ordered = sorted(frames.keys())
    clip = np.stack([frames[i] for i in ordered])
    if clip.shape[0] < cache_frames_per_clip:
        pad = np.repeat(clip[-1:], cache_frames_per_clip - clip.shape[0], axis=0)
        clip = np.concatenate([clip, pad], axis=0)
    return clip


class VideoClipDataset(Dataset):
    """
    Expects a directory layout of:
        root/<ClassName>/*.avi
    """

    def __init__(self, root: str, classes: list, num_frames: int, frame_size: int, train: bool,
                 use_cache: bool = False, cache_dir: str = None, cache_frames_per_clip: int = 32,
                 data_root: str = None):
        self.root = root
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.train = train
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.cache_frames_per_clip = cache_frames_per_clip
        # data_root is the common ancestor cache paths are mirrored relative
        # to (so train/ and test/ both land under cache_dir consistently).
        self.data_root = data_root or os.path.dirname(root.rstrip(os.sep))

        self.samples = []
        for class_name in classes:
            class_dir = os.path.join(root, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".avi", ".mp4")):
                    self.samples.append((os.path.join(class_dir, fname), self.class_to_idx[class_name]))

        if not self.samples:
            raise RuntimeError(
                f"No video files found under {root}. Did you run data/download_ucf101.py first?"
            )

        if self.use_cache:
            cached_count = sum(
                1 for path, _ in self.samples
                if os.path.exists(cache_path_for(path, self.data_root, self.cache_dir))
            )
            print(f"Frame cache: {cached_count}/{len(self.samples)} clips cached under {cache_dir} "
                  f"({'all cached' if cached_count == len(self.samples) else 'partial — uncached clips will decode live and be slower'}).")

    def __len__(self):
        return len(self.samples)

    def _get_source_frames(self, path: str) -> np.ndarray:
        """Returns the frame pool to sample from: cached (if available) or freshly decoded."""
        if self.use_cache:
            cpath = cache_path_for(path, self.data_root, self.cache_dir)
            if os.path.exists(cpath):
                try:
                    return np.load(cpath)  # shape: (cache_frames_per_clip, H, W, 3)
                except (OSError, ValueError, EOFError) as e:
                    # A corrupted/truncated cache file (e.g. from a run that
                    # ran out of disk mid-write before this safeguard existed)
                    # should fall back to live decoding for this clip, not
                    # crash the whole training run. The raw video is fine —
                    # only the cache entry for it is bad.
                    print(f"[WARN] Corrupted cache file for {os.path.basename(path)} "
                          f"({e}) — decoding live instead. Consider deleting "
                          f"{cpath} and re-running data/build_frame_cache.py.")
        # Cache miss or caching disabled — decode live at model's target frame count.
        return load_clip(path, self.num_frames, self.frame_size, self.train)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        source_frames = self._get_source_frames(path)

        if source_frames.shape[0] != self.num_frames:
            # We have a denser cached pool — pick num_frames from it, with the
            # same jitter-during-training logic used for live decoding, so
            # temporal augmentation diversity is preserved from the cache too.
            pool_size = source_frames.shape[0]
            pick = sample_frame_indices(pool_size, self.num_frames, self.train)
            clip = source_frames[pick]
        else:
            clip = source_frames

        if self.train:
            if random.random() < 0.5:
                clip = clip[:, :, ::-1, :]  # horizontal flip
            clip = self._random_crop_resize(clip)

        clip = clip.astype(np.float32) / 255.0
        clip = (clip - MEAN) / STD
        clip = torch.from_numpy(clip.copy()).permute(3, 0, 1, 2).float()  # (C, T, H, W)
        return clip, label

    def _random_crop_resize(self, clip: np.ndarray) -> np.ndarray:
        """Random crop to ~90% then resize back — a mild spatial augmentation."""
        t, h, w, c = clip.shape
        crop_h, crop_w = int(h * 0.9), int(w * 0.9)
        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)
        cropped = clip[:, top:top + crop_h, left:left + crop_w, :]
        resized = np.stack([cv2.resize(f, (w, h)) for f in cropped])
        return resized
