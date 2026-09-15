"""
Central configuration for the action recognition project.

Two class-selection modes:
  1. USE_ALL_CLASSES = True (default) — trains on every class UCF101 has (101).
     data/download_ucf101.py auto-discovers them and writes the resulting
     list to data/UCF101_subset/classes.txt, which get_classes() reads.
  2. USE_ALL_CLASSES = False — uses the fixed CLASSES list below instead,
     useful for fast local iteration/debugging on a smaller class count.
"""

import os

# ---- Class selection ----------------------------------------------------
USE_ALL_CLASSES = True   # False => use the fixed CLASSES list below instead

CLASSES = [
    "JugglingBalls", "JumpRope", "JumpingJack", "WalkingWithDog", "Basketball",
    "BasketballDunk", "TennisSwing", "GolfSwing", "PushUps", "PullUps",
    "Bowling", "BoxingPunchingBag", "Skiing", "SkyDiving", "HorseRiding",
    "SoccerPenalty", "SoccerJuggling", "RockClimbingIndoor", "Archery",
    "Fencing", "Diving", "CliffDiving", "Surfing", "TableTennisShot",
    "VolleyballSpiking", "HighJump", "LongJump", "PlayingGuitar",
]

# ---- Paths --------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "UCF101_subset")
CLASSES_FILE = os.path.join(DATA_DIR, "classes.txt")
CLASS_DIST_FILE = os.path.join(DATA_DIR, "class_distribution.json")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
HISTORY_FILE = os.path.join(OUTPUT_DIR, "training_history.json")
BEST_HPARAMS_FILE = os.path.join(OUTPUT_DIR, "best_hparams.json")


def get_classes() -> list:
    """
    Returns the class list actually used to build the dataset on disk.
    classes.txt (written by data/download_ucf101.py once it has actually
    run) always takes priority over the hardcoded CLASSES above, so
    train/evaluate/visualize/frontend never disagree with what's on disk.
    """
    if os.path.exists(CLASSES_FILE):
        with open(CLASSES_FILE) as f:
            return [line.strip() for line in f if line.strip()]
    return CLASSES


NUM_CLASSES = len(get_classes())

# ---- Frame caching (optional, for training speed) -----------------------
# See src/dataset.py's module docstring and data/build_frame_cache.py for
# the full reasoning. Off by default — build the cache explicitly with
# `python data/build_frame_cache.py` once you've confirmed disk space,
# then flip this to True.
USE_FRAME_CACHE = False
FRAME_CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "frame_cache")
CACHE_FRAMES_PER_CLIP = 32   # > NUM_FRAMES, so temporal jitter still has room to vary each epoch

# ---- Video sampling -----------------------------------------------------
NUM_FRAMES = 16          # frames sampled per clip
FRAME_SIZE = 112         # H = W = 112, matches r2plus1d_18 pretraining
TRAIN_SPLIT = 0.8        # per-group random split (see data/download_ucf101.py)

# ---- Training -------------------------------------------------------------
BATCH_SIZE = 8
NUM_EPOCHS = 40                  # upper bound — early stopping usually ends training sooner
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 2                # linear LR warmup before cosine decay kicks in
GRAD_CLIP_NORM = 5.0             # gradient clipping — stability precaution at 101-class scale
NUM_WORKERS = 2
SEED = 42

# Precaution: UCF101 classes are naturally imbalanced (roughly 70-160 clips
# per class). Rather than ignore this, the training loss is weighted by
# inverse class frequency (computed at train time from the actual dataset
# on disk) so rare classes aren't drowned out by common ones.
USE_CLASS_WEIGHTS = True

# Precaution: rather than training a fixed number of epochs regardless of
# whether the model is still improving, training stops early once
# validation loss stops improving for EARLY_STOPPING_PATIENCE epochs in a
# row (by more than EARLY_STOPPING_MIN_DELTA). This avoids wasting GPU
# time on free-tier Colab/Kaggle quotas, and avoids overfitting.
EARLY_STOPPING_PATIENCE = 6
EARLY_STOPPING_MIN_DELTA = 0.001

# Regularization technique explicitly requested by the project brief.
# 0.0 disables it with zero architectural change (see model.py — dropout
# is inserted without altering any saved checkpoint's parameter keys).
DROPOUT_RATE = 0.2

# Opt-in alternative to plain (class-weighted) cross-entropy, aimed at
# classes that are confidently misclassified due to visual/motion
# similarity (e.g. Basketball vs BasketballDunk) rather than rarity alone.
# Off by default — this changes training dynamics and hasn't been
# validated against the project's already-verified baseline accuracy, so
# it's offered as a documented experiment, not a silent replacement.
USE_FOCAL_LOSS = False
FOCAL_LOSS_GAMMA = 2.0

DEVICE = "cuda"  # falls back to "cpu" automatically in train.py if unavailable


