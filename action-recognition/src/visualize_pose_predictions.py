"""
Produces "sports-analysis style" annotated videos: takes a handful of
random test clips, overlays a MediaPipe Pose skeleton on every frame,
and stamps the true vs. predicted action label — showing how the
model's decision lines up with the actual body movement in each clip.

This is the video-level equivalent of KOLSA's joint-marking visuals,
adapted to action classification instead of a juggling quality score.

NOTE ON MEDIAPIPE VERSIONS: the old `mp.solutions.pose` API was removed
in MediaPipe >=0.10.15 (and isn't installable at all on recent Python
versions anyway). This script uses the current Tasks API instead
(`mediapipe.tasks.python.vision.PoseLandmarker`), which needs a small
model file downloaded once — handled automatically below.

Usage:
    python src/visualize_pose_predictions.py --checkpoint checkpoints/best_model.pt --num_clips 5

Output:
    outputs/pose_visualizations/<n>_true-X_pred-Y.mp4
"""

import argparse
import os
import random
import sys
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import DATA_DIR, FRAME_SIZE, NUM_FRAMES, OUTPUT_DIR, SEED, get_classes  # noqa: E402
from src.dataset import MEAN, STD, load_clip  # noqa: E402
from src.model import build_model  # noqa: E402
from src.utils import get_device  # noqa: E402

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pose_landmarker_lite.task")

# Standard 33-point BlazePose skeleton connections (the same topology
# the old mp.solutions.pose.POSE_CONNECTIONS used to provide directly).
POSE_CONNECTIONS = [
    (0, 1), (0, 4), (1, 2), (2, 3), (3, 7), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (11, 23), (12, 14), (12, 24),
    (13, 15), (14, 16), (15, 17), (15, 19), (15, 21), (16, 18),
    (16, 20), (16, 22), (17, 19), (18, 20), (23, 24), (23, 25),
    (24, 26), (25, 27), (26, 28), (27, 29), (27, 31), (28, 30),
    (28, 32), (29, 31), (30, 32),
]


def ensure_model_downloaded() -> str:
    model_path = os.path.abspath(MODEL_PATH)
    if not os.path.exists(model_path):
        print("Downloading pose landmarker model (one-time)...")
        urllib.request.urlretrieve(MODEL_URL, model_path)
    return model_path


def draw_pose(frame: np.ndarray, landmarks) -> None:
    """Manually draws joints + skeleton lines, since the Tasks API has no built-in drawing helper."""
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

    for a, b in POSE_CONNECTIONS:
        if a < len(points) and b < len(points):
            cv2.line(frame, points[a], points[b], (0, 255, 180), 2)

    for x, y in points:
        cv2.circle(frame, (x, y), 4, (0, 140, 255), -1)


def predict_clip(model, video_path: str, device, num_frames: int, frame_size: int) -> int:
    """Runs the classifier on a clip the same way evaluate.py does, to get a predicted label."""
    clip = load_clip(video_path, num_frames, frame_size, train=False).astype(np.float32) / 255.0
    clip = (clip - MEAN) / STD
    tensor = torch.from_numpy(clip.copy()).permute(3, 0, 1, 2).unsqueeze(0).float().to(device)
    with torch.no_grad():
        pred = model(tensor).argmax(dim=1).item()
    return pred


def annotate_video(landmarker, video_path: str, out_path: str, true_label: str, pred_label: str) -> None:
    """Draws pose skeleton + label banner on every frame, writes an annotated mp4."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    correct = true_label == pred_label
    banner_color = (0, 200, 0) if correct else (0, 0, 220)  # BGR: green if correct, red if wrong

    frame_idx = 0
    frame_duration_ms = int(1000 / fps)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = frame_idx * frame_duration_ms
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks:
            draw_pose(frame, result.pose_landmarks[0])

        banner_h = 46
        cv2.rectangle(frame, (0, 0), (width, banner_h), (30, 30, 30), thickness=-1)
        text = f"true: {true_label}   pred: {pred_label}"
        cv2.putText(frame, text, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, banner_color, 2)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.path.join("checkpoints", "best_model.pt"))
    parser.add_argument("--num_clips", type=int, default=5)
    args = parser.parse_args()

    random.seed(SEED)
    device = get_device()

    out_dir = os.path.join(OUTPUT_DIR, "pose_visualizations")
    os.makedirs(out_dir, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=device)
    classes = ckpt.get("classes", get_classes())

    model = build_model(num_classes=len(classes), freeze_backbone=False, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    test_root = os.path.join(DATA_DIR, "test")
    all_clips = []
    for class_name in classes:
        class_dir = os.path.join(test_root, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".avi", ".mp4")):
                all_clips.append((os.path.join(class_dir, fname), class_name))

    if not all_clips:
        raise RuntimeError(f"No test clips found under {test_root}")

    chosen = random.sample(all_clips, min(args.num_clips, len(all_clips)))

    model_path = ensure_model_downloaded()

    for i, (video_path, true_label) in enumerate(chosen, start=1):
        pred_idx = predict_clip(model, video_path, device, NUM_FRAMES, FRAME_SIZE)
        pred_label = classes[pred_idx]

        status = "correct" if pred_label == true_label else "WRONG"
        out_name = f"{i:02d}_true-{true_label}_pred-{pred_label}.mp4"
        out_path = os.path.join(out_dir, out_name)

        print(f"[{i}/{len(chosen)}] {os.path.basename(video_path)} "
              f"| true: {true_label} | pred: {pred_label} | {status}")

        # A fresh landmarker per clip avoids "timestamp must be monotonically
        # increasing" errors — each clip's frame timestamps restart at 0, and
        # the Tasks API tracks timestamps per landmarker instance, not per video.
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        )
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            annotate_video(landmarker, video_path, out_path, true_label, pred_label)

    print(f"\nSaved {len(chosen)} pose-annotated videos to: {out_dir}")


if __name__ == "__main__":
    main()
