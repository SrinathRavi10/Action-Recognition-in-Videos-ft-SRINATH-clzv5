"""
Action recognition dashboard — single continuous page.

Every section lives on one page (no tabs): hero KPIs, live classification
(the primary interactive feature, near the top), dataset balance,
training, and evaluation with an interactive confusion-matrix heatmap
and per-class metrics chart. Any section without underlying data renders
one compact message instead of empty space.

Live classification can pull ANY clip from ANY of the 101 UCF101
classes on demand — via HTTP range requests against the remote dataset
archive, fetching only that one clip's bytes, not the ~7GB dataset (see
src/remote_dataset.py) — or accept an uploaded video. The fetched
temp file is deleted as soon as a new clip is chosen or the session
moves on.

Run locally after downloading the trained artifacts (checkpoint,
outputs/) from Colab/Kaggle:

    streamlit run app/streamlit_app.py
"""

import json
import os
import random
import sys
import time

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (  # noqa: E402
    CLASS_DIST_FILE, DROPOUT_RATE, EARLY_STOPPING_PATIENCE, FRAME_SIZE,
    GRAD_CLIP_NORM, HISTORY_FILE, LEARNING_RATE, NUM_FRAMES, OUTPUT_DIR,
    USE_ALL_CLASSES, USE_CLASS_WEIGHTS, USE_FOCAL_LOSS, WARMUP_EPOCHS,
    get_classes,
)
from src.dataset import MEAN, STD, load_clip, sample_frame_indices  # noqa: E402
from src.motion_visualization import (  # noqa: E402
    compute_clip_motion_energy, compute_optical_flow, flow_to_color,
)
from src.model import build_model  # noqa: E402
from src.utils import get_device  # noqa: E402
from src.remote_dataset import fetch_clip_to_temp, list_remote_clips  # noqa: E402

CHECKPOINT_PATH = os.path.join("checkpoints", "best_model.pt")

st.set_page_config(page_title="Action Recognition Dashboard", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1300px;}

    h1 {
        font-weight: 700; letter-spacing: -0.5px;
        background: linear-gradient(90deg, #22d3ee, #6366f1);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    h2 {
        font-weight: 600; letter-spacing: 0.5px; text-transform: uppercase; font-size: 1.05rem;
        color: #22d3ee; border-bottom: 1px solid rgba(34,211,238,0.25); padding-bottom: 10px;
        margin-top: 3rem;
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(160deg, rgba(34,211,238,0.08), rgba(99,102,241,0.05));
        border: 1px solid rgba(34,211,238,0.18);
        border-radius: 12px;
        padding: 14px 18px;
    }
    div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; color: #22d3ee; }
    .status-pill {
        display: inline-block; padding: 4px 14px; border-radius: 999px;
        font-size: 13px; font-weight: 600; font-family: 'JetBrains Mono', monospace;
    }
    .status-ok { background: rgba(34,197,94,0.15); color: #22c55e; box-shadow: 0 0 12px rgba(34,197,94,0.25); }
    .status-bad { background: rgba(239,68,68,0.15); color: #ef4444; box-shadow: 0 0 12px rgba(239,68,68,0.25); }
    .status-neutral { background: rgba(148,163,184,0.15); color: #94a3b8; }
    .video-label {
        font-family: 'JetBrains Mono', monospace; font-size: 12px; letter-spacing: 1px;
        text-transform: uppercase; color: #22d3ee; margin-bottom: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model_and_classes():
    if not os.path.exists(CHECKPOINT_PATH):
        return None, None, None, None
    try:
        device = get_device()
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
        classes = ckpt.get("classes", get_classes())
        model = build_model(num_classes=len(classes), freeze_backbone=False, pretrained=False).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model, classes, device, None
    except Exception as e:
        return None, None, None, str(e)


@st.cache_data
def load_json_file(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_resource(show_spinner="Listing available clips from the remote dataset...")
def get_remote_clip_index():
    try:
        return list_remote_clips(), None
    except Exception as e:
        return None, str(e)


def predict_probs(model, device, clip: np.ndarray) -> np.ndarray:
    clip_n = (clip.astype(np.float32) / 255.0 - MEAN) / STD
    tensor = torch.from_numpy(clip_n.copy()).permute(3, 0, 1, 2).unsqueeze(0).float().to(device)
    with torch.no_grad():
        return torch.softmax(model(tensor), dim=1)[0].cpu().numpy()


def top_k_bar(probs: np.ndarray, classes: list, k: int = 5, highlight: str = None) -> go.Figure:
    idx = probs.argsort()[-k:][::-1]
    labels = [classes[i] for i in idx]
    values = [float(probs[i]) for i in idx]
    colors = ["#16a34a" if labels[i] == highlight else "#3b82f6" for i in range(len(labels))]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=colors))
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=280,
                       margin=dict(l=10, r=10, t=10, b=10), xaxis_range=[0, 1])
    return fig


def draw_pose_overlay(frame: np.ndarray, landmarker, timestamp_ms: int):
    """Draws a skeleton over one frame using MediaPipe's Tasks API. Returns
    the annotated frame (a copy); falls back to the original frame if no
    person is detected."""
    import mediapipe as mp
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.shape[-1] == 3 else frame
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb.copy())
    result = landmarker.detect_for_video(mp_image, timestamp_ms)
    annotated = frame.copy()
    if result.pose_landmarks:
        h, w = annotated.shape[:2]
        points = [(int(lm.x * w), int(lm.y * h)) for lm in result.pose_landmarks[0]]
        connections = [
            (0, 1), (0, 4), (1, 2), (2, 3), (3, 7), (4, 5), (5, 6), (6, 8),
            (9, 10), (11, 12), (11, 13), (11, 23), (12, 14), (12, 24),
            (13, 15), (14, 16), (15, 17), (15, 19), (15, 21), (16, 18),
            (16, 20), (16, 22), (17, 19), (18, 20), (23, 24), (23, 25),
            (24, 26), (25, 27), (26, 28), (27, 29), (27, 31), (28, 30),
            (28, 32), (29, 31), (30, 32),
        ]
        for a, b in connections:
            if a < len(points) and b < len(points):
                cv2.line(annotated, points[a], points[b], (0, 255, 180), 2)
        for x, y in points:
            cv2.circle(annotated, (x, y), 4, (0, 140, 255), -1)
    return annotated


def encode_frames_to_mp4(frames: list, fps: float, out_path: str) -> tuple:
    """
    Encodes RGB frames into a real, browser-playable H.264 mp4.

    Uses imageio-ffmpeg, which bundles its OWN ffmpeg binary as a pip
    package — unlike calling a system `ffmpeg` executable, this works
    identically on any machine with just `pip install`, no separate
    system install required (a real gap found in testing: a user's
    machine had no system ffmpeg, silently falling back to OpenCV's
    mp4v encoder, which browsers can't play — that's why videos showed
    "0:00" and never loaded).

    Returns (used_h264: bool, error_detail: str | None).
    """
    try:
        import imageio.v2 as imageio
        writer = imageio.get_writer(out_path, fps=fps, codec="libx264", pixelformat="yuv420p")
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return True, None
        error_detail = "imageio-ffmpeg produced no output"
    except Exception as e:
        error_detail = f"imageio-ffmpeg failed: {e}"

    # Fallback: OpenCV mp4v (works, but not guaranteed to play in every browser)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        return False, f"Both imageio-ffmpeg and OpenCV fallback failed. Detail: {error_detail}"
    return False, error_detail


@st.cache_resource
def get_pose_landmarker_options():
    """Downloads the pose model once (cached across the session) and
    returns ready-to-use options — a fresh landmarker instance is still
    created per video, since the Tasks API requires strictly increasing
    timestamps per instance."""
    import urllib.request
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    model_path = os.path.join(OUTPUT_DIR, "pose_landmarker_lite.task")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(model_path):
        url = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")
        urllib.request.urlretrieve(url, model_path)

    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    return vision.PoseLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.VIDEO, num_poses=1)


classes = get_classes()
model, ckpt_classes, device, load_error = load_model_and_classes()
class_dist = load_json_file(CLASS_DIST_FILE)
history = load_json_file(HISTORY_FILE)
hparams = load_json_file(os.path.join(OUTPUT_DIR, "best_hparams.json"))
confused_pairs = load_json_file(os.path.join(OUTPUT_DIR, "confused_pairs.json"))
cm_data = load_json_file(os.path.join(OUTPUT_DIR, "confusion_matrix_data.json"))

report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
top1_acc = top5_acc = avg_ms = None
if os.path.exists(report_path):
    with open(report_path) as f:
        for line in f:
            if line.startswith("Top-1 accuracy"):
                top1_acc = line.split(":")[1].strip()
            elif line.startswith("Top-5 accuracy"):
                top5_acc = line.split(":")[1].strip()
            elif line.startswith("Avg inference"):
                avg_ms = line.split(":")[1].strip()

# ---------------------------------------------------------------------------
# HERO
# ---------------------------------------------------------------------------
st.title("Action recognition dashboard")
st.caption("R(2+1)D-18 fine-tuned on UCF101 — every chart below reads real artifacts from this project's own scripts.")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Classes", len(classes))
c2.metric("Top-1 accuracy", top1_acc or "—")
c3.metric("Top-5 accuracy", top5_acc or "—")
c4.metric("Avg inference", avg_ms or "—")
c5.metric("Model", "R(2+1)D-18")

with st.expander("Approach and precautions"):
    st.markdown(
        f"R(2+1)D-18 pretrained on Kinetics-400, fine-tuned on "
        f"{'the full 101-class UCF101 dataset' if USE_ALL_CLASSES else 'a UCF101 subset'} "
        f"with the last residual block and classifier head unfrozen. "
        f"Loss: {'focal loss' if USE_FOCAL_LOSS else 'cross-entropy'}"
        f"{', class-weighted' if USE_CLASS_WEIGHTS else ''}. "
        f"Dropout {DROPOUT_RATE} on the classifier head. Early stopping, gradient "
        f"clipping, and mixed precision throughout.\n\n"
        "- **Data leakage** — clips split at the source-group level, not per-clip. "
        "An earlier per-clip split let the model see near-duplicate footage across "
        "train/test and produced a false 100% accuracy — caught and fixed.\n"
        "- **Class imbalance** — loss weighted by inverse frequency from real on-disk counts.\n"
        "- **Early stopping** — halts on validation-loss plateau, avoiding wasted free-tier GPU time.\n"
        "- **Bounded hyperparameter search** — a full grid search isn't feasible on free-tier "
        "GPU time; a small documented probe picks a starting configuration instead.\n"
        "- **On-demand clip fetching** — live classification below pulls single clips via HTTP "
        "range requests against the remote dataset archive, never downloading the full ~7GB "
        "dataset just to browse it."
    )

# ---------------------------------------------------------------------------
# LIVE CLASSIFICATION
# ---------------------------------------------------------------------------
st.header("Live classification")

if load_error is not None:
    st.error(f"Could not load the trained model: {load_error}")
elif model is None:
    st.info("No trained checkpoint yet. Run `python src/train.py` and place `best_model.pt` in `checkpoints/`.")
else:
    source = st.radio("Video source", ["Fetch from dataset (all 101 classes)", "Upload a video"],
                       horizontal=True, label_visibility="collapsed")

    live_path = None
    live_true_label = None
    is_temp_remote_file = False

    if source == "Fetch from dataset (all 101 classes)":
        remote_index, remote_error = get_remote_clip_index()
        if remote_error is not None:
            st.error(f"Could not reach the remote dataset: {remote_error}")
        else:
            col_cat, col_clip = st.columns(2)
            with col_cat:
                category = st.selectbox("Category", sorted(remote_index.keys()))
            with col_clip:
                clip_options = remote_index[category]
                chosen_member = st.selectbox(
                    "Clip", clip_options, format_func=lambda p: os.path.basename(p)
                )
            live_true_label = category

            if st.session_state.get("current_member") != chosen_member:
                old_path = st.session_state.get("current_temp_path")
                if old_path and os.path.exists(old_path):
                    os.remove(old_path)
                with st.spinner(f"Fetching just this one clip ({os.path.basename(chosen_member)})..."):
                    st.session_state["current_temp_path"] = fetch_clip_to_temp(chosen_member)
                st.session_state["current_member"] = chosen_member

            live_path = st.session_state["current_temp_path"]
            is_temp_remote_file = True
    else:
        uploaded = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
        if uploaded is not None:
            import tempfile
            suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(uploaded.read())
            tmp.close()
            live_path = tmp.name
            st.caption(f"Uploaded: `{uploaded.name}` — no ground truth, so only the prediction is shown.")

    if live_path is not None:
        show_pose = st.checkbox("Show pose skeleton overlay", value=True)
        start = st.button("Start live classification", type="primary")

        live_clip_arr = load_clip(live_path, NUM_FRAMES, FRAME_SIZE, train=False)
        left_col, right_col = st.columns(2)

        if start:
            with st.spinner("Decoding clip, classifying, and rendering pose overlay..."):
                # Decode a denser set of frames than the model uses, purely
                # for smooth video playback (the model's prediction still
                # comes from the standard NUM_FRAMES sample below).
                cap = cv2.VideoCapture(live_path)
                native_fps = cap.get(cv2.CAP_PROP_FPS)
                # Some codecs report 0, None, or NaN for fps — "x or 15.0"
                # doesn't catch NaN (NaN is truthy in Python), which was
                # silently corrupting the encoded video's timing/duration.
                if not native_fps or native_fps != native_fps or native_fps <= 0 or native_fps > 120:
                    native_fps = 15.0
                total_native = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_native <= 0:
                    total_native = NUM_FRAMES * 4  # sane bound if frame count is also unreported
                playback_frames = []
                max_playback_frames = 150  # bounds processing time on long clips
                while cap.isOpened() and len(playback_frames) < min(total_native, max_playback_frames):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    playback_frames.append(cv2.cvtColor(
                        cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE)), cv2.COLOR_BGR2RGB))
                cap.release()

                if not playback_frames:
                    st.error("Could not decode any frames from this clip — it may be corrupted. Try a different clip.")
                    st.stop()

                probs = predict_probs(model, device, live_clip_arr)
                final_pred = ckpt_classes[int(probs.argmax())]

                if live_true_label is not None:
                    correct = final_pred == live_true_label
                    banner_color = (34, 197, 94) if correct else (239, 68, 68)
                    banner_text = f"true: {live_true_label}  pred: {final_pred}"
                else:
                    banner_color = (59, 130, 246)
                    banner_text = f"pred: {final_pred}"

                landmarker_cm = None
                if show_pose:
                    try:
                        from mediapipe.tasks.python import vision
                        options = get_pose_landmarker_options()
                        landmarker_cm = vision.PoseLandmarker.create_from_options(options)
                    except Exception as e:
                        st.warning(f"Pose overlay unavailable this run ({e}) — showing plain video instead.")
                        show_pose = False

                left_frames, right_frames = [], []
                frame_duration_ms = int(1000 / native_fps) or 33
                for i, frame in enumerate(playback_frames):
                    banner_frame = frame.copy()
                    cv2.rectangle(banner_frame, (0, 0), (FRAME_SIZE, 28), (20, 20, 20), -1)
                    cv2.putText(banner_frame, banner_text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, banner_color, 1, cv2.LINE_AA)
                    left_frames.append(banner_frame)

                    if show_pose and landmarker_cm is not None:
                        annotated = draw_pose_overlay(frame, landmarker_cm, i * frame_duration_ms)
                    else:
                        annotated = frame.copy()
                    cv2.rectangle(annotated, (0, 0), (FRAME_SIZE, 28), (20, 20, 20), -1)
                    cv2.putText(annotated, banner_text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, banner_color, 1, cv2.LINE_AA)
                    right_frames.append(annotated)

                if landmarker_cm is not None:
                    landmarker_cm.close()

                for key in ("left_video_path", "right_video_path"):
                    old = st.session_state.get(key)
                    if old and os.path.exists(old):
                        os.remove(old)

                import tempfile
                left_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
                right_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
                left_ok, left_err = encode_frames_to_mp4(left_frames, native_fps, left_path)
                right_ok, right_err = encode_frames_to_mp4(right_frames, native_fps, right_path)
                st.session_state["left_video_path"] = left_path
                st.session_state["right_video_path"] = right_path

            if not left_ok or not right_ok:
                st.warning(
                    "Video encoding didn't use the H.264 path (may not play in every browser). "
                    f"Detail: {left_err or right_err}"
                )

            left_col.markdown("<div class='video-label'>Original</div>", unsafe_allow_html=True)
            left_col.video(st.session_state["left_video_path"])
            right_col.markdown("<div class='video-label'>Pose overlay + live classification</div>", unsafe_allow_html=True)
            right_col.video(st.session_state["right_video_path"])

            m1, m2, m3 = st.columns(3)
            m1.metric("Prediction", final_pred)
            m2.metric("Confidence", f"{float(probs.max()) * 100:.1f}%")
            if live_true_label is not None:
                pill_class = "status-ok" if correct else "status-bad"
                pill_text = "Correct" if correct else "Incorrect"
            else:
                pill_class, pill_text = "status-neutral", "No ground truth"
            m3.markdown(f"<div style='padding-top:8px'>"
                        f"<span class='status-pill {pill_class}'>{pill_text}</span></div>",
                        unsafe_allow_html=True)
            st.plotly_chart(top_k_bar(probs, ckpt_classes, highlight=live_true_label), width="stretch")
        else:
            left_col.markdown("<div class='video-label'>Original</div>", unsafe_allow_html=True)
            left_col.image(live_clip_arr[0], width="stretch")
            right_col.markdown("<div class='video-label'>Pose overlay + live classification</div>", unsafe_allow_html=True)
            right_col.image(live_clip_arr[0], width="stretch")
            st.caption("Press **Start live classification** to render both views.")

# ---------------------------------------------------------------------------
# DATASET
# ---------------------------------------------------------------------------
st.header("Dataset")
if class_dist is None:
    st.info("No dataset distribution found yet. Run `python data/download_ucf101.py`.")
else:
    df = pd.DataFrame([
        {"class": k, "train": v["train"], "test": v["test"]} for k, v in class_dist.items()
    ]).sort_values("train", ascending=False)
    train_counts = df["train"][df["train"] > 0]
    imbalance_ratio = train_counts.max() / train_counts.min() if len(train_counts) else 0

    col1, col2 = st.columns([3, 1])
    with col1:
        fig = px.bar(df, x="class", y=["train", "test"], barmode="group")
        fig.update_layout(xaxis_tickangle=-60, height=460, margin=dict(t=10),
                           legend=dict(orientation="h", y=1.05))
        st.plotly_chart(fig, width="stretch")
    with col2:
        st.metric("Classes", len(df))
        st.metric("Train clips", int(df["train"].sum()))
        st.metric("Test clips", int(df["test"].sum()))
        st.metric("Imbalance ratio", f"{imbalance_ratio:.2f}x")

# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------
st.header("Training")
col1, col2 = st.columns(2)
with col1:
    with st.container(border=True):
        st.markdown("**Configuration**")
        st.markdown(
            f"- Learning rate: `{LEARNING_RATE}` ({WARMUP_EPOCHS}-epoch warmup, cosine decay)\n"
            f"- Gradient clip norm: `{GRAD_CLIP_NORM}`\n"
            f"- Dropout: `{DROPOUT_RATE}`\n"
            f"- Loss: `{'focal' if USE_FOCAL_LOSS else 'cross-entropy'}"
            f"{' + class weights' if USE_CLASS_WEIGHTS else ''}`\n"
            f"- Early stopping patience: `{EARLY_STOPPING_PATIENCE}` epochs"
        )
with col2:
    if hparams is None:
        st.info("No hyperparameter probe results yet. Run `python src/hyperparam_search.py`.")
    else:
        with st.container(border=True):
            st.markdown("**Hyperparameter probe** (bounded, not exhaustive)")
            hp_df = pd.DataFrame(hparams["all_results"]).sort_values("probe_val_acc", ascending=False)
            st.dataframe(hp_df, width="stretch", hide_index=True)

if history is not None:
    hist_df = pd.DataFrame(history)
    best_epoch = hist_df.loc[hist_df["val_acc"].idxmax(), "epoch"]

    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(x=hist_df["epoch"], y=hist_df["train_loss"], name="train"))
    fig_loss.add_trace(go.Scatter(x=hist_df["epoch"], y=hist_df["val_loss"], name="val"))
    fig_loss.add_vline(x=best_epoch, line_dash="dash")
    fig_loss.update_layout(title="Loss", height=340, margin=dict(t=40))

    fig_acc = go.Figure()
    fig_acc.add_trace(go.Scatter(x=hist_df["epoch"], y=hist_df["train_acc"], name="train"))
    fig_acc.add_trace(go.Scatter(x=hist_df["epoch"], y=hist_df["val_acc"], name="val"))
    fig_acc.add_vline(x=best_epoch, line_dash="dash")
    fig_acc.update_layout(title="Accuracy", height=340, margin=dict(t=40))

    c1, c2 = st.columns(2)
    c1.plotly_chart(fig_loss, width="stretch")
    c2.plotly_chart(fig_acc, width="stretch")
    st.caption(f"{len(hist_df)} epochs run, best at epoch {int(best_epoch)}"
               f"{' (stopped early)' if len(hist_df) < 40 else ''}.")
else:
    st.info("No training history yet. Run `python src/train.py`.")

# ---------------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------------
st.header("Evaluation")

if cm_data is not None:
    cm = np.array(cm_data["matrix"])
    cm_classes = cm_data["classes"]

    st.markdown("**Confusion matrix** — hover any cell for exact counts; diagonal is correct predictions.")
    fig_cm = px.imshow(
        cm, x=cm_classes, y=cm_classes, color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="True", color="Count"),
        aspect="auto",
    )
    fig_cm.update_layout(height=800, margin=dict(t=10))
    fig_cm.update_xaxes(tickangle=-60, tickfont=dict(size=9))
    fig_cm.update_yaxes(tickfont=dict(size=9))
    st.plotly_chart(fig_cm, width="stretch")

    per_class_df = pd.DataFrame(cm_data["per_class_metrics"]).sort_values("f1")
    st.markdown("**Per-class F1 score** — sorted weakest to strongest")
    fig_f1 = px.bar(per_class_df, x="f1", y="class", orientation="h",
                     color="f1", color_continuous_scale="RdYlGn", range_color=[0, 1])
    fig_f1.update_layout(height=max(400, len(per_class_df) * 11), margin=dict(t=10),
                          yaxis=dict(tickfont=dict(size=8)))
    st.plotly_chart(fig_f1, width="stretch")
elif os.path.exists(os.path.join(OUTPUT_DIR, "confusion_matrix.png")):
    st.image(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), width="stretch")
    st.caption("Re-run `python src/evaluate.py` to get the interactive, hoverable version of this chart.")
else:
    st.info("No evaluation results yet. Run `python src/evaluate.py`.")

if confused_pairs:
    with st.container(border=True):
        st.markdown("**Most-confused class pairs**")
        st.dataframe(pd.DataFrame(confused_pairs), width="stretch", hide_index=True)

report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
if os.path.exists(report_path):
    with st.expander("Full classification report (text)"):
        with open(report_path) as f:
            st.text(f.read())

# ---------------------------------------------------------------------------
# PREPROCESSING & MOTION (compact, at the bottom — supporting detail, not the headline)
# ---------------------------------------------------------------------------
st.header("Preprocessing and motion")
st.caption("What every clip goes through before reaching the model, and how it captures motion dynamics.")

remote_index_for_preview, remote_error_preview = get_remote_clip_index() if model is not None else (None, "no model")
if remote_index_for_preview:
    if st.button("Preview preprocessing on a random clip"):
        st.session_state.pop("preprocess_temp_path", None)
    if "preprocess_temp_path" not in st.session_state:
        rand_class = random.choice(list(remote_index_for_preview.keys()))
        rand_member = random.choice(remote_index_for_preview[rand_class])
        with st.spinner("Fetching one clip for the preprocessing preview..."):
            st.session_state["preprocess_temp_path"] = fetch_clip_to_temp(rand_member)
        st.session_state["preprocess_class"] = rand_class

    clip_path = st.session_state["preprocess_temp_path"]
    st.write(f"Previewing: **{st.session_state.get('preprocess_class', '?')}**")

    cap = cv2.VideoCapture(clip_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = sorted(set(sample_frame_indices(total_frames, NUM_FRAMES, train=False).tolist()))
    raw_frames = []
    idx, wanted = 0, set(indices)
    while cap.isOpened() and len(raw_frames) < len(wanted):
        ret, frame = cap.read()
        if not ret:
            break
        if idx in wanted:
            raw_frames.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (FRAME_SIZE, FRAME_SIZE)))
        idx += 1
    cap.release()

    with st.container(border=True):
        st.markdown(f"**Sampled and resized frames** — {total_frames} frames, {NUM_FRAMES} sampled to {FRAME_SIZE}x{FRAME_SIZE}")
        cols = st.columns(8)
        for i, frame in enumerate(raw_frames[:8]):
            cols[i].image(frame, width="stretch")

    with st.container(border=True):
        st.markdown("**Motion (optical flow)** — hue encodes direction, brightness encodes amount of motion")
        flow_frames = raw_frames[:6]
        if len(flow_frames) >= 2:
            motion_energies = compute_clip_motion_energy(np.stack(flow_frames))
            flow_cols = st.columns(len(flow_frames) - 1)
            for i in range(len(flow_frames) - 1):
                flow = compute_optical_flow(flow_frames[i], flow_frames[i + 1])
                flow_cols[i].image(flow_to_color(flow), width="stretch", caption=f"{motion_energies[i]:.2f}")
else:
    st.info("Preprocessing preview needs the trained model and dataset access to be available.")
