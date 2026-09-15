# Action Recognition in Videos — UCF101 (Full 101-Class)

A deep-learning system that classifies short video clips into human
action categories — all 101 UCF101 classes — using a fine-tuned 3D CNN,
with a step-by-step frontend that shows exactly how each prediction is
made, and documented precautions at every stage of the pipeline.

## Project Description

Given a short video clip, the model predicts which of 101 known human
actions is being performed (walking a dog, jump rope, basketball,
archery, diving, and so on). This covers the full pipeline for
real-world video understanding: leak-safe dataset construction,
imbalance-aware training with early stopping, standard classification
evaluation (top-1/top-5 accuracy, confusion matrix, per-class
precision/recall/F1, inference latency), and a frontend that visualizes
every stage rather than only exposing a black-box prediction.

The domain is sports and everyday-movement analytics — the same space as
automated performance feedback, surveillance/behavior analysis, and
video indexing described in the project brief.

## Precautions & Design Rationale

Every non-obvious choice below exists because of a specific, real
problem encountered while building this — not as a generic checklist.

| Precaution | Why |
|---|---|
| **Group-level train/test split** (`data/download_ucf101.py`) | UCF101 clips come in groups sharing the same actor/background. An early per-clip random split leaked scene information between train and test and produced a suspicious 100% accuracy. Splitting at the group level fixed this — verified by re-running and seeing a realistic, imperfect accuracy with sensible confusions instead. |
| **Class-weighted loss** (`src/train.py`) | UCF101 classes vary from ~70–160 clips each. Weights are computed from the actual on-disk train counts each run (`compute_class_weights`), not an assumed distribution, so rare classes aren't drowned out by common ones. |
| **Early stopping** (`src/train.py`) | Training halts once validation loss stops improving for `EARLY_STOPPING_PATIENCE` epochs, instead of always running the full epoch budget. This matters concretely on free-tier Colab/Kaggle GPU quotas — wasted epochs are wasted quota. |
| **LR warmup + cosine decay** | A flat cosine schedule from step 1 can destabilize the newly-initialized classifier head, which starts with large gradients. A short linear warmup avoids this. |
| **Gradient clipping** | Guards against gradient spikes, which become more likely with more classes and more data at full 101-class scale. |
| **Mixed precision (`torch.cuda.amp`)** | Keeps the larger 101-class model and batch memory footprint inside a single ≤16GB GPU. |
| **Fixed random seeds everywhere** (`src/utils.py`) | Python, NumPy, PyTorch, and CUDA seeds are all fixed so results are reproducible, not lucky. |
| **`pretrained=False` for inference-only code paths** (`src/model.py`) | Evaluation/visualization/frontend scripts load a full trained checkpoint anyway, so downloading Kinetics-400 pretrained weights first is wasted time and an unnecessary internet dependency. Found and fixed while testing the frontend. |
| **Corrupted/unreadable clip fallback** (`src/dataset.py`) | Returns a black placeholder clip instead of crashing an entire training batch on one bad file. |
| **Bounded hyperparameter search, explicitly labeled as such** (`src/hyperparam_search.py`) | A full exhaustive grid search needs far more compute than a free-tier GPU budget allows. Rather than either skipping tuning or dishonestly claiming an exhaustive search, this runs a small, fixed probe (few epochs, few configs) and is explicit in its own output and code comments that it is not exhaustive. |
| **Graceful frontend failure handling** (`app/streamlit_app.py`) | Every block checks whether its required file exists before rendering, and model loading is wrapped in a try/except — a missing checkpoint or dataset shows an informative message instead of crashing the whole dashboard. |
| **`mediapipe` version handling** (`src/visualize_pose_predictions.py`) | MediaPipe removed its legacy `solutions.pose` API in newer releases. This project uses the current Tasks API deliberately, with a fresh `PoseLandmarker` instance created per clip (the API requires monotonically increasing timestamps *per instance*, which broke when one instance was reused across clips — found and fixed during testing). |
| **Frame caching preserves augmentation randomness** (`src/dataset.py`, `data/build_frame_cache.py`) | Video decoding is the expensive, repeated cost across epochs. Rather than caching the final augmented tensor (which would freeze flip/crop/normalization to one fixed version and hurt generalization), only the decoded, resized frames are cached, at a denser pool (32 frames) than the model uses (16). Random flip, crop, and temporal subsampling still happen fresh every epoch on top of the cached pool — verified directly: 8 repeated fetches of the same cached clip produced 8 different augmented tensors, not identical ones. Caching is opt-in (`USE_FRAME_CACHE=False` by default) since it costs meaningful extra disk space (~15GB for the full dataset at default settings) that not every environment has room for. |
| **Symlinks instead of file copies for the dataset split** (`data/download_ucf101.py`) | Copying every video into `train/`/`test/` folders at full 101-class scale nearly doubled disk usage on top of the raw download, causing an actual "No space left on device" failure during a real run. Switched to symlinks (falling back to real copies only if the filesystem doesn't support them), cutting peak disk usage roughly in half. |
| **Dropout inserted without breaking checkpoint compatibility** (`src/model.py`) | Regularization is explicitly requested by the project brief. Wrapping the classifier head directly (`fc`) in a `Sequential` would rename its saved parameter keys, breaking every checkpoint trained before dropout was added. Instead, dropout wraps `avgpool` (zero learnable parameters), so checkpoint keys are identical either way — verified directly by loading an old (no-dropout) checkpoint into a new dropout-enabled model with `strict=True`. |
| **Focal Loss is opt-in, not a silent replacement** (`src/train.py`) | Targets classes confused due to visual/motion similarity (e.g. Basketball vs BasketballDunk) rather than rarity — a different failure mode than class imbalance. Verified its down-weighting behavior directly (a confidently-wrong example's relative loss contribution is ~9x more concentrated under focal loss than plain cross-entropy), but it changes training dynamics and hasn't been validated against the already-confirmed 95%+ baseline, so it defaults to off rather than silently changing results. |

## Approach

**Backbone: R(2+1)D-18, pretrained on Kinetics-400, fine-tuned on UCF101.**

1. **Preprocessing** — each video is decoded with OpenCV, 16 frames are
   sampled evenly across its duration (light random temporal jitter
   during training only), resized to 112×112, and normalized using the
   backbone's pretraining statistics. Every step is visualized live in
   the frontend's Step 2.
2. **Backbone** — [`r2plus1d_18`](https://arxiv.org/abs/1711.11248)
   factorizes 3D convolutions into 2D spatial + 1D temporal convolutions,
   keeping compute low enough to fine-tune on a single ≤16GB GPU while
   still capturing motion dynamics, unlike frame-independent 2D CNNs.
3. **Transfer learning strategy** — Kinetics-400 pretrained weights,
   most layers frozen, only the last residual block + a new classifier
   head fine-tuned. Dramatically reduces the data/time needed to reach
   good accuracy compared to training from scratch.
4. **Class scope** — all 101 UCF101 classes by default
   (`USE_ALL_CLASSES = True` in `src/config.py`). Auto-discovered from
   the downloaded dataset and written to `data/UCF101_subset/classes.txt`,
   so every script (train/evaluate/visualize/frontend) always agrees on
   the actual class list without manual editing. Set `USE_ALL_CLASSES =
   False` to fall back to a smaller fixed subset for fast local iteration.
5. **Training** — AdamW, warmup+cosine LR schedule, class-weighted
   cross-entropy loss, gradient clipping, mixed precision, early
   stopping. Every epoch's metrics are logged to
   `outputs/training_history.json`.
6. **Hyperparameter tuning** — `src/hyperparam_search.py` runs a bounded
   probe over a small grid of learning rates and backbone-freeze depths,
   picking a starting configuration for the full run. See the
   Precautions table above for why this is bounded, not exhaustive.
7. **Evaluation** — top-1/top-5 accuracy, confusion matrix, per-class
   precision/recall/F1, per-clip inference time, and automated
   most-confused-class-pair analysis (`outputs/confused_pairs.json`) —
   directly answering the brief's "confusion matrix analysis to identify
   class-wise performance and common misclassifications."
8. **Motion visualization** — `src/motion_visualization.py` computes
   dense optical flow between sampled frames, visualized in the
   frontend's Step 2e. This answers the brief's "incorporate optical
   flow or motion-based features" stretch goal as an analysis layer
   rather than a second input stream — see that file's docstring for why
   a full Two-Stream architecture wasn't attempted (real risk to the
   already-validated pipeline for an unvalidated accuracy gain).
9. **Real-time inference simulation** — the frontend's Step 7 shows how
   prediction confidence evolves as frames arrive progressively, rather
   than only a single final answer — answering the brief's "implement
   real-time inference capabilities" stretch goal.
10. **Comprehensive report generation** — `src/generate_report.py`
    compiles training history, hyperparameter probe results, evaluation
    metrics, and confusion analysis into a single `outputs/REPORT.md`,
    plus a dedicated "Challenges Encountered & Potential Improvements"
    section — both explicitly named as required deliverables in the
    project brief, separate from this README.
11. **Frontend** — a Streamlit app presenting the entire pipeline as
    seven numbered, inspectable blocks (see below).

## Repository Structure

```
.
├── data/
│   ├── download_ucf101.py         # downloads UCF101, builds leak-safe class split + distribution
│   └── build_frame_cache.py       # optional: decode-once frame cache for faster training
├── src/
│   ├── config.py                   # classes, paths, hyperparameters, precaution flags
│   ├── dataset.py                  # video loading, frame sampling, augmentation, caching
│   ├── model.py                    # R(2+1)D-18 backbone + classifier head + dropout
│   ├── train.py                    # training loop: class weights/focal loss, early stopping, warmup, clipping
│   ├── hyperparam_search.py        # bounded hyperparameter probe (see Precautions)
│   ├── evaluate.py                 # accuracy / confusion matrix / confused-pairs / report / timing
│   ├── generate_report.py          # compiles all artifacts into outputs/REPORT.md
│   ├── motion_visualization.py     # optical flow analysis (motion-dynamics stretch goal)
│   ├── visualize_predictions.py    # static prediction grid on sample clips
│   ├── visualize_pose_predictions.py  # pose-skeleton-overlay annotated videos
│   └── utils.py                    # seeding, device helpers, environment logging
├── app/
│   └── streamlit_app.py            # step-by-step pipeline frontend (7 blocks)
├── notebooks/
│   ├── Action_Recognition_Colab.ipynb
│   └── Action_Recognition_Kaggle.ipynb
├── requirements.txt
└── .gitignore
```

## Prerequisites / Dependencies

- Python 3.9+
- A CUDA-capable GPU with ≤16GB memory is strongly recommended for
  full 101-class training (CPU works but is impractically slow at this scale)
- See `requirements.txt`:
  - `torch`, `torchvision` — model + training
  - `opencv-python` — video decoding
  - `numpy`, `scikit-learn` — numeric ops, evaluation metrics
  - `matplotlib`, `seaborn` — evaluation plots
  - `mediapipe` — pose skeleton overlays (uses the current Tasks API, not the removed legacy `solutions` API)
  - `tqdm` — progress bars
  - `huggingface_hub` — dataset download
  - `streamlit`, `plotly`, `pandas` — the frontend

## Setup

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

**1. Download the full dataset:**
```bash
python data/download_ucf101.py
```
Auto-discovers and downloads all 101 classes (since `USE_ALL_CLASSES =
True`), builds a group-safe train/test split, and writes
`data/UCF101_subset/classes.txt` + `class_distribution.json`. **This is
a much larger download and disk footprint than a small subset** —
expect this step to take a while.

**2. (Optional) Build the frame cache for faster training:**
```bash
python data/build_frame_cache.py
```
Decodes each clip once and saves the frames to `data/frame_cache/`, so
subsequent epochs read from disk instead of re-decoding video every time.
Reads its own disk-space estimate before running — check you have room
(roughly 15GB for the full dataset at default settings). Set
`USE_FRAME_CACHE = True` in `src/config.py` afterward to actually use it;
training works fine without this step too, just slower per epoch.

**3. (Optional) Run the bounded hyperparameter probe:**
```bash
python src/hyperparam_search.py
```
Trains a few epochs across a small grid of learning rates/freeze depths
and writes `outputs/best_hparams.json`. Update `LEARNING_RATE` in
`src/config.py` based on the result if you want to use it.

**4. Train:**
```bash
python src/train.py
```
Runs with class-weighted loss, LR warmup, gradient clipping, and early
stopping. Saves the best checkpoint to `checkpoints/best_model.pt` and
logs every epoch to `outputs/training_history.json`.

**5. Evaluate:**
```bash
python src/evaluate.py --checkpoint checkpoints/best_model.pt
```
Writes `outputs/confusion_matrix.png`, `outputs/classification_report.txt`,
and `outputs/confused_pairs.json` (the specific class pairs the model
confuses most, ranked — not just the raw matrix image).

**6. Visualize predictions:**
```bash
python src/visualize_predictions.py --checkpoint checkpoints/best_model.pt --num_samples 8
python src/visualize_pose_predictions.py --checkpoint checkpoints/best_model.pt --num_clips 5
```
The second command writes pose-skeleton-overlay annotated videos to
`outputs/pose_visualizations/`.

**7. Generate the comprehensive report:**
```bash
python src/generate_report.py
```
Compiles training history, hyperparameter probe results, evaluation
metrics, and confusion analysis into `outputs/REPORT.md` — the project
brief's explicitly-required "comprehensive report" deliverable, separate
from this README. Re-run any time after training/evaluation to refresh
it with the latest numbers.

**8. Run the frontend:**
```bash
streamlit run app/streamlit_app.py
```
Opens a local dashboard with seven numbered blocks: dataset balance,
live preprocessing demo (including optical-flow motion visualization),
model/training configuration, training curves, evaluation results, live
prediction with a top-5 probability breakdown, and a real-time-inference
simulation. Run this **after** downloading the trained artifacts
(checkpoint, `outputs/`, `data/UCF101_subset/`) from wherever you
trained (Colab/Kaggle), since those platforms don't expose local ports
directly.

**Or run everything in Colab/Kaggle:** open
`notebooks/Action_Recognition_Colab.ipynb` or
`notebooks/Action_Recognition_Kaggle.ipynb`, which install dependencies,
download the data, train, and evaluate end-to-end on a free GPU.

## Results

_Fill in after training on the full 101 classes:_

| Metric | Value |
|---|---|
| Top-1 accuracy | — |
| Top-5 accuracy | — |
| Avg inference time / clip | — ms |
| Epochs run (early stopping) | — |

See `outputs/classification_report.txt` and `outputs/confusion_matrix.png`
for the full per-class breakdown.

## Possible Extensions

- Add optical-flow input as a second stream (Two-Stream Network)
- Try a transformer-based backbone (e.g. TimeSformer, Video Swin) once
  the CNN baseline is stable
- Evaluate on HMDB51 to test cross-dataset generalization
- Expand the hyperparameter probe if more compute becomes available
  (e.g. Colab Pro, a local GPU) — the search space in
  `src/hyperparam_search.py` is intentionally small given the free-tier
  budget this was built under
