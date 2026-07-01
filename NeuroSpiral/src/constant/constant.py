from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATASET_ROOT = BASE_DIR / "data"
REDUCERS_PATH = BASE_DIR / "checkpoint" / "feature_reducers.pkl"
ONNX_EXPORT_PATH = BASE_DIR / "checkpoint" / "spiral_best_model.onnx"
CHECKPOINT_PATH = BASE_DIR / "checkpoint" / "spiral_best_model.pth"

# ── Image ──────────────────────────────────────────────────────────────────
IMAGE_SIZE = (224, 224)  # (width, height) for cv2.resize

# ── Split ──────────────────────────────────────────────────────────────────
TEST_SPLIT_RATIO = 0.2
RANDOM_SEED = 49

# ── HOG feature extraction ─────────────────────────────────────────────────
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)

# ── LBP feature extraction ─────────────────────────────────────────────────
LBP_RADIUS = 1
LBP_N_POINTS = 8 * LBP_RADIUS  # 8
LBP_HIST_BINS = 10
LBP_HIST_RANGE = (0, 10)

# ── Data augmentation ──────────────────────────────────────────────────────
NUM_AUGMENTATIONS_PER_IMAGE = 5

# ── DataLoaders ────────────────────────────────────────────────────────────
TRAIN_BATCH_SIZE = 16
VAL_BATCH_SIZE = 8
TEST_BATCH_SIZE = 4

# ── Optimiser / Scheduler ──────────────────────────────────────────────────
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
WARMUP_START_FACTOR = 0.01
WARMUP_END_FACTOR = 1.0
WARMUP_TOTAL_ITERS = 5
COSINE_T_MAX = 10
SCHEDULER_MILESTONES = [5]  # epoch at which warmup ends → cosine begins

# ── Training ───────────────────────────────────────────────────────────────
NUM_EPOCHS = 15


# ── ONNX Runtime Providers ────────────────────────────────────────────────────
PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
