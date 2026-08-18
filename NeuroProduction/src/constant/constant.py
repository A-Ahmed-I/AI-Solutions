from pathlib import Path

# ======================================================
# Dataset & Paths Configuration
# ======================================================
base_dir = Path(__file__).resolve().parent.parent.parent
voice_model_path = base_dir / "checkpoint" / "voice_best_model.onnx"
spiral_model_path = base_dir / "checkpoint" / "spiral_best_model.onnx"
reducers_path = base_dir / "checkpoint" / "feature_reducers.pkl"
TMP_DIR = "tmp"

# Image
image_size = (224, 224)

# HOG

hog_orientations = 9
pixels_per_cell = (16, 16)
cells_per_block = (2, 2)

lbp_radius = 1
lbp_n_points = 8 * lbp_radius
lbp_hist_bins = 10
lbp_hist_range = (0, 10)

# Audio
n_mel = 40
n_fft = 1024
duration = 6
sample_rate = 22050
hop_length = n_fft // 4

# ONNX
PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
providers = PROVIDERS
