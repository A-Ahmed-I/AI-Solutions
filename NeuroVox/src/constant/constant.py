from pathlib import Path


# ======================================================
# Dataset & Paths Configuration
# ======================================================
base_dir = Path(__file__).resolve().parent.parent.parent
model_path = base_dir / "checkpoint" / "best_model.onnx"
base_path = base_dir / "data"
checkpoint_path = base_dir / "checkpoint" / "best_model.pth"
onnx_path = model_path


# ======================================================
# STFT & Mel Spectrogram Parameters
# ======================================================
n_mel = 40
n_fft = 1024
duration = 6
sample_rate = 22050
hop_length = n_fft // 4

# ======================================================
# ONNX Runtime Providers
# ======================================================
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]


# ======================================================
# Training Config
# ======================================================
epochs = 10
batch_size = 64

train_ratio = 0.7
test_ratio = 0.2


# ======================================================
# Audio Preprocessing
# ======================================================
sample_rate = 22050
min_duration = 1
chunk_duration = 6
overlap_ratio = 0.1
energy_threshold = 1e-6
variance_threshold = 1e-6
silence_db_threshold = -40.0
random_seed = 42
n_fft = 1024
n_mels = 40


# ======================================================
# Labels
# ======================================================
labels = {"HC": 0, "PD": 1}


# ======================================================
# Optimizer
# ======================================================
lr = 0.0001
weight_decay = 1e-2
