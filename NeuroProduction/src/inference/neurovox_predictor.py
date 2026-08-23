import librosa
import numpy as np
import onnxruntime as ort
from src.constant.constant import *


class NeuroVoxPredictor:
    """
    NeuroVox ONNX inference class for Parkinson's Disease detection
    from speech audio using Mel-Spectrogram features.
    """

    def __init__(self, model_path: str):
        """
        Initialize ONNX Runtime session and cache model metadata.

        Args:
            model_path (str): Path to ONNX model file.
        """
        try:
            self.session = ort.InferenceSession(model_path, providers=providers)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.target_length = duration * sample_rate

    def preprocess(self, audio_path: str) -> np.ndarray | None:
        """
        Load audio file, normalize its length, and convert it to
        a Mel-Spectrogram tensor suitable for CNN inference.

        Args:
                audio_path (str): Path to input WAV file.

        Returns:
                np.ndarray | None:
                        - Shape (1, 1, n_mels, time_steps) if successful
                        - None if audio loading fails
        """
        try:
            audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
        except Exception as e:
            print(f"Error loading audio: {e}")
            return None

        if len(audio) < self.target_length:
            audio = librosa.util.fix_length(audio, size=self.target_length)
        else:
            audio = audio[: self.target_length]

        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mel
        )

        mel_db = librosa.power_to_db(mel_spec, ref=np.max)

        spec_tensor = mel_db[np.newaxis, np.newaxis, :, :]

        return spec_tensor.astype(np.float32)

    def predict(self, audio_path: str) -> tuple[str, float]:
        """
        Run inference on a single audio file and return prediction.

        Args:
                audio_path (str): Path to input WAV file.

        Returns:
                tuple[str, float]:
                        - Predicted label ("PD" or "HC")
                        - Probability score rounded to 4 decimals
        """
        input_tensor = self.preprocess(audio_path)

        if input_tensor is None:
            return "Error", 0.0

        logits = self.session.run([self.output_name], {self.input_name: input_tensor})[0]

        logit_value = logits.item()

        probability = 1 / (1 + np.exp(-logit_value))

        label = "PD" if probability > 0.5 else "HC"

        return label, round(probability, 4)
