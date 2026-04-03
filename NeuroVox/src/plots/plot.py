import random
import librosa
import numpy as np
import matplotlib.pyplot as plt


class AudioVisualizer:
    """Audio visualization class for displaying waveforms and spectrograms.

    Loads random audio samples from a metadata dataset and displays
    their waveforms and mel-spectrograms side by side.
    """

    def __init__(self, metadata):
        """Initialize the audio visualizer.

        Args:
            metadata: A Polars DataFrame containing audio file paths and labels.
        """
        self.metadata = metadata

    def _load_random_audio(self) -> tuple[tuple[float, ...], int, str]:
        """Load a random audio file from the metadata.

        Returns:
            Tuple containing (audio_signal, sample_rate, label).
        """
        random_index = random.randrange(self.metadata.height)

        path = self.metadata["Path"][random_index]
        label = self.metadata["Label"][random_index]

        audio_signal, sample_rate = librosa.load(path)

        return audio_signal, sample_rate, label

    def plot_random_sample(self) -> None:
        """Plot a random audio sample with waveform and mel-spectrogram.

        Displays two subplots: the audio waveform and its mel-spectrogram
        representation for a randomly selected sample from the dataset.
        """
        audio_signal, sample_rate, label = self._load_random_audio()

        plot_title = "Healthy Control" if label == "HC" else "People with Parkinson's"
        plt.figure(figsize=(20, 10))

        # Waveform
        plt.subplot(1, 2, 1)
        librosa.display.waveshow(audio_signal, sr=sample_rate)
        plt.title(f"Waveform - {plot_title}")

        # Mel-Spectrogram
        plt.subplot(1, 2, 2)
        mel_spectrogram = librosa.feature.melspectrogram(y=audio_signal, sr=sample_rate)
        mel_db_scaled = librosa.power_to_db(mel_spectrogram, ref=np.max)
        librosa.display.specshow(
            mel_db_scaled, sr=sample_rate, x_axis="time", y_axis="mel"
        )
        plt.title(f"Mel-Spectrogram - {plot_title}")
        plt.colorbar(format="%+2.0f dB")

        plt.tight_layout()
        plt.show()
