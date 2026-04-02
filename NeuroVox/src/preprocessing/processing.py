import os
import numpy as np
import librosa
from tqdm import tqdm
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Union, List, Tuple


class PreProcessing:
    """
    Audio preprocessing pipeline for feature extraction and augmentation.

    This class handles:
        - Audio loading
        - Padding and chunking
        - Audio augmentation (time-stretch, pitch-shift, noise)
        - Mel-spectrogram feature extraction
        - Quality validation of audio chunks and spectrograms

    Attributes:
        sample_rate (int): Target sampling rate for audio files.
        min_sample (int): Minimum number of samples for a valid audio chunk.
        chunk_len (int): Number of samples per chunk.
        hop_length_chunks (int): Step size between chunks.
        energy_threshold (float): Minimum energy threshold for a chunk.
        silence_db_threshold (float): Minimum dB percentile for valid mel-spectrogram.
        variance_threshold (float): Minimum variance for valid mel-spectrogram.
        n_fft (int): FFT window size for mel-spectrogram.
        n_mels (int): Number of mel bins.
        rng (np.random.Generator): Random number generator for augmentations.
    """

    def __init__(
        self,
        sample_rate: int,
        min_duration: float,
        chunk_duration: float,
        overlap_ratio: float,
        energy_threshold: float,
        silence_db_threshold: float,
        variance_threshold: float,
        n_fft: int,
        n_mels: int,
        random_seed: Optional[int] = None,
    ) -> None:
        """
        Initialize the preprocessing pipeline.

        Args:
            sample_rate (int): Target sample rate for audio loading.
            min_duration (float): Minimum duration of audio in seconds.
            chunk_duration (float): Duration of audio chunks in seconds.
            overlap_ratio (float): Overlap ratio between consecutive chunks (0-1).
            energy_threshold (float): Minimum mean squared energy for valid chunks.
            silence_db_threshold (float): Minimum 10th percentile dB value for mel-spectrogram.
            variance_threshold (float): Minimum variance for mel-spectrogram.
            n_fft (int): FFT window size for mel-spectrogram computation.
            n_mels (int): Number of mel-frequency bins.
            random_seed (Optional[int]): Random seed for reproducibility of augmentations.
        """
        self.sample_rate = sample_rate
        self.min_sample = int(min_duration * sample_rate)
        self.chunk_len = int(chunk_duration * sample_rate)
        self.hop_length_chunks = max(1, int(self.chunk_len * (1 - overlap_ratio)))

        self.energy_threshold = energy_threshold
        self.silence_db_threshold = silence_db_threshold
        self.variance_threshold = variance_threshold

        self.n_fft = n_fft
        self.n_mels = n_mels

        self.rng = np.random.default_rng(random_seed)

    @lru_cache()
    def load_audio(self, path: str) -> np.ndarray:
        """
        Load an audio file with caching.

        Args:
            path (str): Path to the audio file.

        Returns:
            np.ndarray: Audio signal as a 1D numpy array.
        """
        audio, _ = librosa.load(path, sr=self.sample_rate, mono=True)
        return audio

    def pad(self, audio_chunk: np.ndarray) -> np.ndarray:
        """
        Pad or truncate an audio chunk to a fixed length.

        Args:
            audio_chunk (np.ndarray): Input audio signal.

        Returns:
            np.ndarray: Padded or truncated audio chunk of length `self.chunk_len`.
        """
        if len(audio_chunk) >= self.chunk_len:
            return audio_chunk[: self.chunk_len]
        return np.pad(audio_chunk, (0, self.chunk_len - len(audio_chunk)))

    def split_into_chunks(self, audio: np.ndarray) -> np.ndarray:
        """
        Split an audio signal into overlapping chunks.

        Args:
            audio (np.ndarray): Input audio signal.

        Returns:
            np.ndarray: Array of audio chunks of shape (num_chunks, chunk_len).
        """
        if len(audio) < self.min_sample:
            return np.array([self.pad(audio)])

        chunks = [
            audio[start : start + self.chunk_len]
            for start in range(
                0, len(audio) - self.chunk_len + 1, self.hop_length_chunks
            )
        ]

        # Handle tail chunk
        tail_start = len(chunks) * self.hop_length_chunks
        if len(audio) > tail_start:
            tail = audio[tail_start:]
            if len(tail) >= self.chunk_len // 2:
                chunks.append(self.pad(tail))

        return np.array(chunks)

    def augment_audio(self, audio: np.ndarray) -> List[np.ndarray]:
        """
        Apply augmentations to an audio chunk.

        Augmentations include:
            - Time-stretching
            - Pitch-shifting
            - Adding Gaussian noise

        Args:
            audio (np.ndarray): Audio chunk to augment.

        Returns:
            List[np.ndarray]: List of augmented audio chunks.
        """
        augmented = [self.pad(audio)]

        # Time-stretch
        if len(audio) > 5000:
            try:
                ts = librosa.effects.time_stretch(audio, rate=1.1)
                augmented.append(self.pad(ts))
            except Exception:
                pass

        # Pitch-shift
        if len(audio) > 100:
            try:
                ps = librosa.effects.pitch_shift(audio, sr=self.sample_rate)
                augmented.append(self.pad(ps))
            except Exception:
                pass

        # Add noise
        noisy = audio + 0.005 * self.rng.standard_normal(len(audio))
        augmented.append(self.pad(noisy))

        return augmented

    def mel(self, audio: np.ndarray, hop_length: int) -> np.ndarray:
        """
        Compute mel-spectrogram from an audio chunk.

        Args:
            audio (np.ndarray): Audio signal.
            hop_length (int): Hop length (number of samples between frames).

        Returns:
            np.ndarray: Mel-spectrogram in dB scale (shape: n_mels x num_frames).
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=hop_length,
            n_mels=self.n_mels,
        )
        return librosa.power_to_db(mel_spec, ref=np.max)

    def valid_chunk(self, chunk: np.ndarray) -> bool:
        """
        Check if an audio chunk meets the energy threshold.

        Args:
            chunk (np.ndarray): Audio chunk.

        Returns:
            bool: True if the mean squared energy >= `self.energy_threshold`.
        """
        return np.mean(chunk**2) >= self.energy_threshold

    def valid_mel(self, mel_spec: np.ndarray) -> bool:
        """
        Check if a mel-spectrogram passes quality criteria.

        Criteria:
            - 10th percentile dB >= `self.silence_db_threshold`
            - Variance >= `self.variance_threshold`

        Args:
            mel_spec (np.ndarray): Mel-spectrogram.

        Returns:
            bool: True if mel-spectrogram is valid.
        """
        if np.percentile(mel_spec, 10) < self.silence_db_threshold:
            return False
        if np.var(mel_spec) < self.variance_threshold:
            return False
        return True

    def get_features(
        self, path: str, hop_length: Optional[int] = None
    ) -> Union[np.ndarray, List]:
        """
        Extract valid mel-spectrogram features from an audio file.

        Args:
            path (str): Path to audio file.
            hop_length (Optional[int]): Hop length override. Defaults to n_fft // 4.

        Returns:
            Union[np.ndarray, List]: Stacked mel-spectrograms if any valid chunks exist,
            otherwise an empty list.
        """
        hop_length = hop_length or self.n_fft // 4

        audio = self.load_audio(path)
        chunks = self.split_into_chunks(audio)
        valid_features = []

        for chunk in chunks:
            if not self.valid_chunk(chunk):
                continue
            for aug in self.augment_audio(chunk):
                mel_spec = self.mel(aug, hop_length)
                if self.valid_mel(mel_spec):
                    valid_features.append(mel_spec.astype(np.float32))

        if not valid_features:
            return []

        return np.stack(valid_features)

    def process_one(self, row: Tuple[str, str]) -> List[Tuple[np.ndarray, str]]:
        """
        Process a single audio file and return labeled features.

        Args:
            row (Tuple[str, str]): Tuple of (audio_path, label).

        Returns:
            List[Tuple[np.ndarray, str]]: List of (mel-spectrogram, label) tuples.
        """
        audio_path, label = row
        features = self.get_features(audio_path)
        return [(f, label) for f in features] if len(features) > 0 else []

    def process_all_data(self, metadata) -> List[Tuple[np.ndarray, str]]:
        """
        Process all audio files in a metadata dataset.

        Args:
            metadata: DataFrame-like object with 'Path' and 'Label' columns and an `iter_rows` method.

        Returns:
            List[Tuple[np.ndarray, str]]: List of all (mel-spectrogram, label) pairs.
        """
        rows = [(row[0], row[1]) for row in metadata.iter_rows()]
        all_data = []

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
            for result in tqdm(
                pool.map(self.process_one, rows), total=len(rows), desc="Processing"
            ):
                all_data.extend(result)

        return all_data
