import librosa
import numpy as np
import polars as pl
import soundfile as sf
from pathlib import Path
from typing import Tuple, List


class CreateMetadata:
    """
    Audio metadata creation and validation class.

    This class handles the organization, validation, and labeling of audio files
    from a directory structure into a structured dataset. It ensures that only
    usable audio files (based on duration, integrity, and content) are included.

    Attributes:
        base_path (Path): Base directory containing labeled subdirectories of audio files.
        min_duration (float): Minimum duration (in seconds) for audio files to be considered valid.
    """

    def __init__(self, base_path: str | Path, min_duration: float) -> None:
        """
        Initialize the metadata creator with a base directory and minimum audio duration.

        Args:
            base_path (str | Path): Path to the base directory containing subdirectories with audio files.
            min_duration (float): Minimum acceptable duration (in seconds) for audio files.
        """
        self.base_path = Path(base_path)
        self.min_duration = min_duration

    def _validate_audio_file(self, path: str | Path) -> bool:
        """
        Validate whether an audio file is suitable for processing.

        Performs checks for file integrity, minimum duration, non-silence content,
        and numerical validity using `soundfile` and `librosa`.

        Args:
            path (str | Path): Path to the audio file to validate.

        Returns:
            bool: True if the audio file passes all validation checks, False otherwise.
        """
        audio_signal = None
        sample_rate = None

        try:
            audio_signal, sample_rate = sf.read(path)
            audio_signal = np.array(audio_signal, dtype=float).flatten()
        except Exception:
            pass

        if audio_signal is None or len(audio_signal) == 0:
            try:
                audio_signal, sample_rate = librosa.load(path, sr=None, mono=True)
            except Exception:
                return False

        if audio_signal is None or len(audio_signal) == 0:
            return False

        if np.max(np.abs(audio_signal)) < 1e-6:
            return False

        duration = len(audio_signal) / sample_rate
        if duration < self.min_duration:
            return False

        if not np.isfinite(audio_signal).all():
            return False

        return True

    def _process_directory(
        self, directory: Path, label: str
    ) -> Tuple[list[tuple[str, str]], int]:
        """
        Process all audio files in a directory and assign a label to each file.

        Args:
            directory (Path): Path to the directory containing audio files.
            label (str): Label to assign to all audio files in this directory.

        Returns:
            Tuple[list[tuple[str, str]], int]:
                - List of (filepath, label) tuples for valid audio files.
                - Count of audio files that failed validation.
        """
        records: List[Tuple[str, str]] = []
        failed_count = 0

        files = [f for f in directory.iterdir() if f.is_file()]

        for file_path in files:
            if self._validate_audio_file(file_path):
                records.append((str(file_path), label))

            else:
                failed_count += 1

        return records, failed_count

    def load_metadata(self) -> pl.DataFrame:
        """
        Load and validate audio metadata from a structured directory.

        Assumes the base directory contains exactly two subdirectories:
        - Parkinson's Disease (PD)
        - Healthy Control (HC)

        Each subdirectory is processed to validate audio files and assign labels.
        Returns a Polars DataFrame with file paths and corresponding labels.

        Returns:
            pl.DataFrame: DataFrame containing two columns:
                - "Path" (str): Path to the validated audio file.
                - "Label" (str): Label assigned to the file ("PD" or "HC").
        """
        subdirectories = [d for d in self.base_path.iterdir() if d.is_dir()]

        parkinsons_dir = subdirectories[0]
        healthy_control_dir = subdirectories[1]

        print(f"Processing PD directory: {parkinsons_dir.name}")
        parkinsons_records, parkinsons_failed = self._process_directory(
            parkinsons_dir, "PD"
        )

        print(f"Processing HC directory: {healthy_control_dir.name}")
        healthy_control_records, healthy_control_failed = self._process_directory(
            healthy_control_dir, "HC"
        )

        all_records = parkinsons_records + healthy_control_records
        total_failed = parkinsons_failed + healthy_control_failed

        data = pl.DataFrame(all_records, schema=["Path", "Label"], orient="row")

        print(f"\nSummary:")
        print(f"  Valid files: {len(all_records)}")
        print(f"  Failed files: {total_failed}")
        print(f"  PD samples: {len(parkinsons_records)}")
        print(f"  HC samples: {len(healthy_control_records)}")

        return data
