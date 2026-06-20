import cv2
import numpy as np
import polars as pl
from typing import Dict, Any, Tuple
from src.constant.constant import IMAGE_SIZE


class ImagePreprocessor:
    """
    Reads raw images from disk, resizes them, and returns a structured dict
    ready for downstream augmentation and feature extraction.

    Processing steps per image
    --------------------------
    1. Read with ``cv2.imread`` (BGR, uint8).
    2. Resize to ``IMAGE_SIZE``.

    Output dictionary
    -----------------
    ::

        {
            "img":       np.ndarray  shape (N, H, W, 3)  uint8,
            "type_test": list[str],
            "label":     list[str],
        }
    """

    def __init__(self, metadata: pl.DataFrame) -> None:
        """
        Parameters
        ----------
        metadata : pl.DataFrame
            DataFrame with columns ``path``, ``test_type``, ``label``.
        """
        self.metadata = metadata
        self._buffer: Dict[str, Any] = {"img": [], "type_test": [], "label": []}

    # ------------------------------------------------------------------
    def _load_and_resize(
        self, image_path: str, target_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Load a single image from disk and resize it.

        Parameters
        ----------
        image_path : str
            Absolute path to the PNG file.
        target_size : tuple[int, int]
            ``(width, height)`` for ``cv2.resize``.

        Returns
        -------
        np.ndarray
            BGR image of shape ``(H, W, 3)``, dtype ``uint8``.
        """
        img = cv2.imread(image_path)
        return cv2.resize(img, target_size)

    # ------------------------------------------------------------------
    def process_all(self, target_size: Tuple[int, int] = IMAGE_SIZE) -> Dict[str, Any]:
        """
        Process every image listed in ``self.metadata``.

        Parameters
        ----------
        target_size : tuple[int, int]
            ``(width, height)`` passed to ``cv2.resize``.  Defaults to
            the global ``IMAGE_SIZE`` constant.

        Returns
        -------
        dict
            Keys: ``img`` (ndarray), ``type_test`` (list), ``label`` (list).
        """
        for row in self.metadata.iter_rows(named=True):
            processed = self._load_and_resize(row["path"], target_size)
            self._buffer["img"].append(processed)
            self._buffer["type_test"].append(row["test_type"])
            self._buffer["label"].append(row["label"])

        self._buffer["img"] = np.array(self._buffer["img"])
        return self._buffer
