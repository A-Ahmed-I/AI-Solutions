import cv2
import numpy as np
import polars as pl
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List


class ImagePreprocessor:
    """
    Preprocessing pipeline for Parkinson's handwriting classification.

    Pipeline order:
        1. Resize
        2. Fourier Low-Pass Filter (optional)
        3. Otsu Binarization (optional)
        4. Morphological Operations (optional)

    Notes
    -----
    - Output images are always single-channel (grayscale).
    - Final dtype is uint8.
    """

    def __init__(self, metadata: pl.DataFrame) -> None:
        """
        Parameters
        ----------
        metadata : pl.DataFrame
            DataFrame containing:
                - path (str): image file path
                - type_test (str): test type (e.g., spiral, wave)
                - label (Any): class label
        """
        required_columns = {"path", "type_test", "label"}
        missing_columns = required_columns - set(metadata.columns)

        if missing_columns:
            raise ValueError(f"metadata is missing columns: {missing_columns}")

        self.metadata = metadata

    def _load_and_resize(
        self, image_path: str, target_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Load an image from disk and resize it.

        Parameters
        ----------
        image_path : str
            Path to image file
        target_size : Tuple[int, int]
            Desired output size (width, height)

        Returns
        -------
        np.ndarray
            Resized image (BGR format)

        Raises
        ------
        FileNotFoundError
            If file does not exist
        OSError
            If image cannot be read
        """
        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(str(path))

        if image is None:
            raise OSError(
                f"cv2.imread returned None for '{image_path}'. "
                "File may be corrupted or unsupported."
            )

        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    def _apply_fourier_lpf(
        self, image: np.ndarray, cutoff_radius: int = 30
    ) -> np.ndarray:
        """
        Apply an ideal circular low-pass filter in the Fourier domain.

        Parameters
        ----------
        image : np.ndarray
            Input image (BGR or grayscale)
        cutoff_radius : int
            Radius of the low-pass filter

        Returns
        -------
        np.ndarray
            Filtered grayscale image (uint8)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

        rows, cols = gray.shape
        center_row, center_col = rows // 2, cols // 2

        fft_shifted = np.fft.fftshift(np.fft.fft2(gray))

        y, x = np.ogrid[:rows, :cols]
        mask = (
            (x - center_col) ** 2 + (y - center_row) ** 2 <= cutoff_radius**2
        ).astype(np.uint8)

        filtered = np.fft.ifft2(np.fft.ifftshift(fft_shifted * mask))
        filtered = np.abs(filtered)

        filtered = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX)

        return filtered.astype(np.uint8)

    def _apply_otsu(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Otsu thresholding to convert image to binary.

        Parameters
        ----------
        image : np.ndarray
            Input grayscale or BGR image

        Returns
        -------
        np.ndarray
            Binary image (uint8)
        """
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return binary

    def _apply_morphology(
        self, image: np.ndarray, kernel_size: int = 3, iterations: int = 1
    ) -> np.ndarray:
        """
        Apply morphological opening followed by closing.

        Purpose:
            - Remove noise (opening)
            - Fill small gaps (closing)

        Parameters
        ----------
        image : np.ndarray
            Input image (binary or grayscale)
        kernel_size : int
            Size of structuring element
        iterations : int
            Number of iterations

        Returns
        -------
        np.ndarray
            Processed image
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )

        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)
        closed = cv2.morphologyEx(
            opened, cv2.MORPH_CLOSE, kernel, iterations=iterations
        )

        return closed

    def _process_single_image(
        self,
        image_path: str,
        target_size: Tuple[int, int],
        use_lpf: bool,
        lpf_cutoff: int,
        use_otsu: bool,
        use_morph: bool,
        morph_kernel: int,
        morph_iterations: int,
    ) -> Optional[np.ndarray]:
        """
        Apply full preprocessing pipeline to a single image.

        Returns None if image fails to load.
        """
        try:
            image = self._load_and_resize(image_path, target_size)

            if use_lpf:
                image = self._apply_fourier_lpf(image, cutoff_radius=lpf_cutoff)
            elif image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            if use_otsu:
                image = self._apply_otsu(image)

            if use_morph:
                image = self._apply_morphology(
                    image, kernel_size=morph_kernel, iterations=morph_iterations
                )

            return image

        except (FileNotFoundError, OSError):
            return None

    def process_all(
        self,
        target_size: Tuple[int, int],
        *,
        use_lpf: bool = True,
        lpf_cutoff: int = 30,
        use_otsu: bool = True,
        use_morph: bool = True,
        morph_kernel: int = 3,
        morph_iterations: int = 1,
    ) -> Dict[str, Any]:
        """
        Process all images in metadata.

        Parameters
        ----------
        target_size : Tuple[int, int]
            Resize dimensions (width, height)

        Returns
        -------
        Dict[str, Any]
            {
                "img": np.ndarray (N, H, W),
                "type_test": List[str],
                "label": List[Any]
            }

        Raises
        ------
        RuntimeError
            If no images were successfully processed
        """
        processed_images: List[np.ndarray] = []
        test_types: List[str] = []
        labels: List[Any] = []
        skipped_paths: List[str] = []

        for row in self.metadata.iter_rows(named=True):
            image = self._process_single_image(
                row["path"],
                target_size,
                use_lpf=use_lpf,
                lpf_cutoff=lpf_cutoff,
                use_otsu=use_otsu,
                use_morph=use_morph,
                morph_kernel=morph_kernel,
                morph_iterations=morph_iterations,
            )

            if image is None:
                skipped_paths.append(row["path"])
                continue

            processed_images.append(image)
            test_types.append(row["type_test"])
            labels.append(row["label"])

        if not processed_images:
            raise RuntimeError("No images were successfully processed.")

        return {
            "img": np.stack(processed_images),
            "type_test": test_types,
            "label": labels,
        }
