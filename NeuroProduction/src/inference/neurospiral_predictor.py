from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog, local_binary_pattern
from sklearn.decomposition import PCA
from typing import Tuple, List, Dict
from src.constant.constant import *
import onnxruntime as ort
import numpy as np
import cv2


class NeuroSpiralPredictor:
    """
    A predictor class that uses an ONNX model to classify images based on
    pre-extracted HOG and LBP features combined with raw image data.
    """

    def __init__(
        self,
        model_path: str,
        variance_selector: VarianceThreshold,
        scaler: StandardScaler,
        pca: PCA,
    ) -> None:
        """
        Initializes the NeuroSpiralPredictor by loading the ONNX model.

        Args:
            model_path (str): The file path to the ONNX model.
            variance_selector (VarianceThreshold): Fitted variance threshold selector.
            scaler (StandardScaler): Fitted standard scaler.
            pca (PCA): Fitted PCA model.

        Raises:
            RuntimeError: If the model fails to load.
        """
        try:
            self.session: ort.InferenceSession = ort.InferenceSession(
                model_path, providers=PROVIDERS
            )
            self.input_names: List[str] = [i.name for i in self.session.get_inputs()]
            self.output_name: str = self.session.get_outputs()[0].name
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

        self.variance_selector = variance_selector
        self.scaler = scaler
        self.pca = pca

    def _extract_features(self, img: np.ndarray) -> np.ndarray:
        """
        Extracts HOG and LBP features from the input image,
        then applies the fitted reduction pipeline.

        Args:
            img (np.ndarray): The input image in BGR or grayscale format.

        Returns:
            np.ndarray: Reduced feature vector of shape (1, D).
        """
        gray: np.ndarray = (
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
        )

        img_resized: np.ndarray = cv2.resize(gray, IMAGE_SIZE)

        hog_features: np.ndarray = hog(
            img_resized,
            orientations=HOG_ORIENTATIONS,
            pixels_per_cell=HOG_PIXELS_PER_CELL,
            cells_per_block=HOG_CELLS_PER_BLOCK,
            feature_vector=True,
        )

        lbp: np.ndarray = local_binary_pattern(
            img_resized, LBP_N_POINTS, LBP_RADIUS, method="uniform"
        )
        lbp_hist, _ = np.histogram(
            lbp.ravel(), bins=LBP_HIST_BINS, range=LBP_HIST_RANGE
        )
        lbp_hist = lbp_hist.astype("float32")
        lbp_hist /= lbp_hist.sum() + 1e-6

        raw_features: np.ndarray = np.concatenate([hog_features, lbp_hist]).reshape(
            1, -1
        )

        reduced: np.ndarray = self.variance_selector.transform(raw_features)
        reduced = self.scaler.transform(reduced)
        reduced = self.pca.transform(reduced)

        return reduced.astype(np.float32)

    def predict(self, img: np.ndarray) -> Tuple[str, float]:
        """
        Performs inference on the provided image.

        Args:
            img (np.ndarray): The input image.

        Returns:
            Tuple[str, float]: A tuple containing the classification label
                               ('PD' or 'HC') and the calculated probability.
        """
        img_input: np.ndarray = cv2.resize(img, IMAGE_SIZE)

        if img_input.ndim == 2:
            img_input = cv2.cvtColor(img_input, cv2.COLOR_GRAY2BGR)

        img_tensor: np.ndarray = (
            img_input.transpose(2, 0, 1)[np.newaxis, ...] / 255.0
        ).astype(np.float32)

        features: np.ndarray = self._extract_features(img)

        inputs: Dict[str, np.ndarray] = {
            self.input_names[0]: img_tensor,
            self.input_names[1]: features,
        }

        logits: float = float(self.session.run([self.output_name], inputs)[0].item())

        probability: float = 1 / (1 + np.exp(-logits))

        label: str = "PD" if probability < 0.5 else "HC"

        return label, round(probability, 4)
