import cv2
import numpy as np
from typing import Tuple
import onnxruntime as ort
from src.constant.constant import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from skimage.feature import hog, local_binary_pattern


class NeuroSpiralPredictor:
    def __init__(
        self,
        model_path: str,
        variance_selector: VarianceThreshold,
        scaler: StandardScaler,
        pca: PCA,
    ) -> None:
        try:
            self.session = ort.InferenceSession(model_path, providers=PROVIDERS)
            self.input_names = [i.name for i in self.session.get_inputs()]
            self.output_name = self.session.get_outputs()[0].name
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

        self.variance_selector = variance_selector
        self.scaler = scaler
        self.pca = pca

    def _extract_features(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
        img_resized = cv2.resize(gray, IMAGE_SIZE)

        hog_features = hog(
            img_resized,
            orientations=HOG_ORIENTATIONS,
            pixels_per_cell=HOG_PIXELS_PER_CELL,
            cells_per_block=HOG_CELLS_PER_BLOCK,
            feature_vector=True,
        )

        lbp = local_binary_pattern(
            img_resized, LBP_N_POINTS, LBP_RADIUS, method="uniform"
        )
        lbp_hist, _ = np.histogram(
            lbp.ravel(), bins=LBP_HIST_BINS, range=LBP_HIST_RANGE
        )
        lbp_hist = lbp_hist.astype("float32")
        lbp_hist /= lbp_hist.sum() + 1e-6

        raw_features = np.concatenate([hog_features, lbp_hist]).reshape(1, -1)

        reduced = self.variance_selector.transform(raw_features)
        reduced = self.scaler.transform(reduced)
        reduced = self.pca.transform(reduced)

        return reduced.astype(np.float32)

    def predict(self, img: np.ndarray) -> Tuple[str, float]:
        img_resized = cv2.resize(img, IMAGE_SIZE)

        if img_resized.ndim == 2:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)

        img_tensor = (img_resized.transpose(2, 0, 1)[np.newaxis] / 255.0).astype(
            np.float32
        )

        features = self._extract_features(img)

        logit = float(
            self.session.run(
                [self.output_name],
                {
                    self.input_names[0]: img_tensor,
                    self.input_names[1]: features,
                },
            )[0].item()
        )

        probability = 1 / (1 + np.exp(-logit))
        label = "PD" if probability < 0.5 else "HC"

        return label, round(probability, 4)
