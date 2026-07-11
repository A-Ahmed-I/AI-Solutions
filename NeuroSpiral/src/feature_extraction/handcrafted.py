import cv2
import numpy as np
import polars as pl
from tqdm.auto import tqdm
from typing import List, Dict
from src.constant.constant import *
from skimage.feature import hog, local_binary_pattern


class FeatureExtractor:
    """
    Extract handcrafted features from images using:
        - HOG (Histogram of Oriented Gradients)
        - LBP (Local Binary Pattern)

    Output:
        Combined feature vector per image.
    """

    def __init__(self, dataframe: pl.DataFrame) -> None:
        """
        Parameters
        ----------
        dataframe : pl.DataFrame
            Must contain column "img"
        """
        self.dataframe = dataframe

        self.radius: int = 1
        self.num_points: int = 8 * self.radius

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extract HOG + LBP features from a single image.

        Parameters
        ----------
        image : np.ndarray
            Grayscale image (H, W)

        Returns
        -------
        np.ndarray
            Concatenated feature vector
        """
        hog_features = hog(
            image,
            orientations=HOG_ORIENTATIONS,
            pixels_per_cell=HOG_PIXELS_PER_CELL,
            cells_per_block=HOG_CELLS_PER_BLOCK,
            feature_vector=True,
        )

        lbp_map = local_binary_pattern(
            image, LBP_N_POINTS, LBP_RADIUS, method="uniform"
        )
        lbp_hist, _ = np.histogram(
            lbp_map.ravel(), bins=LBP_HIST_BINS, range=LBP_HIST_RANGE
        )
        lbp_hist = lbp_hist.astype("float32")
        lbp_hist /= lbp_hist.sum() + 1e-6

        return np.concatenate([hog_features, lbp_hist])

    def extract_all_features(self) -> pl.DataFrame:
        """
        Extract features for all images in the DataFrame.

        Returns
        -------
        pl.DataFrame
            Updated DataFrame with new column "math_features"
        """
        feature_storage: Dict[str, List[np.ndarray]] = {"math_features": []}

        for row in tqdm(self.dataframe.iter_rows(named=True)):
            image = np.array(row["img"], dtype=np.uint8)

            if image.ndim == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            features = self.extract_features(image)
            feature_storage["math_features"].append(features)

        self.dataframe = self.dataframe.with_columns(
            pl.Series("math_features", feature_storage["math_features"])
        )

        return self.dataframe
