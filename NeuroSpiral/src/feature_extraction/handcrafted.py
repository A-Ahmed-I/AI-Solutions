import cv2
import numpy as np
import polars as pl
from typing import List
from tqdm.auto import tqdm
from src.constant.constant import *
from skimage.feature import hog, local_binary_pattern


class HandcraftedFeatureExtractor:
    """
    Extracts HOG and LBP features from BGR images and appends them to the
    input DataFrame as a new ``math_features`` column.

    Feature vector layout
    ---------------------
    ``[hog_features | lbp_histogram]``

    LBP histogram is L1-normalised to sum ≈ 1.
    """

    def __init__(self, data: pl.DataFrame) -> None:
        """
        Parameters
        ----------
        data : pl.DataFrame
            DataFrame that must contain an ``img`` column (BGR uint8 arrays).
        """
        self.data = data

    # ------------------------------------------------------------------
    def _extract_single(self, bgr_image: np.ndarray) -> np.ndarray:
        """
        Extract HOG + LBP features from one BGR image.

        Parameters
        ----------
        bgr_image : np.ndarray
            BGR uint8 image of shape ``(H, W, 3)``.

        Returns
        -------
        np.ndarray
            1-D float32 feature vector of length ``MATH_FEATURE_DIM``.
        """
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

        # HOG
        hog_vec = hog(
            gray,
            orientations=HOG_ORIENTATIONS,
            pixels_per_cell=HOG_PIXELS_PER_CELL,
            cells_per_block=HOG_CELLS_PER_BLOCK,
            feature_vector=True,
        )

        # LBP
        lbp_map = local_binary_pattern(gray, LBP_N_POINTS, LBP_RADIUS, method="uniform")
        lbp_hist, _ = np.histogram(
            lbp_map.ravel(), bins=LBP_HIST_BINS, range=LBP_HIST_RANGE
        )
        lbp_hist = lbp_hist.astype("float32")
        lbp_hist /= lbp_hist.sum() + 1e-6  # L1 normalisation

        return np.concatenate([hog_vec, lbp_hist])

    # ------------------------------------------------------------------
    def extract_all(self) -> pl.DataFrame:
        """
        Extract features for every row and return the enriched DataFrame.

        Returns
        -------
        pl.DataFrame
            Original DataFrame with an additional ``math_features`` column
            (each cell is a 1-D float32 ndarray).
        """
        feature_list: List[np.ndarray] = []

        for row in tqdm(self.data.iter_rows(named=True), desc="Extracting features"):
            img = np.array(row["img"], dtype=np.uint8)
            feature = self._extract_single(img)
            feature_list.append(feature)

        return self.data.with_columns(pl.Series("math_features", feature_list))
