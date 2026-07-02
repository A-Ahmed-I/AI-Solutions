import torch
import numpy as np
import polars as pl
from typing import Tuple
from torch.utils.data import Dataset


class ParkinsonDataset(Dataset):
    """
    PyTorch Dataset for Parkinson's handwriting classification (spiral / wave).

    Each sample consists of:
        - Image tensor: shape (3, H, W), float32 in range [0, 1]
        - Handcrafted feature vector: shape (D,), float32
        - Label: scalar float32
            * 1.0 → Healthy Control (HC)
            * 0.0 → Parkinson's Disease (PD)
    """

    def __init__(self, dataframe: pl.DataFrame) -> None:
        """
        Parameters
        ----------
        dataframe : pl.DataFrame
            Must contain columns:
                - "img": 2D image (H, W) as array-like
                - "math_features": feature vector
                - "label": "PD" or "HC"
        """
        self.dataframe = dataframe

    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.dataframe)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve a single sample.

        Parameters
        ----------
        idx : int
            Sample index

        Returns
        -------
        image_tensor : torch.Tensor
            Shape (3, H, W), float32 normalized to [0, 1]

        feature_tensor : torch.Tensor
            Shape (D,), float32

        label_tensor : torch.Tensor
            Scalar tensor (float32)
        """
        image_array = np.array(self.dataframe["img"][idx], dtype=np.float32) / 255.0
        feature_array = np.array(self.dataframe["math_features"][idx], dtype=np.float32)
        label_value = self.dataframe["label"][idx]

        label_value = 0.0 if label_value == "PD" else 1.0

        image_tensor = torch.from_numpy(image_array)  # (H, W)
        image_tensor = image_tensor.unsqueeze(0).repeat(3, 1, 1)  # (3, H, W)

        feature_tensor = torch.from_numpy(feature_array)  # (D,)
        label_tensor = torch.tensor(label_value, dtype=torch.float32)

        return image_tensor, feature_tensor, label_tensor
