import torch
import numpy as np
import polars as pl
from typing import Tuple
from torch.utils.data import Dataset


class ParkinsonDataset(Dataset):
    """
    PyTorch ``Dataset`` for Parkinson's spiral / wave drawing data.

    Each sample contains:
    * normalised RGB image tensor  shape ``(3, H, W)``  float32 ∈ [0, 1]
    * handcrafted feature vector   shape ``(MATH_FEATURE_DIM,)`` float32
    * binary label                 scalar float32  (1.0 = HC, 0.0 = PD)
    """

    def __init__(self, data: pl.DataFrame) -> None:
        """
        Parameters
        ----------
        data : pl.DataFrame
            Must contain columns ``img``, ``math_features``, ``label``.
        """
        self.data = data

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.data)

    # ------------------------------------------------------------------
    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        index : int

        Returns
        -------
        img_tensor      : torch.Tensor  shape (3, H, W)  float32
        feature_tensor  : torch.Tensor  shape (D,)       float32
        label_tensor    : torch.Tensor  scalar            float32
        """
        img = np.array(self.data["img"][index], dtype=np.float32) / 255.0
        label = self.data["label"][index]
        features = np.array(self.data["math_features"][index], dtype=np.float32)

        label_value: float = 0.0 if label == "PD" else 1.0

        img_tensor = torch.from_numpy(img).permute(2, 0, 1)  # (C, H, W)
        feature_tensor = torch.from_numpy(features)
        label_tensor = torch.tensor(label_value, dtype=torch.float32)

        return img_tensor, feature_tensor, label_tensor
