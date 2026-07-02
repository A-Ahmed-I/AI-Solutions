import numpy as np
import polars as pl
from typing import List
import albumentations as A


class TrainAugmentor:
    """
    Generates augmented training samples using an Albumentations pipeline.

    For each original sample the augmentor keeps the original **and** adds
    ``num_augmentations`` synthetic variants.  The resulting dataset is
    shuffled before being returned.

    Augmentation transforms applied
    --------------------------------
    * Random rotation ±25°
    * Affine (scale, translate, shear)
    * Gaussian noise  (simulates hand tremor)
    * Random brightness / contrast
    * Horizontal flip
    """

    def __init__(self, data: pl.DataFrame, num_augmentations: int) -> None:
        """
        Parameters
        ----------
        data : pl.DataFrame
            DataFrame with columns ``img``, ``type_test``, ``label``.
        num_augmentations : int
            Number of extra augmented copies to create per original image.
        """
        self.data = data
        self.num_augmentations = num_augmentations

    def _build_transform(self) -> A.Compose:
        """
        Build and return the Albumentations augmentation pipeline.

        Returns
        -------
        A.Compose
            Composed transformation pipeline.
        """
        return A.Compose(
            [
                A.Rotate(limit=20, p=0.6),
                A.Affine(
                    scale=(0.95, 1.05),
                    translate_percent=(0.03, 0.03),
                    shear=(-3, 3),
                    p=0.5,
                ),
                A.GaussNoise(var_limit=(3.0, 10.0), p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.08, contrast_limit=0.08, p=0.3
                ),
            ]
        )

    def augment(self) -> pl.DataFrame:
        """
        Apply augmentation and return the expanded, shuffled DataFrame.

        Returns
        -------
        pl.DataFrame
            Columns: ``img``, ``type_test``, ``label``.
            Rows ≈ ``len(data) × (1 + num_augmentations)``, shuffled.
        """
        transform = self._build_transform()

        aug_images: List[np.ndarray] = []
        aug_test_types: List[str] = []
        aug_labels: List[str] = []

        for row in self.data.iter_rows(named=True):
            original = np.array(row["img"], dtype=np.uint8)

            # Keep original
            aug_images.append(original)
            aug_test_types.append(row["type_test"])
            aug_labels.append(row["label"])

            # Synthetic copies
            for _ in range(self.num_augmentations):
                augmented = transform(image=original)["image"]
                aug_images.append(augmented)
                aug_test_types.append(row["type_test"])
                aug_labels.append(row["label"])

        df = pl.DataFrame(
            {
                "img": np.array(aug_images),
                "type_test": np.array(aug_test_types),
                "label": np.array(aug_labels),
            }
        )

        return df.sample(fraction=1.0, shuffle=True, seed=True)
