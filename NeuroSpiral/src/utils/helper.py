import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import albumentations as A
import matplotlib.pyplot as plt
from src.constant.constant import *
from typing import Dict, Any, Tuple
from torch.utils.data import DataLoader, Dataset
from torchmetrics import BinaryAccuracy, BinaryF1Score
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
)


class TTAWrapper:
    """
    Test Time Augmentation for Parkinson detection.

    Applies multiple augmentations at inference time and averages
    the predictions for more stable results.
    """

    def __init__(self):
        self.transforms = self._build_tta_transforms()

    def _build_tta_transforms(self) -> list:
        """
        Builds a list of TTA transforms.

        No HorizontalFlip — drawing direction matters for spiral/wave.

        Returns:
            list: List of albumentations transforms.
        """
        return [
            None,  # original — no transform
            A.Rotate(limit=10, p=1.0),
            A.Rotate(limit=-10, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=-0.15, contrast_limit=0, p=1.0),
        ]

    def apply(self, img_np: np.ndarray) -> list:
        """
        Applies all TTA transforms to a single image.

        Args:
            img_np: np.ndarray (H, W, C) uint8 image.

        Returns:
            list of torch.Tensor: Each tensor shape (3, H, W), normalized.
        """
        results = []

        for transform in self.transforms:
            if transform is None:
                aug_img = img_np.copy()
            else:
                aug_img = transform(image=img_np)["image"]

            tensor = torch.from_numpy(aug_img.astype(np.float32) / 255.0)
            tensor = tensor.permute(2, 0, 1)  # (C, H, W)
            results.append(tensor)

        return results  # list of 5 tensors


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """
    Wrap a ``Dataset`` in a ``DataLoader``.

    Parameters
    ----------
    dataset    : Dataset
    batch_size : int
    shuffle    : bool   Default ``True``.

    Returns
    -------
    DataLoader
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_on_test_set(
    model: torch.nn.Module, test_dataloader: DataLoader
) -> Dict[str, Any]:
    """
    Evaluate a trained model on a test dataset using Test-Time Augmentation (TTA).

    This function performs inference on the test dataset, applies multiple
    augmentations (TTA), averages predictions, and computes evaluation metrics
    along with visualization plots.

    Args:
        model (torch.nn.Module):
            The trained PyTorch model. Expected to take (image, math_features)
            as input and output logits.

        test_dataloader (DataLoader):
            DataLoader for the test dataset. Each batch should return:
            (images, math_features, labels)

    Returns:
        Dict[str, Any]: Dictionary containing evaluation results:
            - accuracy (float): Accuracy percentage
            - f1 (float): F1 score percentage
            - auc (float): ROC-AUC score percentage
            - sensitivity (float): Recall for positive class (PD)
            - specificity (float): Recall for negative class (HC)
            - precision (float): Precision percentage
            - cm (np.ndarray): Confusion matrix
            - results (dict):
                - true (np.ndarray): Ground truth labels
                - pred_prob (np.ndarray): Predicted probabilities
                - pred_label (np.ndarray): Binary predictions
    """
    tta = TTAWrapper()
    results = {"true": [], "pred": []}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    acc_fn = BinaryAccuracy(threshold=0.5).to(device)
    f1_fn = BinaryF1Score(threshold=0.5).to(device)

    model.to(device)
    model.eval()
    acc_fn.reset()
    f1_fn.reset()

    with torch.no_grad():
        for imgs, math_features, labels in tqdm(
            test_dataloader, desc="Testing with TTA"
        ):
            batch_size = imgs.shape[0]
            imgs_np = (imgs.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
            tta_probs = torch.zeros(batch_size).to(device)

            for transform in tta.transforms:
                aug_imgs = []
                for i in range(batch_size):
                    img_np = imgs_np[i]
                    aug_img = (
                        img_np.copy()
                        if transform is None
                        else transform(image=img_np)["image"]
                    )
                    tensor = torch.from_numpy(
                        aug_img.astype(np.float32) / 255.0
                    ).permute(2, 0, 1)
                    aug_imgs.append(tensor)

                aug_batch = torch.stack(aug_imgs).to(device)
                math_feat = math_features.to(device)
                logits = model(aug_batch, math_feat)
                probs = torch.sigmoid(logits).squeeze(1)
                tta_probs += probs

            tta_probs /= len(tta.transforms)
            labels = labels.to(device).view(-1)

            acc_fn.update(tta_probs, labels)
            f1_fn.update(tta_probs, labels)
            results["true"].extend(labels.cpu().numpy())
            results["pred"].extend(tta_probs.cpu().numpy())

    accuracy = acc_fn.compute().item() * 100
    f1_score = f1_fn.compute().item() * 100

    true_labels = np.array(results["true"])
    pred_probs = np.array(results["pred"])
    pred_labels = (pred_probs > 0.5).astype(int)

    auc = roc_auc_score(true_labels, pred_probs) * 100

    cm = confusion_matrix(true_labels, pred_labels)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn + 1e-6) * 100  # Recall for PD
    specificity = tn / (tn + fp + 1e-6) * 100  # Recall for HC

    precision = tp / (tp + fp + 1e-6) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["PD", "HC"])
    disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title("Confusion Matrix")

    fpr, tpr, _ = roc_curve(true_labels, pred_probs)
    axes[1].plot(fpr, tpr, label=f"AUC = {auc:.2f}%", color="steelblue")
    axes[1].plot([0, 1], [0, 1], "k--", linewidth=0.8)
    axes[1].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[1].set_ylabel("True Positive Rate (Sensitivity)")
    axes[1].set_title("ROC Curve")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    results = {
        "true": true_labels,
        "pred_prob": np.round(pred_probs, 2),
        "pred_label": pred_labels,
    }

    return {
        "accuracy": np.round(accuracy, 2),
        "f1": np.round(f1_score, 2),
        "auc": np.round(auc, 2),
        "sensitivity": np.round(sensitivity, 2),
        "specificity": np.round(specificity, 2),
        "precision": np.round(precision, 2),
        "cm": cm,
        "results": results,
    }


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    image_size: Tuple[int, int] = IMAGE_SIZE,
) -> None:
    """
    Export a trained ``ParkinsonClassifier`` to ONNX format.

    The function creates two dummy inputs that match the model's expected
    signature and calls ``torch.onnx.export`` with dynamic batch-size support.

    Parameters
    ----------
    model       : nn.Module
        Trained model (will be set to eval mode automatically).
    output_path : str
        Destination ``.onnx`` file path.
    image_size  : tuple[int, int]
        ``(width, height)`` used during training.  Defaults to ``IMAGE_SIZE``.

    Notes
    -----
    * Input names  : ``"image"``, ``"math_features"``
    * Output name  : ``"logit"``
    * Opset        : 17
    * Dynamic axes : batch dimension is dynamic for both inputs and the output.
    """
    model.eval()
    device = next(model.parameters()).device

    h, w = image_size[1], image_size[0]

    dummy_image = torch.zeros(1, 3, h, w, device=device)
    dummy_math_features = torch.zeros(1, MATH_FEATURE_DIM, device=device)

    torch.onnx.export(
        model,
        args=(dummy_image, dummy_math_features),
        f=output_path,
        input_names=["image", "math_features"],
        output_names=["logit"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "math_features": {0: "batch_size"},
            "logit": {0: "batch_size"},
        },
        opset_version=18,
    )

    print(f"Model exported to ONNX → {output_path}")
