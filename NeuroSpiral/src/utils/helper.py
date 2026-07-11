import torch
import torch.nn as nn
import numpy as np
import polars as pl
from tqdm import tqdm
import albumentations as A
import matplotlib.pyplot as plt
from src.constant.constant import *
from typing import Dict, Any, Tuple
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    ConfusionMatrixDisplay,
)


def fit_feature_reducers(
    data: dict,
) -> Tuple[np.ndarray, VarianceThreshold, StandardScaler, PCA]:
    """
    Fit feature reduction pipeline on training data only.

    Pipeline steps:
        1. Variance Threshold (remove low-variance features)
        2. Standard Scaling
        3. PCA (retain 95% variance)

    Parameters
    ----------
    data : dict
        Must contain key "math_features" as a column-like structure
        where each row is a feature vector.

    Returns
    -------
    Tuple containing:
        - X_reduced : np.ndarray
            Transformed feature matrix after PCA
        - vt : VarianceThreshold
            Fitted variance threshold object
        - scaler : StandardScaler
            Fitted scaler
        - pca : PCA
            Fitted PCA model
    """
    X = np.stack(data["math_features"].to_numpy())

    variance_selector = VarianceThreshold(threshold=0.01)
    X_var = variance_selector.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_var)

    pca = PCA(n_components=0.95, random_state=42)
    X_reduced = pca.fit_transform(X_scaled)

    print(f"Features: {X.shape[1]} → {X_reduced.shape[1]}")

    return X_reduced, variance_selector, scaler, pca


def transform_features_only(
    data: dict, variance_selector: VarianceThreshold, scaler: StandardScaler, pca: PCA
) -> np.ndarray:
    """
    Apply pre-fitted feature reduction pipeline (no fitting).

    Used for validation/test data.

    Parameters
    ----------
    data : dict
        Must contain "math_features"
    variance_selector : VarianceThreshold
        Fitted variance threshold
    scaler : StandardScaler
        Fitted scaler
    pca : PCA
        Fitted PCA model

    Returns
    -------
    np.ndarray
        Transformed feature matrix
    """
    X = np.stack(data["math_features"].to_numpy())

    X_var = variance_selector.transform(X)
    X_scaled = scaler.transform(X_var)
    X_reduced = pca.transform(X_scaled)

    return X_reduced


def attach_reduced_features_to_df(
    df: pl.DataFrame, X_reduced: np.ndarray
) -> pl.DataFrame:
    """
    Replace "math_features" column in DataFrame with reduced features.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame
    X_reduced : np.ndarray
        Reduced feature matrix (N, D)

    Returns
    -------
    pl.DataFrame
        Updated DataFrame with new feature representation
    """
    df = df.with_columns(pl.Series("math_features", X_reduced.tolist()))

    df = df.with_columns(pl.col("math_features").list.to_array(X_reduced.shape[1]))

    return df


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



def find_best_threshold(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """
    Find the optimal classification threshold based on F1-score.

    This function evaluates multiple thresholds on validation predictions
    and selects the one that maximizes the F1-score.

    Args:
        probabilities (np.ndarray): Predicted probabilities (sigmoid outputs).
        labels (np.ndarray): Ground truth binary labels.

    Returns:
        float: Best threshold value.
    """
    candidate_thresholds = np.arange(0.3, 0.7, 0.01)

    best_threshold = max(
        candidate_thresholds,
        key=lambda t: f1_score(labels, (probabilities > t).astype(int)),
    )

    return best_threshold


def get_validation_predictions(
    model: torch.nn.Module, val_loader: torch.utils.data.DataLoader, device: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect model predictions and labels from validation dataset.

    Args:
        model (torch.nn.Module): Trained model.
        val_loader (DataLoader): Validation data loader.
        device (str): Computation device ("cuda" or "cpu").

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Predicted probabilities
            - Ground truth labels
    """
    model.eval()

    all_probs: list = []
    all_labels: list = []

    with torch.no_grad():
        for images, math_features, labels in val_loader:
            images = images.to(device)
            math_features = math_features.to(device)

            logits = model(images, math_features)
            probs = torch.sigmoid(logits).squeeze(1)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_probs), np.array(all_labels)

def evaluate_with_tta(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
) -> Dict[str, Any]:
    """
    Evaluate model on test dataset using Test-Time Augmentation (TTA).

    Workflow:
        1. Compute best threshold using validation set.
        2. Apply TTA on test images.
        3. Average predictions.
        4. Compute evaluation metrics.

    Args:
        model (torch.nn.Module): Trained model.
        test_loader (DataLoader): Test dataset loader.
        val_loader (DataLoader): Validation dataset loader.

    Returns:
        Dict[str, Any]: Evaluation results including:
            - accuracy
            - f1 score
            - auc
            - sensitivity
            - specificity
            - precision
            - best threshold
            - confusion matrix
            - raw predictions
    """

    tta_transforms = [
        None,
        A.Rotate(limit=10, p=1.0),
        A.Rotate(limit=-10, p=1.0),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0, p=1.0),
        A.RandomBrightnessContrast(brightness_limit=-0.15, contrast_limit=0, p=1.0),
    ]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    val_probs, val_labels = get_validation_predictions(model, val_loader, device)
    best_threshold = find_best_threshold(val_probs, val_labels)

    print(f"Best threshold (from val): {best_threshold:.2f}")

    results = {"true": [], "pred": []}

    with torch.no_grad():
        for images, math_features, labels in tqdm(test_loader, desc="Testing with TTA"):

            batch_size = images.shape[0]
            images_np = (images[:, 0, :, :].cpu().numpy() * 255).astype(np.uint8)

            tta_probabilities = torch.zeros(batch_size).to(device)

            for transform in tta_transforms:
                augmented_batch = []

                for i in range(batch_size):
                    img_np = images_np[i]

                    if transform is not None:
                        img_np = transform(image=img_np)["image"]

                    tensor = torch.from_numpy(img_np.astype(np.float32) / 255.0)
                    tensor = tensor.unsqueeze(0).repeat(3, 1, 1)

                    augmented_batch.append(tensor)

                augmented_batch = torch.stack(augmented_batch).to(device)
                math_features_device = math_features.to(device)

                logits = model(augmented_batch, math_features_device)
                probs = torch.sigmoid(logits).squeeze(1)

                tta_probabilities += probs

            tta_probabilities /= len(tta_transforms)

            results["true"].extend(labels.cpu().numpy())
            results["pred"].extend(tta_probabilities.cpu().numpy())

    true_labels = np.array(results["true"])
    pred_probs = np.array(results["pred"])
    pred_labels = (pred_probs > best_threshold).astype(int)

    accuracy = accuracy_score(true_labels, pred_labels) * 100
    f1 = f1_score(true_labels, pred_labels) * 100
    auc = roc_auc_score(true_labels, pred_probs) * 100

    cm = confusion_matrix(true_labels, pred_labels)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn + 1e-6) * 100
    specificity = tn / (tn + fp + 1e-6) * 100
    precision = tp / (tp + fp + 1e-6) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["PD", "HC"]).plot(
        ax=axes[0], colorbar=False, cmap="Blues"
    )

    axes[0].set_title("Confusion Matrix")

    fpr, tpr, _ = roc_curve(true_labels, pred_probs)
    axes[1].plot(fpr, tpr, label=f"AUC = {auc:.2f}%", color="steelblue")
    axes[1].plot([0, 1], [0, 1], "k--", linewidth=0.8)

    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # =========================
    # Return Results
    # =========================
    return {
        "accuracy": np.round(accuracy, 2),
        "f1": np.round(f1, 2),
        "auc": np.round(auc, 2),
        "sensitivity": np.round(sensitivity, 2),
        "specificity": np.round(specificity, 2),
        "precision": np.round(precision, 2),
        "best_thresh": np.round(best_threshold, 2),
        "cm": cm,
        "results": {
            "true": true_labels,
            "pred_prob": np.round(pred_probs, 2),
            "pred_label": pred_labels,
        },
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
