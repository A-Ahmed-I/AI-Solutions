import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from src.constant.constant import *
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader, Dataset
from torchmetrics import BinaryAccuracy, BinaryF1Score


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
    model: nn.Module,
    test_loader: DataLoader,
) -> Tuple[float, float, Dict[str, np.ndarray]]:
    """
    Evaluate a trained model on a held-out test set.

    Parameters
    ----------
    model       : nn.Module
    test_loader : DataLoader

    Returns
    -------
    accuracy    : float   percentage
    f1_score    : float   percentage
    results     : dict
        Keys: ``true``, ``pred_prob``, ``pred_label``
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    acc_metric = BinaryAccuracy(threshold=0.5).to(device)
    f1_metric = BinaryF1Score(threshold=0.5).to(device)

    model.to(device)
    model.eval()
    acc_metric.reset()
    f1_metric.reset()

    all_true_labels: List[float] = []
    all_pred_probs: List[float] = []

    with torch.no_grad():
        for images, math_features, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            math_features = math_features.to(device)
            labels = labels.to(device).view(-1)

            logits = model(images, math_features)
            probs = torch.sigmoid(logits).view(-1)

            acc_metric.update(probs, labels)
            f1_metric.update(probs, labels)

            all_true_labels.extend(labels.cpu().numpy())
            all_pred_probs.extend(probs.cpu().numpy())

    accuracy = round(acc_metric.compute().item() * 100, 2)
    f1_score = round(f1_metric.compute().item() * 100, 2)

    pred_probs = np.round(np.array(all_pred_probs), 2)
    pred_labels = (pred_probs > 0.5).astype(int)

    results = {
        "true": np.array(all_true_labels),
        "pred_prob": pred_probs,
        "pred_label": pred_labels,
    }

    return accuracy, f1_score, results


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
