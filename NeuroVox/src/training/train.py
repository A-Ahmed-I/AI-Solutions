import torch
import numpy as np
from tqdm import tqdm
from torch import nn
from typing import Dict, Tuple
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay


class Trainer:
    """
    Trainer class for PyTorch binary classification models.

    Responsibilities:
        - Training loop with validation
        - Metric tracking (Accuracy, F1)
        - Checkpoint saving
        - Training history plotting
        - Inference & confusion matrix generation
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        checkpoint_path: str,
    ) -> None:
        """
        Initialize the Trainer.

        Args:
            model (nn.Module): PyTorch model to train.
            loss_fn (nn.Module): Loss function.
            optimizer (torch.optim.Optimizer): Optimizer.
            epochs (int): Number of training epochs.
            train_loader (DataLoader): Training DataLoader.
            val_loader (DataLoader): Validation DataLoader.
            scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
            checkpoint_path (str): File path to save best model checkpoint.
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_path = checkpoint_path

        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.best_val_acc: float = float("-inf")

        self.acc_fn = BinaryAccuracy(threshold=0.5).to(self.device)
        self.f1_fn = BinaryF1Score(threshold=0.5).to(self.device)

        self.model.to(self.device)

    def run_epoch(
        self,
        train: bool,
        dataloader: DataLoader,
    ) -> Tuple[float, float, float]:
        """
        Run a single epoch of training or validation.

        Args:
            train (bool): True for training, False for validation.
            dataloader (DataLoader): DataLoader for the current phase.

        Returns:
            Tuple containing:
                - average_loss (float): Mean loss for the epoch.
                - accuracy (float): Accuracy (%) for the epoch.
                - f1 (float): F1 score (%) for the epoch.
        """
        self.model.train() if train else self.model.eval()
        context = torch.enable_grad() if train else torch.inference_mode()

        running_loss: float = 0.0
        count: int = 0
        phase: str = "Training" if train else "Validation"

        self.acc_fn.reset()
        self.f1_fn.reset()

        with context:
            for spec, labels in tqdm(dataloader, desc=phase):
                spec = spec.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(spec)
                loss = self.loss_fn(logits, labels.unsqueeze(1).float())

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                preds = torch.sigmoid(logits).squeeze()
                self.acc_fn.update(preds, labels)
                self.f1_fn.update(preds, labels)

                running_loss += loss.item()
                count += 1

        avg_loss: float = running_loss / count
        accuracy: float = self.acc_fn.compute().item() * 100
        f1: float = self.f1_fn.compute().item() * 100

        return avg_loss, accuracy, f1

    def plot_training_history(self, history: Dict[str, list]) -> None:
        """
        Plot training vs validation curves for Loss, Accuracy, and F1.

        Args:
            history (Dict[str, list]): Dictionary containing training history
                with keys: "train_loss", "val_loss", "train_acc", "val_acc", "train_f1", "val_f1".
        """
        epochs = range(1, len(history["train_loss"]) + 1)

        plt.figure(figsize=(20, 5))

        # Loss
        plt.subplot(1, 3, 1)
        plt.plot(epochs, history["train_loss"], marker="o", label="Train Loss")
        plt.plot(epochs, history["val_loss"], marker="s", label="Val Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.grid(alpha=0.3)

        # Accuracy
        plt.subplot(1, 3, 2)
        plt.plot(epochs, history["train_acc"], marker="o", label="Train Acc")
        plt.plot(epochs, history["val_acc"], marker="s", label="Val Acc")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Training vs Validation Accuracy")
        plt.legend()
        plt.grid(alpha=0.3)

        # F1
        plt.subplot(1, 3, 3)
        plt.plot(epochs, history["train_f1"], marker="o", label="Train F1")
        plt.plot(epochs, history["val_f1"], marker="s", label="Val F1")
        plt.xlabel("Epochs")
        plt.ylabel("F1")
        plt.title("Training vs Validation F1")
        plt.legend()
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    def train_model(self) -> Dict[str, list]:
        """
        Main training loop for all epochs.

        Returns:
            history (Dict[str, list]): Dictionary containing training and validation
            metrics per epoch.
        """
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "train_f1": [],
            "val_f1": [],
        }

        for epoch in range(self.epochs):
            print(f"\nEpoch [{epoch + 1}/{self.epochs}]")

            train_loss, train_acc, train_f1 = self.run_epoch(True, self.train_loader)
            val_loss, val_acc, val_f1 = self.run_epoch(False, self.val_loader)

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save(self.model.state_dict(), self.checkpoint_path)
                print(
                    f"Checkpoint saved at epoch {epoch + 1} "
                    f"(Validation Accuracy: {val_acc:.2f}%)"
                )

            self.scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["train_f1"].append(train_f1)
            history["val_f1"].append(val_f1)

            print(
                f"Train Loss: {train_loss:.4f} | "
                f"Acc: {train_acc:.2f}% | "
                f"F1: {train_f1:.4f}"
            )
            print(
                f"Val   Loss: {val_loss:.4f} | "
                f"Acc: {val_acc:.2f}% | "
                f"F1: {val_f1:.4f}"
            )

        return history

    def run_inference(
        self,
        dataloader: DataLoader,
    ) -> Tuple[float, float, np.ndarray]:
        """
        Run inference on a dataset and compute metrics.

        Args:
            dataloader (DataLoader): DataLoader for test set.

        Returns:
            Tuple containing:
                - accuracy (float): Accuracy in %
                - f1 (float): F1 score in %
                - confusion_matrix (np.ndarray): Confusion matrix
        """
        self.model.eval()

        y_true, y_pred = [], []

        with torch.inference_mode():
            for x, y in dataloader:
                x = x.to(self.device)

                probs = torch.sigmoid(self.model(x)).squeeze()
                preds = (probs >= 0.5).int()

                y_true.extend(y.numpy())
                y_pred.extend(preds.cpu().numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        acc: float = (y_true == y_pred).mean() * 100
        f1: float = f1_score(y_true, y_pred)

        cm: np.ndarray = confusion_matrix(y_true, y_pred)
        ConfusionMatrixDisplay(cm).plot(cmap="Blues", values_format="d")
        plt.show()

        return acc, f1, cm
