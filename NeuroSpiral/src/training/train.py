import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from typing import Any, Dict, List, Tuple
from torchmetrics import BinaryAccuracy, BinaryF1Score



class ModelTrainer:
    """
    Handles the training loop, validation, checkpointing, and history
    plotting for ``ParkinsonClassifier``.
    """
 
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        num_epochs: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_path: str,
    ) -> None:
        self.model            = model
        self.loss_fn          = loss_fn
        self.optimizer        = optimizer
        self.scheduler        = scheduler
        self.num_epochs       = num_epochs
        self.train_loader     = train_loader
        self.val_loader       = val_loader
        self.checkpoint_path  = checkpoint_path
 
        self.device           = "cuda" if torch.cuda.is_available() else "cpu"
        self.best_val_f1      = float("-inf")
 
        self.acc_metric = BinaryAccuracy(threshold=0.5).to(self.device)
        self.f1_metric  = BinaryF1Score(threshold=0.5).to(self.device)
 
        self.model.to(self.device)
 
    # ------------------------------------------------------------------
    def _run_one_epoch(
        self,
        is_training: bool,
        loader: DataLoader,
    ) -> Tuple[float, float, float]:
        """
        Run a single epoch (training or validation).
 
        Parameters
        ----------
        is_training : bool
            ``True``  → model.train() + gradient updates.
            ``False`` → model.eval()  + no_grad context.
        loader : DataLoader
 
        Returns
        -------
        avg_loss : float
        accuracy : float   percentage
        f1_score : float   percentage
        """
        self.model.train() if is_training else self.model.eval()
 
        context     = torch.enable_grad() if is_training else torch.no_grad()
        desc        = "Training" if is_training else "Validation"
        total_loss  = 0.0
        num_batches = 0
 
        self.acc_metric.reset()
        self.f1_metric.reset()
 
        with context:
            for images, math_features, labels in tqdm(loader, desc=desc):
                images        = images.to(self.device)
                math_features = math_features.to(self.device)
                labels        = labels.to(self.device)
 
                logits = self.model(images, math_features)
                loss   = self.loss_fn(logits, labels.unsqueeze(1))
 
                if is_training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
 
                probs = torch.sigmoid(logits).squeeze()
                self.acc_metric.update(probs, labels)
                self.f1_metric.update(probs, labels)
 
                total_loss  += loss.item()
                num_batches += 1
 
        avg_loss = total_loss / num_batches
        accuracy = self.acc_metric.compute().item() * 100
        f1_score = self.f1_metric.compute().item() * 100
 
        return avg_loss, accuracy, f1_score
 
    # ------------------------------------------------------------------
    def _plot_history(self, history: Dict[str, List[float]]) -> None:
        """
        Plot training vs validation Loss, Accuracy, and F1 score.
 
        Parameters
        ----------
        history : dict
            Keys: ``train_loss``, ``val_loss``, ``train_acc``,
            ``val_acc``, ``train_f1``, ``val_f1``.
        """
        epoch_range = range(1, len(history["train_loss"]) + 1)
 
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        metrics = [
            ("Loss",     "train_loss", "val_loss"),
            ("Accuracy", "train_acc",  "val_acc"),
            ("F1 Score", "train_f1",   "val_f1"),
        ]
 
        for ax, (title, train_key, val_key) in zip(axes, metrics):
            ax.plot(epoch_range, history[train_key], marker="o", label=f"Train {title}")
            ax.plot(epoch_range, history[val_key],   marker="s", label=f"Val {title}")
            ax.set(xlabel="Epochs", ylabel=title, title=f"Training vs Validation {title}")
            ax.legend()
            ax.grid(alpha=0.3)
 
        plt.tight_layout()
        plt.show()
 
    # ------------------------------------------------------------------
    def fit(self) -> Dict[str, List[float]]:
        """
        Run the full training loop.
 
        Saves the model checkpoint whenever validation F1 improves.
 
        Returns
        -------
        history : dict
            Training history with keys:
            ``train_loss``, ``val_loss``, ``train_acc``,
            ``val_acc``, ``train_f1``, ``val_f1``.
        """
        history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss":  [],
            "train_acc":  [], "val_acc":   [],
            "train_f1":   [], "val_f1":    [],
        }
 
        for epoch in range(self.num_epochs):
            print(f"\nEpoch [{epoch + 1}/{self.num_epochs}]")
 
            train_loss, train_acc, train_f1 = self._run_one_epoch(True,  self.train_loader)
            val_loss,   val_acc,   val_f1   = self._run_one_epoch(False, self.val_loader)
 
            # Checkpoint on best validation F1
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                torch.save(self.model.state_dict(), self.checkpoint_path)
                print(f"  Checkpoint saved — Val F1: {val_f1:.2f}%")
 
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]
 
            print(
                f"  LR: {current_lr:.6f}\n"
                f"  Train → Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% | F1: {train_f1:.2f}%\n"
                f"  Val   → Loss: {val_loss:.4f}   | Acc: {val_acc:.2f}%   | F1: {val_f1:.2f}%"
            )
 
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)
            history["train_f1"].append(train_f1)
            history["val_f1"].append(val_f1)
 
        self._plot_history(history)
        return history
 