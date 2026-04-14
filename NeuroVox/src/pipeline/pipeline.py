import torch
import numpy as np
import polars as pl
import torch.nn as nn
from typing import List, Tuple
from src.constant.constant import *
from src.training.train import Trainer
from src.data.data_loader import Loader
from torch.utils.data import DataLoader
from src.data.metadata import CreateMetadata
from src.models.neurovox_tl import NeuroVoxTL
from src.preprocessing.processing import PreProcessing


class PipeLine:
    """
    Full Parkinson's Disease audio classification pipeline.

    Responsibilities:
        - Create and validate metadata
        - Preprocess audio (padding, chunking, augmentation, mel-spectrogram)
        - Split data and create PyTorch DataLoaders
        - Initialize and train model
        - Plot training history
        - Run inference on test set
        - Export trained model to ONNX
    """

    def __init__(self, base_path: str, checkpoint_path: str, onnx_path: str) -> None:
        """
        Args:
            base_path (str): Dataset root directory.
            checkpoint_path (str): Path to save/load model checkpoints.
            onnx_path (str): Path to export the trained model as ONNX.
        """
        self.base_path: str = base_path
        self.checkpoint_path: str = checkpoint_path
        self.onnx_path: str = onnx_path

    def create_metadata(self, min_duration: float) -> pl.DataFrame:
        """Load and validate audio metadata."""
        metadata_creator = CreateMetadata(self.base_path, min_duration)
        return metadata_creator.load_metadata()

    def preprocess_audio(self, metadata: pl.DataFrame) -> List[Tuple[np.ndarray, str]]:
        """Process all audio files into mel-spectrogram features."""
        preprocessor = PreProcessing(
            sample_rate,
            min_duration,
            chunk_duration,
            overlap_ratio,
            energy_threshold,
            silence_db_threshold,
            variance_threshold,
            n_fft,
            n_mels,
        )
        return preprocessor.process_all_data(metadata)

    def prepare_dataloaders(
        self, full_data: List[Tuple[np.ndarray, str]]
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Split data and create PyTorch DataLoaders."""
        loader = Loader(full_data, train_ratio, test_ratio, batch_size)
        return loader.get_dataloaders()

    def train_model(self, train_loader: DataLoader, val_loader: DataLoader) -> Trainer:
        """Initialize model, optimizer, scheduler and train."""
        model = NeuroVoxTL(1)
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2
        )

        trainer = Trainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epochs=epochs,
            train_loader=train_loader,
            val_loader=val_loader,
            scheduler=scheduler,
            checkpoint_path=self.checkpoint_path,
        )

        history = trainer.train_model()
        trainer.plot_training_history(history)
        return trainer

    def run_inference(self, trainer: Trainer, test_loader: DataLoader) -> None:
        """Run inference and display confusion matrix."""
        trainer.run_inference(test_loader)

    def export_onnx(self, trainer: Trainer, model: nn.Module) -> None:
        """Export trained model to ONNX format."""
        model.load_state_dict(torch.load(self.checkpoint_path))
        model.eval()

        dummy_input = torch.randn(1, 1, n_mels, 517).to(trainer.device)

        torch.onnx.export(
            model,
            dummy_input,
            self.onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["outputs"],
            dynamic_axes={"input": {0: "batch_size"}, "outputs": {0: "batch_size"}},
        )
        print(f"Model exported to {self.onnx_path}")

    def run(self) -> None:
        """Execute the full pipeline."""
        metadata = self.create_metadata(min_duration)
        full_data = self.preprocess_audio(metadata)
        train_loader, test_loader, val_loader = self.prepare_dataloaders(full_data)
        trainer = self.train_model(train_loader, val_loader)
        self.run_inference(trainer, test_loader)
        self.export_onnx(trainer, trainer.model)
