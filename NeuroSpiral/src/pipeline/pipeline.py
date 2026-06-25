import torch
import numpy as np
import polars as pl
import torch.nn as nn
from pathlib import Path
from src.utils.helper import *
from src.constant.constant import *
from torch.utils.data import DataLoader
from src.training.train import ModelTrainer
from src.data.metadata import MetadataBuilder
from typing import Any, Dict, List, Tuple, Union
from src.data.custom_data import ParkinsonDataset
from sklearn.model_selection import train_test_split
from src.model.neurospiral import ParkinsonClassifier
from src.augmentation.augmented import TrainAugmentor
from src.processing.processing import ImagePreprocessor
from src.feature_extraction.handcrafted import HandcraftedFeatureExtractor
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR


class ParkinsonPipeline:
    """
    Orchestrates the full Parkinson's detection workflow:

    1. ``build_data``    — load metadata, preprocess images, train/test split
    2. ``build_loaders`` — augment training set, extract features, make DataLoaders
    3. ``build_model``   — instantiate model, optimizer, scheduler
    4. ``train``         — run training loop, save best checkpoint
    5. ``evaluate``      — load best checkpoint, run on test set
    6. ``export``        — export model to ONNX

    Usage
    -----
    ::

        pipeline = ParkinsonPipeline(DATASET_ROOT, CHECKPOINT_PATH, ONNX_EXPORT_PATH)
        pipeline.run()
    """

    def __init__(
        self,
        dataset_root: Union[str, Path],
        checkpoint_path: str,
        onnx_export_path: str,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.checkpoint_path = checkpoint_path
        self.onnx_export_path = onnx_export_path

        self.train_raw: pl.DataFrame
        self.test_raw: pl.DataFrame
        self.train_loader: DataLoader
        self.val_loader: DataLoader
        self.test_loader: DataLoader
        self.model: ParkinsonClassifier
        self.optimizer: torch.optim.Optimizer
        self.scheduler: Any
        self.history: Dict[str, List[float]]
        self.test_results: Dict[str, np.ndarray]

    # ------------------------------------------------------------------
    def build_data(self) -> None:
        """Load metadata, preprocess images, and split into train / test."""
        metadata = MetadataBuilder(self.dataset_root).build()
        raw_data = ImagePreprocessor(metadata).process_all(IMAGE_SIZE)
        raw_df = pl.DataFrame(raw_data)

        self.train_raw, self.test_raw = train_test_split(
            raw_df,
            test_size=TEST_SPLIT_RATIO,
            stratify=raw_df["label"],
            random_state=RANDOM_SEED,
        )

    # ------------------------------------------------------------------
    def build_loaders(self) -> None:
        """Augment training data, extract features, build all DataLoaders."""
        # Augment training set only
        augmented_train = TrainAugmentor(
            self.train_raw, NUM_AUGMENTATIONS_PER_IMAGE
        ).augment()

        # Feature extraction
        train_with_features = HandcraftedFeatureExtractor(augmented_train).extract_all()
        test_with_features = HandcraftedFeatureExtractor(self.test_raw).extract_all()

        # Train / validation split
        train_split, val_split = train_test_split(
            train_with_features,
            test_size=TEST_SPLIT_RATIO,
            stratify=train_with_features["label"],
            random_state=RANDOM_SEED,
        )

        self.train_loader = build_dataloader(
            ParkinsonDataset(train_split), TRAIN_BATCH_SIZE
        )
        self.val_loader = build_dataloader(ParkinsonDataset(val_split), VAL_BATCH_SIZE)
        self.test_loader = build_dataloader(
            ParkinsonDataset(test_with_features), TEST_BATCH_SIZE, shuffle=False
        )

    # ------------------------------------------------------------------
    def build_model(self) -> None:
        """Instantiate model, loss function, optimizer, and LR scheduler."""
        self.model = ParkinsonClassifier()
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        warmup_scheduler = LinearLR(
            self.optimizer, WARMUP_START_FACTOR, WARMUP_END_FACTOR, WARMUP_TOTAL_ITERS
        )
        cosine_scheduler = CosineAnnealingLR(self.optimizer, T_max=COSINE_T_MAX)
        self.scheduler = SequentialLR(
            self.optimizer,
            [warmup_scheduler, cosine_scheduler],
            milestones=SCHEDULER_MILESTONES,
        )

    # ------------------------------------------------------------------
    def train(self) -> None:
        """Run the training loop and plot the training history."""
        trainer = ModelTrainer(
            model=self.model,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            num_epochs=NUM_EPOCHS,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            checkpoint_path=self.checkpoint_path,
        )
        self.history = trainer.fit()

    # ------------------------------------------------------------------
    def evaluate(self) -> Tuple[float, float]:
        """
        Load the best checkpoint and evaluate on the test set.

        Returns
        -------
        accuracy : float
        f1_score : float
        """
        self.model.load_state_dict(torch.load(self.checkpoint_path))
        metrics = evaluate_on_test_set(model, test_loader)
        print(f"Accuracy    : {metrics['accuracy']}%")
        print(f"F1 Score    : {metrics['f1']}%")
        print(f"AUC-ROC     : {metrics['auc']}%")
        print(f"Sensitivity : {metrics['sensitivity']}%")  # PD detection rate
        print(f"Specificity : {metrics['specificity']}%")  # HC detection rate
        print(f"Precision   : {metrics['precision']}%")

    # ------------------------------------------------------------------
    def export_onnx(self) -> None:
        """Export the trained model to ONNX format."""
        export_to_onnx(self.model, self.onnx_export_path)

    # ------------------------------------------------------------------
    def run(self) -> None:
        """Execute the full end-to-end pipeline."""
        self.build_data()
        self.build_loaders()
        self.build_model()
        self.train()
        self.evaluate()
        self.export_onnx()
