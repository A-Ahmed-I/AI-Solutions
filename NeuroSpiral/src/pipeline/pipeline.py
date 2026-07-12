import torch
import joblib
import polars as pl
import torch.nn as nn
from pathlib import Path
from src.utils.helper import *
from src.training.train import *
from src.constant.constant import *
from torch.utils.data import DataLoader
from typing import Any, Dict, List, Union
from src.data.metadata import MetadataBuilder
from src.data.custom_data import ParkinsonDataset
from sklearn.model_selection import train_test_split
from src.augmentation.augmented import TrainAugmentor
from src.model.neurospiral import MultimodalGatedModel
from src.processing.processing import ImagePreprocessor
from src.feature_extraction.handcrafted import FeatureExtractor
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


class ParkinsonPipeline:
    """
    Orchestrates the full Parkinson's spiral-drawing detection workflow.

    Steps
    -----
    1. ``build_data``    — scan dataset, preprocess images, train/test split
    2. ``build_loaders`` — augment train set, extract + reduce features, build DataLoaders
    3. ``build_model``   — instantiate model, optimizer, and LR scheduler
    4. ``train``         — run training loop, save best checkpoint
    5. ``evaluate``      — load best checkpoint, run TTA evaluation on test set
    6. ``export_onnx``   — export trained model to ONNX

    Usage
    -----
    ::

        pipeline = ParkinsonPipeline(DATASET_ROOT, CHECKPOINT_PATH, ONNX_EXPORT_PATH)
        pipeline.run()
    """

    def __init__(
        self,
        dataset_root: Union[str, Path] = DATASET_ROOT,
        checkpoint_path: str = CHECKPOINT_PATH,
        onnx_export_path: str = ONNX_EXPORT_PATH,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.checkpoint_path = checkpoint_path
        self.onnx_path = onnx_export_path

        self.train_raw: pl.DataFrame
        self.test_raw: pl.DataFrame
        self.train_loader: DataLoader
        self.val_loader: DataLoader
        self.test_loader: DataLoader
        self.model: MultimodalGatedModel
        self.optimizer: torch.optim.Optimizer
        self.scheduler: Any
        self.history: Dict[str, List[float]]
        self.metrics: Dict[str, Any]

        self._vt = None
        self._scaler = None
        self._pca = None

    # ------------------------------------------------------------------ #
    def build_data(self) -> None:
        """Scan dataset directory, preprocess images, split into train / test."""
        print("\n[1/6] Building dataset …")

        metadata = MetadataBuilder(self.dataset_root).build()
        raw_data = ImagePreprocessor(metadata).process_all(IMAGE_SIZE)
        raw_df = pl.DataFrame(raw_data)

        self.train_raw, self.test_raw = train_test_split(
            raw_df,
            test_size=TEST_SPLIT_RATIO,
            stratify=raw_df["label"],
            random_state=RANDOM_SEED,
        )
        print(f"    Train: {len(self.train_raw)} | Test: {len(self.test_raw)}")

    # ------------------------------------------------------------------ #
    def build_loaders(self) -> None:
        """
        Augment training images → extract HOG+LBP features →
        fit PCA on train → build all three DataLoaders.
        """
        print("\n[2/6] Building data loaders …")

        # ── augment ──────────────────────────────────────────────────────
        augmented_train = TrainAugmentor(
            self.train_raw, NUM_AUGMENTATIONS_PER_IMAGE
        ).augment()

        # ── feature extraction ───────────────────────────────────────────
        augmented_train = FeatureExtractor(augmented_train).extract_all_features()
        test_with_feat = FeatureExtractor(self.test_raw).extract_all_features()

        # ── train / val split ────────────────────────────────────────────
        train_split, val_split = train_test_split(
            augmented_train,
            test_size=0.1,
            stratify=augmented_train["label"],
            random_state=RANDOM_SEED,
        )

        # ── feature reduction (fit only on train) ────────────────────────
        m_f_train, self._vt, self._scaler, self._pca = fit_feature_reducers(train_split)
        train_split = attach_reduced_features_to_df(train_split, m_f_train)

        m_f_val = transform_features_only(val_split, self._vt, self._scaler, self._pca)
        m_f_test = transform_features_only(
            test_with_feat, self._vt, self._scaler, self._pca
        )
        val_split = attach_reduced_features_to_df(val_split, m_f_val)
        test_with_feat = attach_reduced_features_to_df(test_with_feat, m_f_test)

        # store reduced dim so the model knows its input size
        self._feature_dim = m_f_train.shape[1]

        # ── DataLoaders ──────────────────────────────────────────────────
        self.train_loader = DataLoader(
            ParkinsonDataset(train_split), batch_size=TRAIN_BATCH_SIZE, shuffle=True
        )
        self.val_loader = DataLoader(
            ParkinsonDataset(val_split), batch_size=VAL_BATCH_SIZE, shuffle=False
        )
        self.test_loader = DataLoader(
            ParkinsonDataset(test_with_feat), batch_size=TEST_BATCH_SIZE, shuffle=False
        )

        print(
            f"    Train batches: {len(self.train_loader)} | "
            f"Val batches: {len(self.val_loader)} | "
            f"Test batches: {len(self.test_loader)}"
        )

    # ------------------------------------------------------------------ #
    def build_model(self) -> None:
        """Instantiate MultimodalGatedModel, loss, AdamW optimizer, and LR scheduler."""
        print("\n[3/6] Building model …")

        self.model = MultimodalGatedModel(feature_dim=self._feature_dim)
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        warmup = LinearLR(
            self.optimizer, WARMUP_START_FACTOR, WARMUP_END_FACTOR, WARMUP_TOTAL_ITERS
        )
        cosine = CosineAnnealingLR(self.optimizer, T_max=COSINE_T_MAX)
        self.scheduler = SequentialLR(
            self.optimizer, [warmup, cosine], milestones=SCHEDULER_MILESTONES
        )

        total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"    Trainable parameters: {total_params:,}")

    # ------------------------------------------------------------------ #
    def train(self) -> None:
        """Run the full training loop and save the best checkpoint."""
        print("\n[4/6] Training …")

        trainer = Trainer(
            model=self.model,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epochs=NUM_EPOCHS,
            train_dataloader=self.train_loader,
            val_dataloader=self.val_loader,
            checkpoint_path=self.checkpoint_path,
        )
        self.history = trainer.train_model()
        trainer.plot_training_history(self.history)

    # ------------------------------------------------------------------ #
    def evaluate(self) -> None:
        """Load best checkpoint and run TTA evaluation on the held-out test set."""
        print("\n[5/6] Evaluating …")

        self.model.load_state_dict(torch.load(self.checkpoint_path))
        self.metrics = evaluate_with_tta(self.model, self.test_loader, self.val_loader)

        print(f"    Accuracy    : {self.metrics['accuracy']}%")
        print(f"    F1 Score    : {self.metrics['f1']}%")
        print(f"    AUC-ROC     : {self.metrics['auc']}%")
        print(f"    Sensitivity : {self.metrics['sensitivity']}%")
        print(f"    Specificity : {self.metrics['specificity']}%")
        print(f"    Precision   : {self.metrics['precision']}%")
        print(f"    Best Thresh : {self.metrics['best_thresh']}")
        print(self.metrics["cm"])

    # ------------------------------------------------------------------ #
    def export_onnx(self) -> None:
        """Export the trained model to ONNX."""
        print("\n[6/6] Exporting to ONNX …")
        export_to_onnx(self.model, self.onnx_path)

    # ------------------------------------------------------------------ #
    def run(self) -> None:
        """Execute the complete end-to-end pipeline."""
        self.build_data()
        self.build_loaders()
        self.build_model()
        self.train()
        self.evaluate()
        self.export_onnx()
        joblib.dump(
            {
                "variance_selector": self._vt,
                "scaler": self._scaler,
                "pca": _pca,
            },
            REDUCERS_PATH,
        )
        print("\nPipeline finished successfully.")
