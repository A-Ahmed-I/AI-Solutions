from src.constant.constant import *
from src.pipeline.pipeline import ParkinsonPipeline


def main() -> None:
    """
    Main entry point for the Parkinson's classification pipeline.

    This function initializes the pipeline with dataset paths and runs all stages:
    data loading, augmentation, feature extraction, model training, evaluation, and ONNX export.

     Usage:
        python src/pipeline/main.py
    """
    pipeline = ParkinsonPipeline(DATASET_ROOT, CHECKPOINT_PATH, ONNX_EXPORT_PATH)
    pipeline.run()


if __name__ == "__main__":
    main()
