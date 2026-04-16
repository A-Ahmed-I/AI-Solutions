from src.constant.constant import *
from src.pipeline.pipeline import PipeLine


def main():
    pipeline = PipeLine(base_path, checkpoint_path, onnx_path)

    pipeline.run()


if __name__ == "__main__":
    main()
