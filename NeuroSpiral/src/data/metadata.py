import polars as pl
from pathlib import Path
from typing import Generator, Union, List, Dict


class MetadataBuilder:
    """
    Scans the Parkinson's drawing dataset directory and builds a structured
    Polars DataFrame of image paths, test types, and class labels.

    Supported test types : ``spiral``, ``wave``
    Labels               : ``HC`` (Healthy Control), ``PD`` (Parkinson's Disease)
    """

    TEST_TYPES: List[str] = ["spiral", "wave"]

    def __init__(self, dataset_root: Union[str, Path]) -> None:
        """
        Parameters
        ----------
        dataset_root : str | Path
            Root directory of the parkinsons-drawings dataset.
        """
        self.dataset_root = Path(dataset_root)

    # ------------------------------------------------------------------
    def _iter_directory(self, test_type: str) -> Generator[Dict[str, str], None, None]:
        """
        Recursively scan one test-type sub-directory and yield metadata dicts.

        Parameters
        ----------
        test_type : str
            ``"spiral"`` or ``"wave"``.

        Yields
        ------
        dict
            Keys: ``path``, ``test_type``, ``label``
        """
        search_root = self.dataset_root / test_type

        for img_path in search_root.rglob("*.png"):
            folder_name = img_path.parent.name.lower()
            label = "HC" if "healthy" in folder_name else "PD"

            yield {
                "path": str(img_path),
                "test_type": test_type,
                "label": label,
            }

    # ------------------------------------------------------------------
    def build(self) -> pl.DataFrame:
        """
        Build a Polars DataFrame for all images across all test types.

        Returns
        -------
        pl.DataFrame
            Columns: ``path``, ``test_type``, ``label``.
        """
        records: List[Dict[str, str]] = []

        for test_type in self.TEST_TYPES:
            records.extend(self._iter_directory(test_type))

        return pl.DataFrame(records)
