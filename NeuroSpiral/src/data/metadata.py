import polars as pl
from pathlib import Path
from typing import Generator, Union, List, Dict


class MetadataBuilder:
    """
    Scans the Parkinson's drawing dataset directory and builds a structured
    Polars DataFrame of image paths, type tests, and class labels.

    Supported test types : ``spiral``, ``wave``
    Labels               : ``HC`` (Healthy Control), ``PD`` (Parkinson's Disease)
    """

    TYPES_TEST: List[str] = ["spiral", "wave"]

    def __init__(self, dataset_root: Union[str, Path]) -> None:
        """
        Parameters
        ----------
        dataset_root : str | Path
            Root directory of the parkinsons-drawings dataset.
        """
        self.dataset_root = Path(dataset_root)

    # ------------------------------------------------------------------
    def _iter_directory(self, type_test: str) -> Generator[Dict[str, str], None, None]:
        """
        Recursively scan one test-type sub-directory and yield metadata dicts.

        Parameters
        ----------
        type_test : str
            ``"spiral"`` or ``"wave"``.

        Yields
        ------
        dict
            Keys: ``path``, ``type_test``, ``label``
        """
        search_root = self.dataset_root / type_test

        for img_path in search_root.rglob("*.png"):
            folder_name = img_path.parent.name.lower()
            label = "HC" if "healthy" in folder_name else "PD"

            yield {
                "path": str(img_path),
                "type_test": type_test,
                "label": label,
            }

    def build(self) -> pl.DataFrame:
        """
        Build a Polars DataFrame for all images across all test types.

        Returns
        -------
        pl.DataFrame
            Columns: ``path``, ``type_test``, ``label``.
        """
        records: List[Dict[str, str]] = []

        for type_test in self.TYPES_TEST:
            records.extend(self._iter_directory(type_test))

        return pl.DataFrame(records)
