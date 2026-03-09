"""Download and locate the Expedia Hotel Search dataset via kagglehub.

The dataset is cached locally by kagglehub, so repeated calls are free.
"""

import logging
from pathlib import Path

import kagglehub

from hotel_ranker.config import RAW_DATA_DIR

logger = logging.getLogger(__name__)

KAGGLE_DATASET = "vijeetnigam26/expedia-hotel"


def download_dataset() -> Path:
    """Download the Expedia Hotel dataset and return the path to its directory.

    Returns
    -------
    Path
        Directory containing the downloaded CSV file(s).
    """
    logger.info("Downloading dataset '%s' via kagglehub ...", KAGGLE_DATASET)
    path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    logger.info("Dataset available at: %s", path)
    return path


def find_csv(dataset_dir: Path | None = None) -> Path:
    """Return the path to the main CSV file inside the dataset directory.

    Parameters
    ----------
    dataset_dir : Path, optional
        If None, downloads the dataset first.

    Returns
    -------
    Path
        Path to the CSV file.

    Raises
    ------
    FileNotFoundError
        If no CSV file is found in the dataset directory.
    """
    if dataset_dir is None:
        dataset_dir = download_dataset()

    csv_files = list(dataset_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {dataset_dir}")

    # Pick the largest CSV (likely the main training file)
    target = max(csv_files, key=lambda p: p.stat().st_size)
    logger.info("Using CSV: %s (%.1f MB)", target.name, target.stat().st_size / 1e6)
    return target
