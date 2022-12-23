from __future__ import annotations

import logging
from pathlib import Path

LOG = logging.getLogger(__name__)


def log_dataset_export_info(path: Path | None) -> None:
    LOG.info("Dataset has been exported to: %s", path)
