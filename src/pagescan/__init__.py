"""pagescan - Turn phone photos of documents into clean, print-ready PDFs."""

from pagescan.config import (
    PRESET_A4_300,
    PRESET_FAST,
    PRESET_LETTER_300,
    PRESET_RAW,
    ScanConfig,
)
from pagescan.pipeline import scan, scan_batch

__version__ = "0.1.0"
__all__ = [
    "PRESET_A4_300",
    "PRESET_FAST",
    "PRESET_LETTER_300",
    "PRESET_RAW",
    "ScanConfig",
    "scan",
    "scan_batch",
]
