"""pagescan - Turn phone photos of documents into clean, print-ready PDFs."""

from pagescan.config import ScanConfig
from pagescan.pipeline import scan, scan_batch

__version__ = "0.1.0"
__all__ = ["scan", "scan_batch", "ScanConfig"]
