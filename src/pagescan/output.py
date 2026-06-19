"""PDF output generation."""

from __future__ import annotations

import io
import logging
from pathlib import Path

import cv2
import img2pdf
import numpy as np
from PIL import Image

from pagescan.config import ScanConfig

logger = logging.getLogger(__name__)


def save_pdf(image: np.ndarray, output_path: str, config: ScanConfig | None = None) -> None:
    """Save image as a single-page PDF.

    The image is the finished canvas produced by ``place_on_canvas`` — the
    corrected document centred on a white page of ``output_width`` x
    ``output_height`` pixels. It is JPEG-encoded and embedded losslessly; the
    physical PDF page size is derived from those pixel dimensions and
    ``output_dpi`` (e.g. 2480x3508 px @ 300 DPI -> A4, 2550x3300 @ 300 -> US
    Letter). The page therefore always matches the configured canvas, and
    img2pdf never distorts the aspect ratio.
    """
    if config is None:
        config = ScanConfig()

    if image.ndim == 2:
        pil_img = Image.fromarray(image, mode='L')
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

    buf = io.BytesIO()
    dpi = config.output_dpi
    pil_img.save(buf, format='JPEG', quality=config.jpeg_quality,
                 optimize=True, dpi=(dpi, dpi))
    buf.seek(0)

    # Page size follows the embedded image's DPI (set above), so the PDF page
    # matches the configured canvas instead of being forced to A4.
    pdf_bytes = img2pdf.convert(buf.read())

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_bytes(pdf_bytes)


def save_image(image: np.ndarray, output_path: str) -> None:
    """Save image as JPEG or PNG (inferred from extension)."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, image)
