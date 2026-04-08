"""PDF output generation."""

import io
import logging
from pathlib import Path

import cv2
import img2pdf
import numpy as np
from PIL import Image

from pagescan.config import ScanConfig

logger = logging.getLogger(__name__)


def save_pdf(image: np.ndarray, output_path: str, config: ScanConfig = None) -> None:
    """Save image as a single-page PDF.

    Encodes as JPEG and wraps in a PDF sized to the configured page
    dimensions (default A4).
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

    a4_mm = (img2pdf.mm_to_pt(210), img2pdf.mm_to_pt(297))
    layout = img2pdf.get_layout_fun(a4_mm)
    pdf_bytes = img2pdf.convert(buf.read(), layout_fun=layout)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_bytes(pdf_bytes)


def save_image(image: np.ndarray, output_path: str) -> None:
    """Save image as JPEG or PNG (inferred from extension)."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, image)
