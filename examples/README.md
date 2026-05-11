# Examples

End-to-end runnable notebooks for pagescan.

| Notebook | Demonstrates |
|---|---|
| [`quickstart.ipynb`](quickstart.ipynb) | Single scan, batch processing, `ScanConfig` presets, debug visualisation |

## Running the notebooks

```bash
pip install "pagescan[ml]" jupyter matplotlib
jupyter notebook examples/
```

All notebooks generate their own demo images synthetically — you don't need to download anything. The pre-trained pagescan model weights (~50 MB) download from Hugging Face on first run and cache locally.

## What to scan

Once you've worked through the quickstart with the synthetic image, swap in a real photo:

```python
result = pagescan.scan("path/to/your/photo.jpg", "out.pdf")
```

Good test photos: any document on a non-white background (wood, table, fabric), shot from above with a phone. The model handles mild perspective tilt, uneven lighting, and partial hand occlusion well.
