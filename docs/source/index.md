---
sd_hide_title: true
---

# pagescan

```{image} https://img.shields.io/pypi/v/pagescan
:alt: PyPI
:target: https://pypi.org/project/pagescan/
```
```{image} https://img.shields.io/pypi/pyversions/pagescan
:alt: Python versions
:target: https://pypi.org/project/pagescan/
```
```{image} https://img.shields.io/badge/license-MIT-blue.svg
:alt: License: MIT
:target: https://github.com/7RPlus-GmbH/pagescan/blob/main/LICENSE
```

**A privacy-first document scanner for Python.** Phone photo in, deskewed print-ready PDF out — without uploading anything to a cloud.

```python
import pagescan

pagescan.scan("photo.jpg", "output.pdf")
```

That's it. No service account, no API key, no network round-trip. The pre-trained models (~50 MB) download from Hugging Face Hub on first use and cache locally.

## Why pagescan?

- **No cloud round-trip.** Everything runs locally; nothing leaves the machine.
- **Headless / scriptable.** No GUI, no mobile dependency, batch-friendly.
- **Open weights.** Hosted on [Hugging Face](https://huggingface.co/7rplus/pagescan-weights) under MIT-compatible licenses; auditable and fine-tunable.
- **Tuned for real-world phone photos.** Wood tables, uneven lighting, hand occlusion, perspective tilt.
- **Built for regulated environments.** EU data-residency requirements, on-prem deployments, scriptable pipelines.

If you need a mobile SDK, [Apple's VisionKit](https://developer.apple.com/documentation/visionkit) and [Google's ML Kit](https://developers.google.com/ml-kit/vision/document-scanner) are excellent. If you need server/desktop/headless and care about data residency, pagescan is built for that case.

## Get started

```{toctree}
:caption: User guide
:maxdepth: 1

installation
quickstart
architecture
config
benchmark
troubleshooting
```

```{toctree}
:caption: Reference
:maxdepth: 1

api
```

```{toctree}
:caption: Project
:maxdepth: 1

contributing
changelog
```

## Project status

pagescan is developed by [7R+ GmbH](https://7rplus.com). The first public release on PyPI is `0.1.0`. Until then, the API is stabilising — pin to an exact version if you depend on it from production code.
