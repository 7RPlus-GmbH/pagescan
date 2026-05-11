# Contributing to pagescan

pagescan is an open-source product of [7r+ GmbH](https://7rplus.com). We
welcome bug reports, fixes, and discussion. Before opening a PR, please read
through.

## Quick orientation

- **The package** lives in `src/pagescan/`. That's what gets shipped to PyPI.
- **The benchmark** lives in `benchmark/`. The production benchmark is `benchmark/cascade/`; `benchmark/ceiling/` is a development reference (cascade upper bound with GT bboxes).
- **The training pipeline** lives in `training/yolo/`. Currently a YOLO11n single-class document detector. See `training/yolo/README.md`.
- **Tests** live in `tests/`. Run with `pytest`.
- **Anything in `_archive/`** is not part of the product. Don't import from it. See `_archive/README.md`.

## Before opening an issue

- Check the [docs](https://7rplus-gmbh.github.io/pagescan/) and the [README](README.md).
- Search existing issues to avoid duplicates.
- For bug reports, include: pagescan version, Python version, OS, a minimal reproducer, and the *input image* if possible (or a description of the document type, lighting, background).

## Before opening a pull request

- Discuss large architectural changes in an issue first. The 0.1.0 scope is
  opinionated.
- Run all three quality gates locally — same as CI:
  - `pytest -q`
  - `mypy src/pagescan`
  - `ruff check src tests`
- Update `CHANGELOG.md` under `[Unreleased]` if user-facing.
- Keep PRs focused. One change per PR.

## What we're looking for help with

- **Labeled training data** in the failure distributions (magazines, white-on-cream, hand-occluded). Even 20–50 labeled photos help.
- Bug reports with attached images that pagescan currently scans poorly.
- Documentation improvements, especially examples and tutorials.
- Benchmark results from real workflows.

## What we're not doing in 0.1.0

- Mobile / on-device SDK.
- Cloud-hosted scanning service.
- Receipt / non-rectangular document support.
- Languages beyond German / English.

These are out of scope for 0.1.0 but may be considered for later versions.

## License & legal

By contributing, you agree your contributions will be licensed under the
[MIT License](LICENSE).

If you submit a labeled image as part of a bug report or contribution, you
must own the rights or have permission to share it under MIT terms.

## Contact

For non-technical questions: contact@7rplus.com.
