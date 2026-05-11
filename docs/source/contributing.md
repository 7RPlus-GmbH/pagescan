# Contributing

pagescan is an open-source product of [7R+ GmbH](https://7rplus.com). Bug reports, fixes, and discussion are all welcome.

## Quick orientation

| Directory | What's there |
|---|---|
| `src/pagescan/` | The package — what gets shipped to PyPI. |
| `tests/` | Test suite. Run with `pytest`. |
| `benchmark/` | Real-data benchmarks. Production harness is `benchmark/comparison/`; `benchmark/ceiling/` is a development reference with GT bboxes. |
| `training/yolo/` | YOLO11 detector training pipeline. |
| `docs/` | This documentation site (Sphinx + MyST + pydata-sphinx-theme). |
| `data/model/` | Local model-weight cache (gitignored). |
| `_archive/` | Failed-approach reference. Not part of the product — don't import from it. |

## Dev setup

```bash
git clone https://github.com/7RPlus-GmbH/pagescan.git
cd pagescan
pip install -e ".[dev,ml,docs]"
```

The `[dev]` extras pull in the test and lint toolchain; `[ml]` enables the cascade; `[docs]` builds this site.

## Quality gates

Before opening a PR, all three must be clean:

```bash
ruff check src tests             # lint
mypy src/pagescan                # type-check
pytest -q                        # tests
```

The same gates run in CI on every PR via [`.github/workflows/ci.yml`](https://github.com/7RPlus-GmbH/pagescan/blob/main/.github/workflows/ci.yml) — pytest matrix on Python 3.9–3.13.

## Building the docs locally

```bash
cd docs
make html
make serve     # opens http://localhost:8000
```

`make linkcheck` validates every external link in the docs. Run it before any PR that touches the documentation.

## Before opening an issue

- Search [existing issues](https://github.com/7RPlus-GmbH/pagescan/issues).
- For bug reports, include: pagescan version, Python version, OS, a minimal reproducer, and the *input image* if possible (or a description of the document type, lighting, and background).

## Before opening a pull request

- Discuss large changes in an issue first. The 0.1.0 release plan is opinionated about scope.
- Quality gates clean (`ruff` + `mypy` + `pytest`).
- Update `CHANGELOG.md` under `[Unreleased]` for user-facing changes.
- Keep PRs focused. One change per PR.

## What we're looking for help with

The roadmap is opinionated, but the following are always welcome:

- **Labeled training data** in our failure distributions — magazines, white-on-cream paper, hand-occluded scenes. Even 20–50 labeled photos help.
- Bug reports with attached images pagescan scans poorly.
- Documentation improvements — especially examples and tutorials.
- Benchmark results from real workflows.

## What we're not doing in 0.1.0

- Mobile / on-device SDK.
- Cloud-hosted scanning service.
- Receipt / non-rectangular document support.
- Languages beyond German / English.

These are out of scope for 0.1.0 and may be considered for later versions.

## License & legal

By contributing, you agree your contributions will be licensed under the [MIT License](https://github.com/7RPlus-GmbH/pagescan/blob/main/LICENSE).

If you submit a labeled image as part of a bug report or contribution, you must own the rights or have permission to share it under MIT terms.

## Contact

For non-technical questions: `contact@7rplus.com`.
