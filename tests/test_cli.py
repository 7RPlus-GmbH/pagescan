"""Hermetic tests for pagescan.cli and pagescan.__main__ (no real processing)."""

import runpy
import sys

import pytest

from pagescan import cli
from pagescan.config import ScanConfig


def _set_argv(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["pagescan", *argv])


class TestSingleFile:
    def test_success_passes_flags_to_scan(self, monkeypatch, tmp_path, capsys):
        captured = {}

        def fake_scan(input_path, output_path, config=None):
            captured["input"] = input_path
            captured["output"] = output_path
            captured["config"] = config
            return {"success": True, "quality_score": 0.87,
                    "quality_passed": True, "message": "ok"}

        monkeypatch.setattr(cli, "scan", fake_scan)
        monkeypatch.setattr(cli, "scan_batch",
                            lambda *a, **k: pytest.fail("scan_batch should not run"))

        inp = str(tmp_path / "doc.jpg")
        _set_argv(monkeypatch, inp, "--no-ml", "--raw", "--quality", "80", "--no-deskew")
        cli.main()

        cfg = captured["config"]
        assert isinstance(cfg, ScanConfig)
        assert cfg.use_ml is False        # --no-ml
        assert cfg.enhance is False       # --raw
        assert cfg.shadow_removal is False
        assert cfg.white_balance is False
        assert cfg.jpeg_quality == 80     # --quality 80
        assert cfg.deskew is False        # --no-deskew
        assert captured["input"] == inp
        # default output: extension swapped to .pdf
        assert captured["output"].endswith(".pdf")
        out = capsys.readouterr().out
        assert "Saved:" in out
        assert "0.87" in out

    def test_explicit_output_path(self, monkeypatch, tmp_path):
        captured = {}
        monkeypatch.setattr(cli, "scan", lambda i, o, config=None: captured.update(
            out=o) or {"success": True, "quality_score": 1.0,
                       "quality_passed": True, "message": "ok"})
        inp = str(tmp_path / "doc.jpg")
        outp = str(tmp_path / "custom.pdf")
        _set_argv(monkeypatch, inp, outp)
        cli.main()
        assert captured["out"] == outp

    def test_default_flags(self, monkeypatch, tmp_path):
        captured = {}
        monkeypatch.setattr(cli, "scan", lambda i, o, config=None: captured.update(
            config=config) or {"success": True, "quality_score": 1.0,
                               "quality_passed": True, "message": "ok"})
        _set_argv(monkeypatch, str(tmp_path / "doc.jpg"))
        cli.main()
        cfg = captured["config"]
        assert cfg.use_ml is True
        assert cfg.enhance is True
        assert cfg.deskew is True
        assert cfg.auto_orient is True
        assert cfg.jpeg_quality == 50

    def test_failure_exits_1(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(cli, "scan", lambda i, o, config=None: {
            "success": False, "message": "cannot read"})
        _set_argv(monkeypatch, str(tmp_path / "doc.jpg"))
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 1
        assert "Failed" in capsys.readouterr().out


class TestBatch:
    def test_batch_success_exit_0(self, monkeypatch, tmp_path):
        captured = {}

        def fake_batch(input_dir, output_dir, config=None, workers=None):
            captured["input_dir"] = input_dir
            captured["workers"] = workers
            return {"processed": 3, "failed": 0, "low_quality": 0}

        monkeypatch.setattr(cli, "scan_batch", fake_batch)
        monkeypatch.setattr(cli, "scan",
                            lambda *a, **k: pytest.fail("scan should not run"))
        _set_argv(monkeypatch, "--batch", "--input-dir", str(tmp_path),
                  "--workers", "2")
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 0
        assert captured["input_dir"] == str(tmp_path)
        assert captured["workers"] == 2

    def test_batch_with_failures_exit_1(self, monkeypatch, tmp_path):
        monkeypatch.setattr(cli, "scan_batch", lambda *a, **k: {
            "processed": 2, "failed": 1, "low_quality": 0})
        _set_argv(monkeypatch, "--batch", "--input-dir", str(tmp_path))
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 1


class TestNoArgs:
    def test_no_args_prints_help_and_exits_1(self, monkeypatch, capsys):
        _set_argv(monkeypatch)
        with pytest.raises(SystemExit) as exc:
            cli.main()
        assert exc.value.code == 1
        out = capsys.readouterr().out
        assert "usage" in out.lower() or "pagescan" in out.lower()


class TestMainModule:
    def test_running_module_invokes_main(self, monkeypatch):
        called = {"n": 0}
        monkeypatch.setattr(cli, "main", lambda: called.update(n=called["n"] + 1))
        runpy.run_module("pagescan", run_name="__main__")
        assert called["n"] == 1
