# SPDX-License-Identifier: Apache-2.0
"""Phase 3: CLI wiring."""

from __future__ import annotations

from ragwarden.cli import main


def test_no_command_prints_help() -> None:
    assert main([]) == 0


def test_benchmark_unknown_dataset_returns_error(capsys) -> None:
    rc = main(["benchmark", "--dataset", "nope", "--detector", "keyword_stub"])
    assert rc == 1
    assert "error" in capsys.readouterr().err.lower()


def test_calibrate_missing_file_returns_error() -> None:
    rc = main(["calibrate", "--dataset", "does-not-exist.jsonl", "--detector", "keyword_stub"])
    assert rc == 1


def test_warmup_healthy_detector_exits_zero(capsys) -> None:
    rc = main(["warmup", "--detector", "keyword_stub"])
    assert rc == 0
    assert "[OK" in capsys.readouterr().out


def test_warmup_broken_detector_exits_nonzero(capsys) -> None:
    rc = main(["warmup", "--detector", "lettucedetect"])
    assert rc == 1
    out = capsys.readouterr()
    assert "[FAIL" in out.out
    assert "warmup failed" in out.err
