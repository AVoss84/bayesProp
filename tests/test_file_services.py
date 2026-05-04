"""Unit tests for bayesAB.services.file module."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest
import yaml

from bayesAB.services.file import CSVService, JSONService, YAMLService


@pytest.fixture
def tmp_dir(tmp_path: Path) -> str:
    """Return a temporary directory path as string."""
    return str(tmp_path)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Return a small sample DataFrame."""
    return pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})


# ── CSVService ────────────────────────────────────────────────


class TestCSVService:
    """Tests for CSVService."""

    def test_write_and_read_roundtrip(self, tmp_dir: str, sample_df: pd.DataFrame) -> None:
        svc = CSVService(path="test.csv", root_path=tmp_dir, delimiter=",")
        svc.doWrite(sample_df)
        result = svc.doRead()
        pd.testing.assert_frame_equal(result, sample_df)

    def test_read_with_schema_map(self, tmp_dir: str, sample_df: pd.DataFrame) -> None:
        svc_write = CSVService(path="test.csv", root_path=tmp_dir, delimiter=",")
        svc_write.doWrite(sample_df)

        svc_read = CSVService(
            path="test.csv",
            root_path=tmp_dir,
            delimiter=",",
            schema_map={"a": "x", "b": "y"},
        )
        result = svc_read.doRead()
        assert list(result.columns) == ["x", "y"]

    def test_read_missing_file_returns_empty_df(self, tmp_dir: str) -> None:
        svc = CSVService(path="nonexistent.csv", root_path=tmp_dir)
        result = svc.doRead()
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_verbose_prints(self, tmp_dir: str, sample_df: pd.DataFrame, capsys: pytest.CaptureFixture[str]) -> None:
        svc = CSVService(path="test.csv", root_path=tmp_dir, delimiter=",", verbose=True)
        svc.doWrite(sample_df)
        svc.doRead()
        captured = capsys.readouterr()
        assert "CSV Service output to file" in captured.out
        assert "CSV Service read from file" in captured.out

    def test_creates_root_dir_if_missing(self, tmp_dir: str) -> None:
        nested = os.path.join(tmp_dir, "sub", "dir")
        CSVService(path="test.csv", root_path=nested)
        assert os.path.isdir(nested)

    def test_tab_delimiter_default(self, tmp_dir: str, sample_df: pd.DataFrame) -> None:
        svc = CSVService(path="test.tsv", root_path=tmp_dir)
        svc.doWrite(sample_df)
        result = svc.doRead()
        pd.testing.assert_frame_equal(result, sample_df)


# ── YAMLService ───────────────────────────────────────────────


class TestYAMLService:
    """Tests for YAMLService."""

    def test_write_and_read_roundtrip(self, tmp_dir: str) -> None:
        data = {"key": "value", "nested": {"a": 1, "b": [1, 2, 3]}}
        svc = YAMLService(path="test.yaml", root_path=tmp_dir)
        svc.doWrite(data)
        result = svc.doRead()
        assert result == data

    def test_write_and_read_list(self, tmp_dir: str) -> None:
        data = [{"a": 1}, {"b": 2}]
        svc = YAMLService(path="test.yaml", root_path=tmp_dir)
        svc.doWrite(data)
        result = svc.doRead()
        assert result == data

    def test_verbose_prints(self, tmp_dir: str, capsys: pytest.CaptureFixture[str]) -> None:
        data = {"x": 1}
        svc = YAMLService(path="test.yaml", root_path=tmp_dir, verbose=True)
        svc.doWrite(data)
        svc.doRead()
        captured = capsys.readouterr()
        assert "Write to:" in captured.out
        assert "Read:" in captured.out

    def test_read_invalid_yaml_raises(self, tmp_dir: str) -> None:
        fpath = os.path.join(tmp_dir, "bad.yaml")
        with open(fpath, "w") as f:
            f.write(":\n  :\n    - :\n      bad: [unterminated")
        svc = YAMLService(path="bad.yaml", root_path=tmp_dir)
        with pytest.raises(yaml.YAMLError):
            svc.doRead()


# ── JSONService ───────────────────────────────────────────────


class TestJSONService:
    """Tests for JSONService."""

    def test_write_and_read_roundtrip(self, tmp_dir: str) -> None:
        data = {"key": "value", "numbers": [1, 2, 3]}
        svc = JSONService(path="test.json", root_path=tmp_dir, verbose=False)
        svc.doWrite(data)
        result = svc.doRead()
        assert result == data

    def test_read_empty_file_returns_empty_dict(self, tmp_dir: str) -> None:
        fpath = os.path.join(tmp_dir, "empty.json")
        Path(fpath).touch()
        svc = JSONService(path="empty.json", root_path=tmp_dir, verbose=False)
        result = svc.doRead()
        assert result == {}

    def test_read_invalid_json_returns_empty_dict(self, tmp_dir: str) -> None:
        fpath = os.path.join(tmp_dir, "bad.json")
        with open(fpath, "w") as f:
            f.write("{not valid json")
        svc = JSONService(path="bad.json", root_path=tmp_dir, verbose=False)
        result = svc.doRead()
        assert result == {}

    def test_verbose_prints(self, tmp_dir: str, capsys: pytest.CaptureFixture[str]) -> None:
        data = {"x": 1}
        svc = JSONService(path="test.json", root_path=tmp_dir, verbose=True)
        svc.doWrite(data)
        svc.doRead()
        captured = capsys.readouterr()
        assert "JSON Service Output to File" in captured.out
        assert "Read:" in captured.out

    def test_unicode_roundtrip(self, tmp_dir: str) -> None:
        data = {"emoji": "🎉", "german": "Ärger mit Übung"}
        svc = JSONService(path="unicode.json", root_path=tmp_dir, verbose=False)
        svc.doWrite(data)
        result = svc.doRead()
        assert result == data

    def test_write_produces_pretty_json(self, tmp_dir: str) -> None:
        data = {"a": 1}
        svc = JSONService(path="test.json", root_path=tmp_dir, verbose=False)
        svc.doWrite(data)
        with open(os.path.join(tmp_dir, "test.json")) as f:
            raw = f.read()
        assert "\n" in raw  # indented, not single-line
        assert json.loads(raw) == data
