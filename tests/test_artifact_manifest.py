from __future__ import annotations

import pandas as pd

from cs_market_model.pipeline.artifact_manifest import (
    artifact_manifest_row,
    build_artifact_manifest,
)


def test_artifact_manifest_hashes_existing_csv(tmp_path) -> None:
    path = tmp_path / "artifact.csv"
    path.write_text("a\n1\n2\n", encoding="utf-8")

    row = artifact_manifest_row(path)

    assert row["exists"] is True
    assert row["rows"] == 2
    assert len(str(row["sha256"])) == 64


def test_artifact_manifest_reports_missing_path(tmp_path) -> None:
    path = tmp_path / "missing.csv"

    manifest = build_artifact_manifest([path])

    assert isinstance(manifest, pd.DataFrame)
    assert not bool(manifest.loc[0, "exists"])
    assert manifest.loc[0, "rows"] == 0
