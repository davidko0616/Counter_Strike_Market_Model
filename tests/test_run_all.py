from __future__ import annotations

from cs_market_model.pipeline.run_all import run_pipeline, selected_pipeline_steps


def test_selected_pipeline_steps_skip_api_collectors_by_default() -> None:
    steps = selected_pipeline_steps()

    assert all(not step.requires_api for step in steps)
    assert "collect_csfloat_snapshots" not in {step.name for step in steps}


def test_run_pipeline_dry_run_returns_planned_commands() -> None:
    rows = run_pipeline(dry_run=True)

    assert rows
    assert rows[-1]["step"] == "artifact_manifest"
    assert rows[-1]["status"] == "planned"
