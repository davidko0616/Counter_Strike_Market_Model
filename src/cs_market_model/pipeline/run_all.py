"""One-command orchestration for the conservative paper-trade pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineStep:
    """One module invocation in the reproducible pipeline."""

    name: str
    module: str
    args: tuple[str, ...] = ()
    requires_api: bool = False

    def command(self) -> list[str]:
        return [sys.executable, "-m", self.module, *self.args]


PIPELINE_STEPS = (
    PipelineStep(
        name="collect_csfloat_snapshots",
        module="cs_market_model.collectors.csfloat_batch",
        args=("--config", "universe_mvp_v2.yaml"),
        requires_api=True,
    ),
    PipelineStep(name="build_csfloat_features", module="cs_market_model.features.csfloat_listings"),
    PipelineStep(
        name="csfloat_coverage",
        module="cs_market_model.research.day19_csfloat_coverage",
    ),
    PipelineStep(
        name="csfloat_ablation_setup",
        module="cs_market_model.research.day18_csfloat_ablation",
    ),
    PipelineStep(
        name="paper_trade_rejection",
        module="cs_market_model.research.day25_paper_trade_rejection",
    ),
    PipelineStep(name="signal_explanations", module="cs_market_model.models.explain"),
    PipelineStep(
        name="paper_trade_ab_test",
        module="cs_market_model.research.paper_trade_ab_test",
    ),
    PipelineStep(
        name="model_health_monitor",
        module="cs_market_model.research.model_health_monitor",
    ),
    PipelineStep(
        name="artifact_manifest",
        module="cs_market_model.pipeline.artifact_manifest",
    ),
)


def selected_pipeline_steps(
    *,
    include_collectors: bool = False,
) -> list[PipelineStep]:
    """Return steps for the local reproducible pipeline."""
    return [
        step
        for step in PIPELINE_STEPS
        if include_collectors or not step.requires_api
    ]


def run_pipeline(
    *,
    include_collectors: bool = False,
    dry_run: bool = False,
) -> list[dict[str, str]]:
    """Run or describe the conservative paper-trade pipeline."""
    rows: list[dict[str, str]] = []
    for step in selected_pipeline_steps(include_collectors=include_collectors):
        command = step.command()
        rows.append({"step": step.name, "command": " ".join(command), "status": "planned"})
        if dry_run:
            continue
        completed = subprocess.run(command, check=False)
        rows[-1]["status"] = "passed" if completed.returncode == 0 else "failed"
        rows[-1]["returncode"] = str(completed.returncode)
        if completed.returncode != 0:
            raise SystemExit(completed.returncode)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the conservative paper-trade pipeline.")
    parser.add_argument(
        "--include-collectors",
        action="store_true",
        help="Also run API-backed collectors. Requires configured credentials.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without executing them.",
    )
    args = parser.parse_args()

    rows = run_pipeline(
        include_collectors=args.include_collectors,
        dry_run=args.dry_run,
    )
    for row in rows:
        print(f"{row['status']}: {row['step']} -> {row['command']}")


if __name__ == "__main__":
    main()
