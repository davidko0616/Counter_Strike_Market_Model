from __future__ import annotations

import pandas as pd

from cs_market_model.collectors.csfloat_batch import collect_csfloat_snapshots, load_universe_items


class FakeCSFloatClient:
    def __init__(self, fail_on: str | None = None) -> None:
        self.fail_on = fail_on
        self.calls: list[tuple[str, int]] = []

    def listings(self, market_hash_name: str, limit: int = 50):
        self.calls.append((market_hash_name, limit))
        if market_hash_name == self.fail_on:
            raise RuntimeError("simulated failure")
        return {"data": [{"price": 100, "item": {"float_value": 0.2}}]}


def test_load_universe_items_reads_day3_config() -> None:
    items = load_universe_items()

    assert len(items) == 48
    assert "AK-47 | Elite Build (Field-Tested)" in items


def test_collect_csfloat_snapshots_writes_raw_and_log(tmp_path) -> None:
    client = FakeCSFloatClient()
    log_output = tmp_path / "log.csv"
    result = collect_csfloat_snapshots(
        ["A", "B"],
        client=client,
        output_dir=tmp_path / "raw",
        log_output=log_output,
        limit=2,
        sleep_seconds=0.0,
    )

    log = pd.read_csv(log_output)
    assert result.requested_count == 2
    assert result.success_count == 2
    assert len(list((tmp_path / "raw").glob("*.json"))) == 2
    assert log["status"].tolist() == ["success", "success"]
    assert client.calls == [("A", 2), ("B", 2)]


def test_collect_csfloat_snapshots_logs_failures(tmp_path) -> None:
    client = FakeCSFloatClient(fail_on="B")
    result = collect_csfloat_snapshots(
        ["A", "B"],
        client=client,
        output_dir=tmp_path / "raw",
        log_output=tmp_path / "log.csv",
        sleep_seconds=0.0,
    )

    assert result.success_count == 1
    assert result.failure_count == 1
