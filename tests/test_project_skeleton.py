from pathlib import Path

from cs_market_model.config import load_yaml_config
from cs_market_model.security.scan_secrets import scan_line


def test_expected_project_paths_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    expected_paths = [
        root / "pyproject.toml",
        root / ".env.example",
        root / "configs" / "venues.yaml",
        root / "configs" / "fees.yaml",
        root / "configs" / "backtest.yaml",
        root / "configs" / "universe_mvp.yaml",
        root / "configs" / "universe_day3_first_pull.yaml",
        root / "configs" / "universe_gun_skins_candidate.yaml",
        root / "docs" / "steamdt_kline_schema.md",
        root / "docs" / "csfloat_api_coverage.md",
        root / "src" / "cs_market_model" / "collectors" / "steamdt.py",
        root / "src" / "cs_market_model" / "collectors" / "steamdt_batch.py",
        root / "src" / "cs_market_model" / "collectors" / "csfloat.py",
        root / "src" / "cs_market_model" / "features" / "build_features.py",
        root / "src" / "cs_market_model" / "features" / "ranks.py",
        root / "src" / "cs_market_model" / "labeling" / "build_labels.py",
        root / "src" / "cs_market_model" / "labeling" / "sensitivity.py",
        root / "src" / "cs_market_model" / "models" / "train.py",
        root / "src" / "cs_market_model" / "models" / "lightgbm_train.py",
        root / "src" / "cs_market_model" / "models" / "calibrate.py",
        root / "src" / "cs_market_model" / "audit_steamdt_sample.py",
        root / "src" / "cs_market_model" / "audit_csfloat_sample.py",
        root / "src" / "cs_market_model" / "security" / "scan_secrets.py",
        root / "SECURITY.md",
    ]
    missing = [path for path in expected_paths if not path.exists()]
    assert not missing


def test_no_hardcoded_source_api_keys() -> None:
    root = Path(__file__).resolve().parents[1]
    source_text = "\n".join(
        path.read_text(encoding="utf-8") for path in (root / "src").rglob("*.py")
    )
    assert "STEAMDT_API_KEY" in source_text
    assert "CSFLOAT_API_KEY" in source_text
    assert "paste your" not in source_text.lower()
    assert "replace this" not in source_text.lower()


def test_secret_scanner_allows_universe_metadata_keys() -> None:
    findings = scan_line(
        Path("configs/universe_day3_first_pull.yaml"),
        1,
        "  include_in_day3_first_pull: true",
    )

    assert findings == []


def test_secret_scanner_allows_feature_identifiers() -> None:
    findings = scan_line(
        Path("src/cs_market_model/features/indices.py"),
        1,
        'features["category_median_return_1d"] = values',
    )

    assert findings == []


def test_fee_config_uses_steamdt_proxy_baseline_not_steam_marketplace_fee() -> None:
    fees = load_yaml_config("fees.yaml")

    assert fees["venues"]["steam"]["sell_fee_bps"] == 250
    assert fees["venues"]["youpin"]["sell_fee_bps"] == 100
    assert fees["venues"]["buff"]["sell_fee_bps"] == 250
