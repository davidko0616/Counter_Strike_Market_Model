from pathlib import Path


def test_expected_project_paths_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    expected_paths = [
        root / "pyproject.toml",
        root / ".env.example",
        root / "configs" / "venues.yaml",
        root / "configs" / "fees.yaml",
        root / "configs" / "backtest.yaml",
        root / "configs" / "universe_mvp.yaml",
        root / "src" / "cs_market_model" / "collectors" / "steamdt.py",
        root / "src" / "cs_market_model" / "collectors" / "csfloat.py",
        root / "src" / "cs_market_model" / "security" / "scan_secrets.py",
        root / "SECURITY.md",
    ]
    missing = [path for path in expected_paths if not path.exists()]
    assert not missing


def test_no_hardcoded_source_api_keys() -> None:
    root = Path(__file__).resolve().parents[1]
    source_text = "\n".join(path.read_text(encoding="utf-8") for path in (root / "src").rglob("*.py"))
    assert "STEAMDT_API_KEY" in source_text
    assert "CSFLOAT_API_KEY" in source_text
    assert "paste your" not in source_text.lower()
    assert "replace this" not in source_text.lower()
