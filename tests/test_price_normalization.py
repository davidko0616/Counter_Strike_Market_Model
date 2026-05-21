from cs_market_model.normalization.prices import (
    parse_steamdt_kline_payload,
    unwrap_steamdt_kline_payload,
)


def test_parse_steamdt_kline_payload_maps_ohlc_columns() -> None:
    payload = {
        "success": True,
        "data": [
            ["1768233600", 332.13, 330.34, 356.38, 325.28],
            ["1768320000", 330.34, 342.42, 350.52, 330.34],
        ],
    }

    bars = parse_steamdt_kline_payload(payload, "AK-47 | Redline (Field-Tested)")

    assert list(bars[["open", "close", "high", "low"]].iloc[0]) == [
        332.13,
        330.34,
        356.38,
        325.28,
    ]
    assert bars["timestamp"].is_monotonic_increasing
    assert bars["volume"].isna().all()
    assert bars["data_resolution"].eq("1d").all()


def test_unwrap_steamdt_kline_payload_accepts_saved_raw_envelope() -> None:
    raw = {"success": True, "data": [["1768233600", 1, 2, 3, 1]]}
    envelope = {"source": "steamdt", "raw_json": raw}

    assert unwrap_steamdt_kline_payload(envelope) == raw
    assert unwrap_steamdt_kline_payload(raw) == raw
