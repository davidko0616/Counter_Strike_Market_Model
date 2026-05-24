from __future__ import annotations

import pandas as pd

from cs_market_model.models.lightgbm_train import _append_lightgbm_rank_blend


def test_lightgbm_rank_blend_is_appended_from_binary_and_multiclass_scores() -> None:
    predictions = pd.DataFrame(
        {
            "split_id": ["wf_00"] * 6,
            "event_id": ["a", "b", "c"] * 2,
            "model_name": ["lightgbm_binary"] * 3 + ["lightgbm_multiclass"] * 3,
            "label_code": [2, 0, 2] * 2,
            "predicted_label_code": [2, 0, 2] * 2,
            "strong_buy_score": [0.9, 0.2, 0.4, 0.8, 0.3, 0.5],
        }
    )

    with_ensemble = _append_lightgbm_rank_blend(predictions)
    ensemble = with_ensemble[with_ensemble["model_name"] == "lightgbm_rank_blend"]

    assert len(ensemble) == 3
    assert ensemble["strong_buy_score"].between(0, 1).all()
