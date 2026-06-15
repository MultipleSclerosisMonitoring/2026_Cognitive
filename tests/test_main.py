import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import calibration.main as calibration_main


class _DummyProcessor:
    pass


def test_stratified_routing_uses_subgroup_specific_models(monkeypatch):
    X_train = pd.DataFrame({"signal": [1.0, 1.0, 8.0, 8.0]})
    y_train = pd.Series([10.0, 11.0, 20.0, 21.0], name="target")
    groups_train = pd.Series(["S1", "S2", "U1", "U2"])
    metadata_train = pd.DataFrame({"education_band": ["Secondary", "Secondary", "University", "University"]})

    X_test = pd.DataFrame({"signal": [1.5, 8.5]})
    metadata_test = pd.DataFrame({"education_band": ["Secondary", "University"]})

    def fake_fit_predict(**kwargs):
        subgroup_signal = float(kwargs["X_train"]["signal"].iloc[0])
        preds = np.full(len(kwargs["X_test"]), subgroup_signal, dtype=float)
        return preds, None, None

    monkeypatch.setattr(calibration_main, "_fit_predict_with_optional_conformal", fake_fit_predict)

    predictions, low, high = calibration_main._fit_predict_with_optional_stratification(
        processor=_DummyProcessor(),
        model_type="rf",
        multi_output=False,
        X_train=X_train,
        y_train=y_train,
        groups_train=groups_train,
        metadata_train=metadata_train,
        X_test=X_test,
        metadata_test=metadata_test,
        cv_folds_num=3,
        param_grid=None,
        search_strategy="grid",
        n_iter=10,
        uncertainty_cfg={"enabled": False},
        stratify_by="education_band",
        min_train_samples_per_stratum=2,
        min_unique_groups_per_stratum=2,
    )

    assert np.allclose(predictions, [1.0, 8.0])
    assert low is None
    assert high is None


def test_stratified_routing_falls_back_to_global_when_subgroup_is_small(monkeypatch):
    X_train = pd.DataFrame({"signal": [1.0, 1.0, 8.0]})
    y_train = pd.Series([10.0, 11.0, 20.0], name="target")
    groups_train = pd.Series(["S1", "S2", "U1"])
    metadata_train = pd.DataFrame({"education_band": ["Secondary", "Secondary", "University"]})

    X_test = pd.DataFrame({"signal": [1.5, 8.5]})
    metadata_test = pd.DataFrame({"education_band": ["Secondary", "University"]})

    def fake_fit_predict(**kwargs):
        subgroup_values = kwargs["X_train"]["signal"].to_numpy(dtype=float)
        fill_value = float(np.mean(subgroup_values))
        preds = np.full(len(kwargs["X_test"]), fill_value, dtype=float)
        return preds, None, None

    monkeypatch.setattr(calibration_main, "_fit_predict_with_optional_conformal", fake_fit_predict)

    predictions, _, _ = calibration_main._fit_predict_with_optional_stratification(
        processor=_DummyProcessor(),
        model_type="rf",
        multi_output=False,
        X_train=X_train,
        y_train=y_train,
        groups_train=groups_train,
        metadata_train=metadata_train,
        X_test=X_test,
        metadata_test=metadata_test,
        cv_folds_num=3,
        param_grid=None,
        search_strategy="grid",
        n_iter=10,
        uncertainty_cfg={"enabled": False},
        stratify_by="education_band",
        min_train_samples_per_stratum=2,
        min_unique_groups_per_stratum=2,
    )

    assert np.allclose(predictions, [1.0, 10.0 / 3.0])
