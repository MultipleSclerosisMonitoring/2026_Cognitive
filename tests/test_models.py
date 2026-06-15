import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration.models.algoritmos import get_model


def test_stacking_ensemble_model_trains_and_predicts():
    X = pd.DataFrame({
        "f1": np.linspace(0.0, 1.0, 18),
        "f2": np.linspace(1.0, 2.0, 18),
        "f3": np.sin(np.linspace(0.0, 3.0, 18)),
    })
    y = pd.Series(10.0 + 2.5 * X["f1"] + 0.8 * X["f2"] + X["f3"], name="target")
    cv_folds = [
        (np.arange(6, 18), np.arange(0, 6)),
        (np.r_[0:6, 12:18], np.arange(6, 12)),
        (np.arange(0, 12), np.arange(12, 18)),
    ]

    model = get_model("ensemble")
    model._make_base_estimators = lambda multi_output: {
        "ridge": Ridge(alpha=1.0, random_state=42),
        "rf": RandomForestRegressor(n_estimators=20, max_depth=4, random_state=42, n_jobs=1),
        "xgboost": xgb.XGBRegressor(
            n_estimators=12,
            max_depth=2,
            learning_rate=0.1,
            subsample=1.0,
            colsample_bytree=1.0,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1,
        ),
    }
    model.train(X=X, y=y, cv_folds=cv_folds)
    predictions = model.predict(X.iloc[:4].copy())

    assert len(predictions) == 4
    assert np.isfinite(predictions).all()


def test_error_classifier_model_trains_and_predicts_sparse_errors():
    X = pd.DataFrame({
        "f1": np.linspace(0.0, 1.0, 20),
        "f2": np.linspace(1.0, 2.0, 20),
    })
    y = pd.Series([0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0, 1, 0, 0, 4, 0, 0, 2, 0], name="errors")
    cv_folds = [
        (np.r_[5:20], np.arange(0, 5)),
        (np.r_[0:5, 10:20], np.arange(5, 10)),
        (np.arange(0, 15), np.arange(15, 20)),
    ]

    model = get_model("error_classifier")
    model.train(X=X, y=y, cv_folds=cv_folds)
    predictions = model.predict(X.iloc[:6].copy())

    assert len(predictions) == 6
    assert np.isfinite(predictions).all()
    assert (predictions >= 0).all()
    assert set(np.unique(predictions)).issubset({0.0, 1.0, 2.0, 4.0})


def test_error_classifier_uses_binary_encoding_for_errors_a():
    X = pd.DataFrame({
        "f1": np.linspace(0.0, 1.0, 12),
        "f2": np.linspace(1.0, 2.0, 12),
    })
    y = pd.Series([0, 0, 1, 0, 2, 0, 1, 0, 0, 2, 0, 1], name="tmt_paper_errors_a")

    model = get_model("error_classifier")
    model.train(X=X, y=y, cv_folds=3)
    predictions = model.predict(X.iloc[:5].copy())

    assert len(predictions) == 5
    assert np.isfinite(predictions).all()
    assert set(np.unique(predictions)).issubset({0.0, 1.0, 2.0})


def test_spline_gam_model_trains_and_predicts_nonlinear_signal():
    X = pd.DataFrame({
        "f1": np.linspace(-1.0, 1.0, 24),
        "f2": np.linspace(0.0, 2.0, 24),
        "binary_flag": ([0.0, 1.0] * 12),
    })
    y = pd.Series(
        20.0 + 4.0 * (X["f1"] ** 2) + 1.5 * X["f2"] - 0.8 * X["binary_flag"],
        name="target",
    )
    cv_folds = [
        (np.arange(8, 24), np.arange(0, 8)),
        (np.r_[0:8, 16:24], np.arange(8, 16)),
        (np.arange(0, 16), np.arange(16, 24)),
    ]

    model = get_model("gam", n_knots=4, degree=3, alpha=0.5, min_unique_for_spline=5)
    model.train(X=X, y=y, cv_folds=cv_folds)
    predictions = model.predict(X.iloc[:5].copy())

    assert len(predictions) == 5
    assert np.isfinite(predictions).all()
    assert model.metadata["spline_columns"] == ["f1", "f2"]
    assert model.metadata["linear_columns"] == ["binary_flag"]
