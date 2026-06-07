import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration.data.loader import DataProcessor


def test_resolve_tmt_target_columns_uses_real_paper_columns():
    df = pd.DataFrame({
        "Orden papel/digital": ["papel", "digital"],
        "DIA 1 TMT papel (errores) A": [0, 1],
        "DIA 1 TMT papel (errores) B": [2, 0],
        "DIA 1 TMT papel (tiempo) A": [31.2, 35.1],
        "DIA 1 TMT papel (tiempo) B": [78.0, 82.4],
    })

    config = DataProcessor._default_column_config()["tmt"]
    resolved = DataProcessor._resolve_target_columns(df, config)

    assert resolved == {
        "tmt_paper_score_a": "DIA 1 TMT papel (tiempo) A",
        "tmt_paper_score_b": "DIA 1 TMT papel (tiempo) B",
        "tmt_paper_errors_a": "DIA 1 TMT papel (errores) A",
        "tmt_paper_errors_b": "DIA 1 TMT papel (errores) B",
    }


def test_add_clinical_covariates_derives_age_sex_group_and_delta():
    df = pd.DataFrame({
        "Fecha nacimiento": ["1980-01-01", "1990-06-15"],
        "dia1 monització  fecha": ["2026-01-01", "2026-06-15"],
        "ts_created": ["2025-12-30", "2026-06-10"],
        "Sexo": ["hombre", "mujer"],
        "clinical_source_sheet": ["EM", "Controles"],
    })

    enriched = DataProcessor._add_clinical_covariates(df)

    assert "age_at_test" in enriched.columns
    assert enriched["age_at_test"].notna().all()
    assert "age_band" in enriched.columns
    assert set(enriched["age_band"].dropna().astype(str)).issubset({"<40", "40-54", "55+"})
    assert "sex_binary" in enriched.columns
    assert enriched["sex_binary"].tolist() == [0.0, 1.0]
    assert "clinical_group" in enriched.columns
    assert enriched["clinical_group"].tolist() == ["EM", "Controles"]
    assert "delta_dias_digital_papel" in enriched.columns
    assert enriched["delta_dias_digital_papel"].tolist() == [2.0, 5.0]


def test_prepare_splits_respects_patient_groups_and_train_only_imputation():
    processor = DataProcessor("sqlite://", "unused.xlsx")
    processor.digital_source_columns = {"patient_id", "patient_id_base", "codeid", "feature_a", "feature_b"}
    df = pd.DataFrame({
        "patient_id": ["P1", "P1", "P2", "P2", "P3", "P3", "P4", "P4", "P5", "P5"],
        "patient_id_base": ["P1", "P1", "P2", "P2", "P3", "P3", "P4", "P4", "P5", "P5"],
        "codeid": ["P1", "P1", "P2", "P2", "P3", "P3", "P4", "P4", "P5", "P5"],
        "feature_a": [1.0, None, 3.0, None, 5.0, None, 7.0, None, 9.0, None],
        "feature_b": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        "Día 1 SDMT Papel score": [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
        "Fecha nacimiento": [
            "1980-01-01", "1981-01-01", "1982-01-01", "1983-01-01", "1984-01-01",
            "1985-01-01", "1986-01-01", "1987-01-01", "1988-01-01", "1989-01-01",
        ],
        "dia1 monització  fecha": ["2026-01-01"] * 10,
        "ts_created": ["2025-12-28"] * 10,
        "Sexo": ["hombre", "hombre", "mujer", "mujer", "hombre", "hombre", "mujer", "mujer", "hombre", "hombre"],
        "clinical_source_sheet": ["EM", "EM", "Controles", "Controles", "EM", "EM", "Controles", "Controles", "EM", "EM"],
    })

    X_train, X_test, y_train, y_test, groups_train, groups_test, metadata_train, metadata_test, _ = processor.prepare_splits(df, "sdmt", test_size=0.2)

    assert not X_train.isna().any().any()
    assert not X_test.isna().any().any()
    train_groups = set(groups_train)
    test_groups = set(groups_test)
    assert train_groups
    assert test_groups
    assert train_groups.isdisjoint(test_groups)
    assert len(X_train) + len(X_test) == len(df)
    assert len(y_train) + len(y_test) == len(df)
    assert "clinical_group" in metadata_test.columns


def test_prepare_splits_keeps_only_digital_features_and_stable_covariates():
    processor = DataProcessor("sqlite://", "unused.xlsx")
    processor.digital_source_columns = {"patient_id", "codeid", "digital_speed", "digital_errors"}
    df = pd.DataFrame({
        "patient_id": [f"P{i}" for i in range(1, 11)],
        "codeid": [f"P{i}" for i in range(1, 11)],
        "digital_speed": [10.0 + i for i in range(10)],
        "digital_errors": [i % 3 for i in range(10)],
        "Día 1 SDMT Papel score": [40 + i for i in range(10)],
        "Día 3 M-FIS score fis": [5 + i for i in range(10)],
        "DIA 1 TMT papel (tiempo) A": [20 + i for i in range(10)],
        "Fecha nacimiento": [
            "1980-01-01", "1981-01-01", "1982-01-01", "1983-01-01", "1984-01-01",
            "1985-01-01", "1986-01-01", "1987-01-01", "1988-01-01", "1989-01-01",
        ],
        "dia1 monització  fecha": ["2026-01-01"] * 10,
        "ts_created": ["2025-12-28"] * 10,
        "Sexo": ["hombre", "mujer"] * 5,
        "clinical_source_sheet": ["EM", "Controles"] * 5,
    })

    X_train, X_test, *_ = processor.prepare_splits(df, "sdmt", test_size=0.2)

    assert "Día 3 M-FIS score fis" not in X_train.columns
    assert "DIA 1 TMT papel (tiempo) A" not in X_train.columns
    assert "digital_speed" in X_train.columns
    assert "digital_errors" in X_train.columns
    assert "age_at_test" in X_train.columns
    assert "sex_binary" in X_train.columns
