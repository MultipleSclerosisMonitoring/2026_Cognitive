import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration.utils.clinical_discussion import (
    build_clinical_discussion_workbook,
    save_feature_audit,
    save_stratified_metrics,
)
from calibration.utils.reporting import ReportGenerator


def test_reporting_includes_concordance_metrics(tmp_path):
    reporter = ReportGenerator(
        output_excel=str(tmp_path / "metrics.xlsx"),
        output_html=str(tmp_path / "plots" / "index.html"),
    )

    y_true = pd.Series([10.0, 12.0, 14.0, 16.0, 18.0])
    y_pred = pd.Series([10.5, 11.5, 13.5, 16.5, 18.5])

    metrics = reporter.evaluate_and_save(y_true=y_true, y_pred=y_pred, model_name="demo")

    assert "CCC_Lin" in metrics
    assert "ICC_A1" in metrics
    assert "BlandAltman_Bias" in metrics
    assert "BlandAltman_LoA_Low" in metrics
    assert "BlandAltman_LoA_High" in metrics
    assert "Calibration_Slope" in metrics
    assert Path(tmp_path / "metrics.xlsx").exists()


def test_reporting_generates_bland_altman_plot(tmp_path):
    reporter = ReportGenerator(
        output_excel=str(tmp_path / "metrics.xlsx"),
        output_html=str(tmp_path / "plots" / "index.html"),
    )

    y_true = pd.Series([20.0, 22.0, 24.0, 26.0])
    y_pred = pd.Series([19.5, 21.0, 24.5, 27.0])

    reporter.generate_bland_altman_plot(y_true=y_true, y_pred=y_pred, model_name="demo")

    assert Path(tmp_path / "plots" / "bland_altman_demo.html").exists()


def test_reporting_includes_grouped_bootstrap_and_interval_metrics(tmp_path):
    reporter = ReportGenerator(
        output_excel=str(tmp_path / "metrics.xlsx"),
        output_html=str(tmp_path / "plots" / "index.html"),
    )

    y_true = pd.Series([10.0, 11.0, 14.0, 15.0, 18.0, 19.0])
    y_pred = pd.Series([10.4, 10.7, 13.5, 15.2, 17.8, 18.9])
    groups = pd.Series(["P1", "P1", "P2", "P2", "P3", "P3"])
    lower = pd.Series([9.0, 9.5, 12.5, 14.0, 16.5, 17.5])
    upper = pd.Series([11.5, 12.0, 14.5, 16.5, 19.5, 20.5])

    metrics = reporter.evaluate_and_save(
        y_true=y_true,
        y_pred=y_pred,
        model_name="demo_uncertainty",
        groups=groups,
        prediction_interval_low=lower,
        prediction_interval_high=upper,
        interval_alpha=0.10,
        bootstrap_iterations=30,
        bootstrap_confidence=0.90,
        bootstrap_random_state=7,
    )

    assert metrics["PI_Coverage"] == 1.0
    assert metrics["PI_Mean_Width"] > 0.0
    assert metrics["Bootstrap_Iterations_Requested"] == 30
    assert metrics["Bootstrap_Iterations_Valid"] > 0
    assert "RMSE_CI_Low" in metrics
    assert "RMSE_CI_High" in metrics
    assert "CCC_Lin_CI_Low" in metrics
    assert "PI_Coverage_CI_High" in metrics


def test_clinical_discussion_outputs_stratified_and_summary_files(tmp_path):
    output_excel = tmp_path / "metrics.xlsx"
    reporter = ReportGenerator(
        output_excel=str(output_excel),
        output_html=str(tmp_path / "plots" / "index.html"),
    )
    metrics = reporter.evaluate_and_save(
        y_true=pd.Series([10.0, 12.0, 14.0, 16.0, 18.0, 20.0]),
        y_pred=pd.Series([10.1, 11.8, 13.9, 16.4, 18.2, 19.7]),
        model_name="rf_sdmt_paper_score",
    )
    metadata = pd.DataFrame({
        "clinical_group": ["EM", "EM", "Controles", "Controles", "EM", "Controles"],
        "age_band": ["<40", "40-54", "40-54", "55+", "55+", "<40"],
        "sex_binary": [0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
    })
    save_stratified_metrics(
        output_excel=str(output_excel),
        model_name="rf_sdmt_paper_score",
        y_true=pd.Series([10.0, 12.0, 14.0, 16.0, 18.0, 20.0]),
        y_pred=pd.Series([10.1, 11.8, 13.9, 16.4, 18.2, 19.7]),
        metadata=metadata,
    )
    save_feature_audit(
        output_excel=str(output_excel),
        test_type="sdmt",
        feature_audit={
            "removed_by_reason": {"non_digital_or_non_stable_clinical": ["foo"]},
            "selected_features_final": ["bar"],
        },
    )
    comparison_path = tmp_path / "comparison.xlsx"
    build_clinical_discussion_workbook(str(comparison_path), [str(output_excel)])

    assert metrics["RMSE"] > 0.0
    stratified = pd.read_excel(output_excel, sheet_name="StratifiedMetrics")
    assert not stratified.empty
    comparison = pd.read_excel(comparison_path, sheet_name="BestByTarget")
    assert "Clinical_Status" in comparison.columns
