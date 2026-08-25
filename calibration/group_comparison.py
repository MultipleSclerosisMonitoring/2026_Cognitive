"""Compare cognitive-test results between EM and control participants."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import mannwhitneyu, ttest_ind
from statsmodels.stats.multitest import multipletests

from calibration.data.loader import DataProcessor


TEST_COLUMNS = {
    "SDMT": {
        "digital_score": "Día 1 SDMT Digital score",
        "paper_score": "Día 1 SDMT Papel score",
    },
    "TMT": {
        "digital_time_a": "DIA 1 TMT Digital (tiempo) A",
        "digital_time_b": "DIA 1 TMT Digital (tiempo) B",
        "digital_errors_a": "DIA 1 TMT Digital (errores) A",
        "digital_errors_b": "DIA 1 TMT Digital (errores) B",
        "paper_time_a": "DIA 1 TMT papel (tiempo) A",
        "paper_time_b": "DIA 1 TMT papel (tiempo) B",
        "paper_errors_a": "DIA 1 TMT papel (errores) A",
        "paper_errors_b": "DIA 1 TMT papel (errores) B",
    },
}


def _find_column(frame: pd.DataFrame, expected: str) -> str | None:
    normalized = DataProcessor._normalize_lookup_text(expected)
    for column in frame.columns:
        if DataProcessor._normalize_lookup_text(column) == normalized:
            return str(column)
    return None


def _first_column(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    return next((found for candidate in candidates if (found := _find_column(frame, candidate))), None)


def _prepare(
    frame: pd.DataFrame,
    outcome: str,
    group: str,
    age: str | None,
    sex: str | None,
    education: str | None,
) -> tuple[pd.DataFrame, list[str]]:
    covariates = [column for column in (age, sex, education) if column]
    columns = [group, outcome, *covariates]
    data = frame.loc[:, [column for column in columns if column in frame]].copy()
    data = data.rename(columns={group: "group", outcome: "outcome"})
    rename_map = {}
    if age:
        rename_map[age] = "age"
    if sex:
        rename_map[sex] = "sex"
    if education:
        rename_map[education] = "education"
    data = data.rename(columns=rename_map)
    data["group"] = data["group"].map({"EM": 1, "Controles": 0})
    data["outcome"] = pd.to_numeric(data["outcome"], errors="coerce")
    model_covariates = [name for name in ("age", "sex", "education") if name in data]
    return data.dropna(subset=["group", "outcome"]), model_covariates


def compare_outcome(
    frame: pd.DataFrame,
    outcome: str,
    group: str,
    age: str | None,
    sex: str | None,
    education: str | None,
) -> dict:
    data, model_covariates = _prepare(frame, outcome, group, age, sex, education)
    controls = data.loc[data["group"] == 0, "outcome"]
    affected = data.loc[data["group"] == 1, "outcome"]
    result = {
        "Outcome": outcome,
        "N_EM": int(len(affected)),
        "N_Controles": int(len(controls)),
        "Mean_EM": affected.mean(),
        "Mean_Controles": controls.mean(),
        "Median_EM": affected.median(),
        "Median_Controles": controls.median(),
        "Welch_t_p": float("nan"),
        "Mann_Whitney_p": float("nan"),
        "Adjusted_group_effect": float("nan"),
        "Adjusted_p": float("nan"),
        "Adjusted_N": int(len(data)),
        "Adjustment": "age + sex + education",
    }
    if len(affected) > 1 and len(controls) > 1:
        result["Welch_t_p"] = float(ttest_ind(affected, controls, equal_var=False).pvalue)
    if len(affected) and len(controls):
        result["Mann_Whitney_p"] = float(mannwhitneyu(affected, controls, alternative="two-sided").pvalue)

    if len(data) >= 10 and data["group"].nunique() == 2 and model_covariates:
        formula = "outcome ~ group + " + " + ".join(
            "C(sex)" if covariate == "sex" else "C(education)" if covariate == "education" else covariate
            for covariate in model_covariates
        )
        model_data = data.dropna(subset=model_covariates)
        try:
            fitted = smf.ols(formula, data=model_data).fit(cov_type="HC3")
            result["Adjusted_group_effect"] = float(fitted.params["group"])
            result["Adjusted_p"] = float(fitted.pvalues["group"])
            result["Adjusted_N"] = int(fitted.nobs)
        except (KeyError, ValueError, TypeError):
            pass
    return result


def _build_summary_frames(
    frame: pd.DataFrame,
    group: str,
    age: str | None,
    sex: str | None,
    education: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    participant = pd.DataFrame(index=frame.index)
    participant["Group"] = frame[group].map({"EM": "MS", "Controles": "Healthy"})
    code = _find_column(frame, "Código")
    participant["Participant"] = frame[code] if code else pd.NA
    participant["Sex"] = frame[sex].astype("string").str.strip().str.lower() if sex else pd.NA
    participant["Education"] = frame[education].astype("string").str.strip().str.lower() if education else pd.NA
    participant["Age"] = pd.to_numeric(frame[age], errors="coerce") if age else pd.NA
    participant.loc[participant["Age"] <= 0, "Age"] = pd.NA
    participant["Age_band"] = pd.cut(
        participant["Age"], bins=[0, 40, 55, float("inf")], labels=["<40", "40-54", "55+"], right=False
    )

    measure_rows = []
    for test_type, outcomes in TEST_COLUMNS.items():
        for measure, expected in outcomes.items():
            column = _find_column(frame, expected)
            if not column:
                continue
            implementation = "digital" if measure.startswith("digital_") else "paper"
            values = pd.to_numeric(frame[column], errors="coerce")
            value_frame = participant[["Group", "Sex", "Education", "Age_band"]].copy()
            value_frame["Value"] = values.to_numpy()
            value_frame = value_frame.dropna(subset=["Group", "Value"])
            grouping_specs = [
                ("Group", ["Group"]),
                ("Sex", ["Sex"]),
                ("Education", ["Education"]),
                ("Age_band", ["Age_band"]),
                ("Group x Sex", ["Group", "Sex"]),
                ("Group x Education", ["Group", "Education"]),
                ("Group x Age_band", ["Group", "Age_band"]),
            ]
            for stratifier, grouping_columns in grouping_specs:
                grouped = value_frame.dropna(subset=grouping_columns).groupby(grouping_columns, observed=True)["Value"]
                for stratum, values_by_stratum in grouped:
                    if not isinstance(stratum, tuple):
                        stratum = (stratum,)
                    measure_rows.append({
                        "Test": test_type,
                        "Measure": measure,
                        "Implementation": implementation,
                        "Stratifier": stratifier,
                        "Stratum": " | ".join(str(value) for value in stratum),
                        "N": int(values_by_stratum.size),
                        "Mean": float(values_by_stratum.mean()),
                        "Median": float(values_by_stratum.median()),
                        "SD": float(values_by_stratum.std(ddof=1)) if values_by_stratum.size > 1 else float("nan"),
                    })
            participant[f"{test_type}_{measure}"] = values.to_numpy()
    participant = participant.reset_index(drop=True)
    return participant, pd.DataFrame(measure_rows)


def run_analysis(input_path: str, output_path: str) -> pd.DataFrame:
    processor = DataProcessor("sqlite://", input_path)
    frame = processor._load_clinical_data()
    group = "clinical_source_sheet"
    age = _first_column(frame, ("Edad Inclusión estudio", "Edad inclusión estudio", "Edad"))
    sex = _find_column(frame, "Sexo")
    education = _find_column(frame, "Nivel escolarización")
    rows = []
    for test_type, outcomes in TEST_COLUMNS.items():
        for measure, expected in outcomes.items():
            column = _find_column(frame, expected)
            if column is None:
                rows.append({"Test": test_type, "Measure": measure, "Outcome": expected, "Status": "column not found"})
                continue
            row = compare_outcome(frame, column, group, age, sex, education)
            row.update({"Test": test_type, "Measure": measure, "Status": "ok"})
            rows.append(row)

    results = pd.DataFrame(rows)
    valid = results["Adjusted_p"].notna()
    results["FDR_q"] = float("nan")
    if valid.any():
        results.loc[valid, "FDR_q"] = multipletests(results.loc[valid, "Adjusted_p"], method="fdr_bh")[1]
    counts = frame.groupby(group).size().rename("N").reset_index()
    covariate_summary = pd.DataFrame({"Variable": ["age", "sex", "education"], "Column": [age, sex, education]})
    participant_summary, stratified_summary = _build_summary_frames(frame, group, age, sex, education)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        results.to_excel(writer, sheet_name="GroupComparison", index=False)
        counts.to_excel(writer, sheet_name="GroupCounts", index=False)
        covariate_summary.to_excel(writer, sheet_name="Covariates", index=False)
        participant_summary.to_excel(writer, sheet_name="ParticipantSummary", index=False)
        stratified_summary.to_excel(writer, sheet_name="StratifiedSummary", index=False)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare EM and control cognitive-test results.")
    parser.add_argument("--input", default="calibration/data/datos_papel.xlsx")
    parser.add_argument("--output", default="group_comparison.xlsx")
    args = parser.parse_args()
    run_analysis(args.input, args.output)
    print(f"Analysis written to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
