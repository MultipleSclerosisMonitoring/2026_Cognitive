# Clinical Calibration System for Mobile Cognitive Tests

This software architecture is designed for the validation and calibration of mobile-based digital cognitive assessments, specifically targeting the Symbol Digit Modalities Test (SDMT) and the Trail Making Test (TMT). Its primary objective is to map spatiotemporal interaction features captured from digital screens onto traditional paper-based scores (the clinical gold standard) using machine learning.

## Architectural Design & Scientific Justification

The codebase adheres strictly to object-oriented programming principles and clean architecture standards, decoupling core concerns such as data ingestion, model orchestration, and performance reporting. This design decision guarantees that clinical researchers can expand the platform's algorithmic capabilities without destabilizing verified data processing logic.

### Data Isolation and Leakage Prevention
The data loading framework manages connections to secure PostgreSQL databases and merges digital assessment telemetry with tabular clinical sheets. A foundational feature within this component is the implementation of a Grouped Cross-Validation strategy partitioned by patient identifiers. In clinical modeling, a system must not evaluate its performance based on data splits that share samples from the same individual across training and validation folds. If multiple test sessions from a single patient appear on both sides of a split, the model will inadvertently memorize individual-specific traits rather than isolating generalizable patterns associated with neurological trajectories. Grouping by patient ID ensures a strict blind evaluation against completely unseen cohorts.

### Clinical Safety Boundaries and Extrapolation Control
The algorithm layer implements an abstract class contract enforcing structural parity across diverse analytical models. Every estimator wrapper monitors and logs the absolute boundaries (minimum and maximum limits) of all independent variables during the training phase. If the model encounters feature values that fall outside these historical parameters during inference, it issues formal system warnings. This architectural safeguard is crucial for preventing silent extrapolation, a common failure mode where machine learning models produce structurally valid outputs for patient distributions that deviate dramatically from the initial validation population.

### Dynamic Reporting and Medical Interpretability
The reporting framework acts as the operational endpoint, transforming multi-dimensional arrays into clinically actionable assets. It automatically logs statistical metrics into cumulative spreadsheets and constructs standalone interactive visualizations. These visualizations include ideal reference lines that allow clinical investigators to rapidly detect statistical bias, assess calibration quality, and audit systemic error distributions across different performance spectrums.

## Directory Structure

A complete textual schema that clearly represents the project's directory tree, visualizing the location of data components, models, utilities, and internationalization files.

```text
.
├── .env.example
├── pyproject.toml
├── poetry.lock
├── README.md
├── calibration/
│   ├── config.yaml              # SDMT
│   ├── config_tmt.yaml          # TMT
│   ├── group_comparison.py      # EM vs Controles
│   ├── main.py
│   ├── data/
│   │   ├── datos_papel.xlsx
│   │   └── loader.py
│   ├── models/
│   │   ├── base.py
│   │   └── algoritmos.py
│   ├── utils/
│   │   ├── clinical_discussion.py
│   │   └── reporting.py
│   └── locales/
│       ├── es/LC_MESSAGES/
│       ├── en/LC_MESSAGES/
│       └── fr/LC_MESSAGES/
├── library/
└── tests/
```

## Core Functionalities: 

A detailed explanation of the system's current capabilities, covering multi-source data consolidation, pipeline automation using the Factory design pattern, hyperparameter optimization methods, and localization engine integration for multiple languages.


## Configuration Schema
The pipeline is driven by YAML configuration files under `calibration/` and environment variables loaded from the repository root `.env` file.

- `calibration/config.yaml`: SDMT experiment.
- `calibration/config_tmt.yaml`: TMT experiment.
- `.env`: database credentials, language, and Excel path. Database user, password, and name are required for the calibration pipeline.
- `.env.example`: safe template for creating `.env`.

Example `.env`:

```bash
DB_HOST=your-postgres-host
DB_PORT=5432
DB_USER=your_db_user
DB_PASSWORD=your_db_password
DB_NAME=your_db_name
APP_LANG=en
EXCEL_DATA_PATH=./calibration/data/datos_papel.xlsx
```

Example configuration excerpt:

```yaml
data:
  excel_path: "./data/datos_papel.xlsx"
  clinical_skiprows: 2
  test_type: "sdmt"
models:
  - type: "linear"
    target_mode: "single_output"
    cv_folds: 3
  - type: "rf"
    target_mode: "single_output"
    cv_folds: 3
output:
  excel_report: "resultados.xlsx"
  html_plot: "graficos.html"
uncertainty:
  enabled: true
  conformal_alpha: 0.10
  calibration_size: 0.20
  bootstrap_iterations: 500
```

## Execution Protocol

### 1. Prepare the environment

Create the runtime environment file from the template and fill in the PostgreSQL credentials:

```bash
cp .env.example .env
```

Install dependencies if needed:

```bash
poetry install
```

### 2. Run SDMT model creation

This command trains and evaluates the SDMT calibration models against the paper gold standard:

```bash
python -m calibration.main --config calibration/config.yaml --verbose 3
```

Outputs generated by default:

- `resultados.xlsx`
- `calibration_*sdmt*.html`
- `residuals_*sdmt*.html`
- `bland_altman_*sdmt*.html`

### 3. Run TMT model creation

This command trains and evaluates the TMT calibration models against the paper gold standard:

```bash
python -m calibration.main --config calibration/config_tmt.yaml --verbose 3
```

Outputs generated by default:

- `resultados_tmt.xlsx`
- `calibration_*tmt*.html`
- `residuals_*tmt*.html`
- `bland_altman_*tmt*.html`

### 4. Notes

- The pipeline loads `.env` automatically from the repository root and also accepts `calibration/.env` as a legacy fallback.
- SDMT and TMT should be run with separate configuration files so their reports are not mixed.
- The current evaluation includes grouped patient splits, grouped bootstrap confidence intervals, and conformal prediction intervals.

### 5. Compare affected participants with controls

The group-comparison analysis evaluates the digital and paper versions of SDMT and TMT between the `EM` and `Controles` sheets. It reports Welch and Mann-Whitney tests, plus an OLS model with HC3 errors adjusted for age, sex, and education. The `FDR_q` column applies Benjamini-Hochberg correction across adjusted comparisons.

```bash
python -m calibration.group_comparison \
  --input calibration/data/datos_papel.xlsx \
  --output group_comparison.xlsx
```

The output workbook contains `GroupComparison`, `GroupCounts`, `Covariates`, `ParticipantSummary`, and `StratifiedSummary` sheets. Small control-group counts and missing paper/digital measurements should be considered when interpreting p-values.

The calibration pipeline also writes `comparativa_clinica_final.xlsx` when both SDMT and TMT result workbooks are available. This file summarizes the model metrics and subgroup calibration results; it is separate from `group_comparison.xlsx`, which compares the observed digital and paper outcomes between clinical groups.

## Quick Result Interpretation

After each run, the main summary tables are written to:

- `resultados.xlsx` for SDMT.
- `resultados_tmt.xlsx` for TMT.

The most useful columns to review first are:

- `RMSE`: average prediction error in the original paper-test scale. Lower is better.
- `MAE`: median-like average absolute error, often easier to interpret clinically than RMSE. Lower is better.
- `R2_Score`: proportion of variance explained. Higher is better, and negative values indicate poor calibration.
- `CCC_Lin`: Lin's concordance coefficient between digital prediction and paper result. Higher is better.
- `ICC_A1`: absolute-agreement intraclass correlation. Higher is better.
- `PI_Coverage`: empirical coverage of the conformal prediction interval. Values near `1 - alpha` are desirable.
- `PI_Mean_Width`: average width of the prediction interval. Narrower is better if coverage remains acceptable.

A practical reading order is:

1. Check `RMSE` and `MAE` to understand raw prediction error.
2. Check `CCC_Lin` and `ICC_A1` to assess agreement with the paper gold standard.
3. Check `PI_Coverage` and `PI_Mean_Width` together to see whether uncertainty is both reliable and informative.
4. Review the Bland-Altman HTML plots to detect systematic bias or heteroscedasticity.

As a rule of thumb:

- A model with lower error but extremely wide intervals may be statistically safe but clinically less useful.
- A model with good `R2` but low `CCC` or `ICC` may rank patients reasonably well while still failing to reproduce paper-equivalent scores.
- Negative `R2` usually means the model is not yet suitable for clinical translation.

## Future Engineering Extensions 

Una hoja de ruta técnica que detalla las extensiones arquitectónicas naturales que el sistema admite gracias a su alta modularidad, tales como el soporte para Deep Learning (redes neuronales recurrentes o Transformers), motores automáticos de ingeniería de características, interfaces de programación de aplicaciones (APIs) en tiempo real para aplicaciones periféricas e integración multinivel de biomarcadores complementarios.

