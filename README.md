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
calibration/
├── pyproject.toml
├── poetry.lock
├── config.yaml
├── main.py
├── data/
│   └── loader.py
├── models/
│   ├── base.py
│   └── algoritmos.py
├── utils/
│   └── reporting.py
└── locales/
    ├── es/LC_MESSAGES/
    ├── en/LC_MESSAGES/
    └── fr/LC_MESSAGES/
```

## Core Functionalities: 

A detailed explanation of the system's current capabilities, covering multi-source data consolidation, pipeline automation using the Factory design pattern, hyperparameter optimization methods, and localization engine integration for multiple languages.


## Configuration Schema
The behavior of the analytics pipeline is governed by a single structured YAML configuration file. This document manages secure database connection endpoints, gold standard spreadsheet paths, evaluation targets, target models, and optimization spaces.

```yaml
data:
  db_uri: "postgresql://clinical_user:secure_password@localhost:5432/cognitive_db"
  excel_path: "clinical_gold_standard.xlsx"
  test_type: "sdmt"
models:
  - type: "xgboost"
    cv_folds: 5
    search_strategy: "random"
    n_iter: 20
    param_grid:
      learning_rate: [0.01, 0.05, 0.1]
      max_depth: [3, 5, 7]
      n_estimators: [100, 200]
output:
  excel_report: "calibration_performance.xlsx"
  html_plot: "digital_vs_paper_dispersion.html"
language: "en"

```

## Execution Protocol

Once the validation profile is defined, launch the main experiment pipeline through the command line interface, providing the configuration path and the desired application log verbosity level:

```bash
poetry run python main.py --config config.yaml --verbose 3
```

## Future Engineering Extensions 

Una hoja de ruta técnica que detalla las extensiones arquitectónicas naturales que el sistema admite gracias a su alta modularidad, tales como el soporte para Deep Learning (redes neuronales recurrentes o Transformers), motores automáticos de ingeniería de características, interfaces de programación de aplicaciones (APIs) en tiempo real para aplicaciones periféricas e integración multinivel de biomarcadores complementarios.


