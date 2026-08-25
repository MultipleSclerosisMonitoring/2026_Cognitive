Pipeline
========

The calibration command is the main operational entry point::

   python -m calibration.main --config calibration/config.yaml --verbose 3
   python -m calibration.main --config calibration/config_tmt.yaml --verbose 3

The first command runs the SDMT configuration and the second runs TMT. The
configuration path is resolved from the repository, while relative input and
output paths are resolved against the configuration directory or repository
root as implemented by :func:`calibration.main._resolve_existing_path` and
:func:`calibration.main._resolve_output_path`.

Execution stages
----------------

#. Load digital data from PostgreSQL table ``sdmt`` or ``tmt``. If the database
   cannot be read, use the matching local CSV fallback.
#. Read every clinical workbook sheet and detect its header row independently.
   This supports the different header positions in ``EM`` and ``Controles``.
#. Standardize patient identifiers and merge digital and clinical records.
#. Resolve paper targets, derive clinical covariates, filter features, and
   split patients with :class:`sklearn.model_selection.GroupShuffleSplit`.
#. Train each configured model using grouped cross-validation.
#. Optionally estimate split-conformal intervals and grouped bootstrap metrics.
#. Write Excel metrics and HTML visualizations, then update the clinical
   discussion workbook when both SDMT and TMT reports exist.

.. mermaid::

   flowchart LR
      A[PostgreSQL or CSV] --> B[DataProcessor]
      C[Clinical Excel] --> B
      B --> D[Patient grouped splits]
      D --> E[Calibrator]
      E --> F[ReportGenerator]
      F --> G[Excel metrics]
      F --> H[HTML plots]
      G --> I[Clinical discussion workbook]
