Reports and artifacts
=====================

Calibration output
------------------

The configured Excel report contains a ``Metrics`` sheet with RMSE, MAE, R2,
Lin's CCC, absolute-agreement ICC, Bland-Altman statistics, and optional
prediction-interval and bootstrap columns. Additional sheets can include
``FeatureAudit`` and ``StratifiedMetrics``. HTML artifacts include scatter,
residual, calibration, and Bland-Altman plots named from the model and target.

Clinical comparison output
--------------------------

Run::

   python -m calibration.group_comparison \
      --input calibration/data/datos_papel.xlsx \
      --output group_comparison.xlsx

The workbook contains:

* ``GroupComparison``: Welch, Mann-Whitney, HC3-adjusted group effects, and
  Benjamini-Hochberg ``FDR_q`` values.
* ``GroupCounts`` and ``Covariates``: sample and variable coverage.
* ``ParticipantSummary``: one row per loaded participant and observed outcome.
* ``StratifiedSummary``: means, medians, standard deviations, and effective N
  by implementation, group, sex, education, age band, and crossed strata.

The global ``comparativa_clinica_final.xlsx`` is assembled from the SDMT and
TMT calibration workbooks when both are available.
