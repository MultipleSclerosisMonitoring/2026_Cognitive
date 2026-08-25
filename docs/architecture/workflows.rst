Workflows
=========

Calibration sequence
--------------------

.. mermaid::

   sequenceDiagram
      participant CLI as calibration.main
      participant DP as DataProcessor
      participant M as Calibrator
      participant R as ReportGenerator
      CLI->>DP: load_and_merge(test_type)
      DP-->>CLI: merged clinical/digital frame
      CLI->>DP: prepare_splits(frame, test_type)
      DP-->>CLI: grouped train/test data and metadata
      CLI->>M: train(X_train, y_train, grouped CV)
      M-->>CLI: fitted model and intervals
      CLI->>M: predict(X_test)
      M-->>CLI: predictions
      CLI->>R: evaluate_and_save(...)
      R-->>CLI: Excel metrics and plots

Clinical comparison state
-------------------------

.. mermaid::

   stateDiagram-v2
      [*] --> LoadWorkbook
      LoadWorkbook --> DetectHeaders
      DetectHeaders --> NormalizeRows
      NormalizeRows --> BuildStrata
      BuildStrata --> CompareOutcomes
      CompareOutcomes --> AdjustCovariates
      AdjustCovariates --> CorrectFDR
      CorrectFDR --> WriteWorkbook
      WriteWorkbook --> [*]

GitHub Pages deployment
-----------------------

The workflow in ``.github/workflows/docs.yml`` builds this tree with
``sphinx-build -W --keep-going`` and publishes ``docs/_build/html`` using the
official Pages artifact and deployment actions. Sphinx already produces a
complete static HTML site, so a Jekyll theme is not required. The workflow adds
``.nojekyll`` to the artifact to prevent branch-style Pages processing from
rewriting Sphinx paths or ignoring underscore-prefixed assets.

In repository settings, choose **Settings > Pages > Source > GitHub Actions**.
Do not select **Deploy from a branch**, because that mode is the one that
expects a Jekyll-oriented branch layout.
