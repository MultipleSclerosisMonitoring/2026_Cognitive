Configuration
=============

YAML files under ``calibration/`` define data sources, target mappings, model
queues, uncertainty settings, and output names. The default files are:

* ``config.yaml``: SDMT.
* ``config_tmt.yaml``: TMT.
* ``config_sdmt_*.yaml``: SDMT comparison scenarios.
* ``config_tmt_stratified_focus.yaml``: TMT education-stratified scenario.

A minimal data section is::

   data:
     excel_path: "./data/datos_papel.xlsx"
     clinical_skiprows: 2
     test_type: "sdmt"

The ``clinical_skiprows`` value is now a fallback search position: the loader
also detects the row containing ``Código`` separately in each worksheet.

Model configuration
-------------------

Each model entry supports ``type``, ``target_mode``, ``cv_folds``,
``param_grid``, ``search_strategy``, and ``n_iter``. TMT configurations can
also use ``stratify_by: education_band`` and grouped multi-output targets.
The available queue is configuration-driven; do not infer that every model
listed in the technical protocol is enabled in every run.

Environment precedence
----------------------

``DB_*`` and ``EXCEL_DATA_PATH`` environment variables take precedence over
YAML values. ``--lang`` takes precedence over ``APP_LANG``. Database user,
password, and name are required by :mod:`calibration.main`.
