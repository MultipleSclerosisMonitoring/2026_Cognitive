Command-line entry points
=========================

Calibration pipeline
--------------------

.. automodule:: calibration.main
   :members: load_environment, setup_logging

The executable command is::

   python -m calibration.main --config CONFIG.yml [--verbose 0..4] [--lang es|en|fr]

Clinical comparison
-------------------

The executable command is::

   python -m calibration.group_comparison --input WORKBOOK.xlsx --output REPORT.xlsx
