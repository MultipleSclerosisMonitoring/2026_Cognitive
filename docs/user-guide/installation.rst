Installation
============

Requirements
------------

* Python 3.12 or newer.
* PostgreSQL access for the calibration pipeline, or a local ``sdmt.csv`` /
  ``tmt.csv`` fallback in the working directory.
* The clinical workbook ``calibration/data/datos_papel.xlsx``.

Using the project virtual environment
-------------------------------------

The repository can be run with the shared virtual environment::

   /home/jordieres/soft/sclerosis/vpy-ms/bin/python -m pip install -e .

For a portable setup, install the project in any Python 3.12 environment::

   python -m pip install -e .

Build this documentation locally
---------------------------------

Install the documentation dependencies and build with warnings treated as
errors::

   python -m pip install -r docs/requirements.txt
   sphinx-build -b html docs docs/_build/html -W --keep-going

Open ``docs/_build/html/index.html`` after a successful build.

Environment
-----------

Copy ``.env.example`` to ``.env`` and set ``DB_HOST``, ``DB_PORT``, ``DB_USER``,
``DB_PASSWORD``, ``DB_NAME``, and optionally ``EXCEL_DATA_PATH`` and ``APP_LANG``.
The application requires the three database credential variables even when
host and port are supplied by YAML.
