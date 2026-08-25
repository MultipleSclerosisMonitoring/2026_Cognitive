Data contract
=============

Clinical workbook
-----------------

The clinical workbook contains at least the ``EM`` and ``Controles`` sheets.
Each sheet should have a row containing the ``Código`` column, followed by
participant records. Important columns include ``Sexo``, ``Nivel
escolarización``, ``Edad Inclusión estudio``, SDMT digital/paper values, and TMT
digital/paper times and errors.

The loader preserves ``clinical_source_sheet`` as the clinical group and maps
``EM`` to affected participants and ``Controles`` to healthy controls in the
clinical comparison report. Empty rows are removed before this marker is added.

Digital source
--------------

The database table is named after the test type. The digital data must contain
``codeid`` so it can be standardized to ``patient_id``. When a table cannot be
read, ``sdmt.csv`` or ``tmt.csv`` is sought in the current working directory.

Leakage prevention
------------------

Patient identifiers are used only for grouping and are excluded from model
features. Hold-out and cross-validation splits keep all sessions for a patient
in the same group. Paper targets and post-target clinical fields are excluded
from the digital feature matrix according to the configuration and loader
filters.

Derived variables
-----------------

The loader can derive ``age_at_test``, ``sex_binary``, ``education_band``,
``clinical_group``, ``delta_dias_digital_papel``, and disease/impact bands when
source columns are available. Missingness is retained in metadata and handled
by the train-fitted feature preparation logic.
