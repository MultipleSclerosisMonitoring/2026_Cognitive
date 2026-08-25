Limitations and research status
===============================

The implemented pipeline is a calibration and reporting system, not a clinical
diagnostic device. Current limitations include:

* one grouped hold-out split per run rather than the repeated nested CV proposed
  in the research protocol;
* a small healthy-control sample in the current workbook;
* missing and unevenly distributed digital, paper, sex, education, and age data;
* model selection and target availability depend on the selected YAML;
* the clinical comparison uses the source sheet as the group label;
* model persistence and dataset hashing are available as model-level concerns
  but are not an end-to-end immutable experiment registry.

Treat p-values, subgroup means, and model metrics as exploratory until the
sample size, missing-data mechanism, and analysis plan are prespecified.
