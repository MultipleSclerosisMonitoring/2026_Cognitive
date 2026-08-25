Architecture overview
=====================

The package separates data preparation, model contracts, algorithm adapters,
reporting, clinical comparison, and localization.

.. mermaid::

   classDiagram
      class DataProcessor {
         +load_and_merge(test_type) DataFrame
         +prepare_splits(df, test_type) tuple
         +get_cv_folds(groups_train, n_splits) Any
      }
      class BaseModelCalibrator {
         <<abstract>>
         +train(X, y, **kwargs) None
         +predict(X) ndarray
         +check_boundaries(X) Dict
         +save(filepath) None
         +load(filepath) BaseModelCalibrator
      }
      class StandardCalibrator
      class CountErrorClassifierCalibrator
      class StackingEnsembleCalibrator
      class SplineGAMCalibrator
      class ReportGenerator {
         +evaluate_and_save(y_true, y_pred, model_name) Dict
         +generate_scatter_plot(y_true, y_pred, model_name) None
         +generate_residuals_plot(y_true, y_pred, model_name) None
         +generate_bland_altman_plot(y_true, y_pred, model_name) None
      }
      DataProcessor --> BaseModelCalibrator : supplies prepared data
      BaseModelCalibrator <|-- StandardCalibrator
      BaseModelCalibrator <|-- CountErrorClassifierCalibrator
      BaseModelCalibrator <|-- StackingEnsembleCalibrator
      BaseModelCalibrator <|-- SplineGAMCalibrator
      ReportGenerator --> BaseModelCalibrator : evaluates predictions
