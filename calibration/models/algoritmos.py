import logging
from typing import Any, Dict, Optional, List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.dummy import DummyClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import xgboost as xgb
from .base import BaseModelCalibrator, PipelineNotTrainedError
from datetime import datetime
from calibration.i18n import get_translator

logger = logging.getLogger(__name__)
_ = get_translator()

class StandardCalibrator(BaseModelCalibrator):
    """Standardized wrapper for managing the lifecycle of predictive estimators.
    
    This class acts as a decorator around scikit-learn and XGBoost models. 
    Its primary responsibility is to ensure that clinical data is always 
    pre-processed in the exact same manner (scaling) and to guarantee safety 
    during inference time by verifying that new incoming inputs do not exceed 
    the ranges observed during the training phase.
    
    Attributes:
        base_estimator (Any): The underlying machine learning algorithm to be trained.
        model (Any): The final fitted model. If hyperparameter optimization is used, 
            this will store the `best_estimator_` obtained from the search process.
        scaler (StandardScaler): Instance responsible for standardizing features 
            by removing the mean and scaling to unit variance.
        feature_ranges (Dict[str, Tuple[float, float]]): Dictionary storing the 
            minimum and maximum values for each feature encountered during training.
        best_params (Dict): Best hyperparameters found during search
        cv_results (Dict): Cross-validation results from search
    """
    
    def __init__(self, estimator: Any):
        """Initializes the calibration context for a given estimator.
        
        Upon instantiation, the final model attribute is initially set to `None` 
        because the true fitted model will only be resolved after the training 
        and potential hyperparameter search processes are completed.
        
        Args:
            estimator (Any): An instance of a regressor or classifier compatible 
                with the scikit-learn API (e.g., it must implement `fit` and `predict`).
        """
        super().__init__()
        self.base_estimator = estimator
        self.model = None
        self.scaler = StandardScaler()
        self.feature_ranges: Dict[str, Tuple[float, float]] = {}
        self.best_params: Dict = {}
        self.cv_results: Dict = {}
        self._estimator_name = type(estimator).__name__
        
    def train(
        self, 
        X: pd.DataFrame, 
        y: pd.Series | pd.DataFrame, 
        param_grid: Optional[Dict[str, list]] = None, 
        cv_folds: Any = 3, 
        search_strategy: str = "grid", 
        n_iter: int = 10
    ) -> None:
        """Fits the scaler and the base estimator integrating cross-validation and optimization.
        
        The logical process of this function follows these steps:
        Firstly, it validates the input data and stores baseline training statistics.
        Secondly, it explores the training dataset to record safety boundaries for all features.
        Thirdly, it fits and applies standard scaling exclusively on the training data.
        Fourthly, it evaluates whether hyperparameter tuning is required based on the presence 
        of a parameter grid.
        Fifthly, if required, it branches between a randomized search for computational 
        efficiency or an exhaustive grid search for maximum precision.
        Finally, it internally stores the winning estimator for future inference.
        
        :param X: Feature matrix extracted from mobile-device interactions.
        :param y: Paper-based target score vector.
        :param param_grid: Optional hyperparameter search space.
        :param cv_folds: Fold count or a pre-configured grouped splitter.
        :param search_strategy: Either "grid" or "random".
        :param n_iter: Maximum sampled settings for random search.
        :raises ValueError: If the feature matrix or target vector has null data or incompatible shapes.
        """
        # Validate inputs first (will call parent's validation)
        super().train(X, y)
        
        logger.info(_("▶️ Iniciando fase de entrenamiento para modelo: %s"), self._estimator_name)
        
        # Store feature ranges for boundary checking during inference
        self.feature_ranges = {
            col: (float(X[col].min()), float(X[col].max())) for col in X.columns
        }
        
        # Fit scaler on training data only
        logger.debug(_("Ajustando escalador StandardScaler..."))
        X_scaled = self.scaler.fit_transform(X)

        target_columns: List[str]
        n_targets: int
        if isinstance(y, pd.DataFrame):
            target_columns = y.columns.astype(str).tolist()
            n_targets = int(y.shape[1])
        else:
            target_columns = [str(getattr(y, "name", "target"))]
            n_targets = 1
        
        # Store training metadata
        self.metadata = {
            'estimator_type': self._estimator_name,
            'n_features': X.shape[1],
            'n_samples': X.shape[0],
            'training_date': datetime.now().isoformat(),
            'target_columns': target_columns,
            'n_targets': n_targets,
        }
        if isinstance(y, pd.DataFrame):
            self.metadata['target_summary'] = {
                column: {
                    'mean': float(y[column].mean()),
                    'std': float(y[column].std()),
                    'min': float(y[column].min()),
                    'max': float(y[column].max())
                }
                for column in y.columns
            }
        else:
            self.metadata.update({
                'target_mean': float(y.mean()),
                'target_std': float(y.std()),
                'target_min': float(y.min()),
                'target_max': float(y.max())
            })
        
        logger.debug(_("Metadatos de entrenamiento: %d features, %d samples"), 
                    self.metadata['n_features'], self.metadata['n_samples'])
        
        if param_grid:
            logger.info(_("Detectado grid de parámetros. Iniciando búsqueda: %s"), search_strategy.upper())
            self._hyperparameter_search(X_scaled, y, param_grid, cv_folds, search_strategy, n_iter)
        else:
            logger.info(_("No se detectó malla de parámetros. Entrenando con valores por defecto."))
            self.base_estimator.fit(X_scaled, y)
            self.model = self.base_estimator
            logger.debug(_("Modelo entrenado con parámetros por defecto"))
        
        # Mark as trained
        self._is_trained = True
        logger.info(_("✓ Entrenamiento completado exitosamente"))
    
    def _hyperparameter_search(
        self, 
        X_scaled: np.ndarray, 
        y: pd.Series | pd.DataFrame, 
        param_grid: Dict[str, list], 
        cv_folds: Any, 
        strategy: str, 
        n_iter: int
    ) -> None:
        """Performs hyperparameter search using grid or random strategy.
        
        Args:
            X_scaled: Scaled feature matrix
            y: Target vector
            param_grid: Parameter grid for search
            cv_folds: Cross-validation folds
            strategy: 'grid' or 'random'
            n_iter: Number of iterations for random search
        """
        try:
            if strategy.lower() == "random":
                logger.info(_("Iniciando búsqueda ALEATORIA con %d iteraciones máximas"), n_iter)
                search = RandomizedSearchCV(
                    estimator=self.base_estimator,
                    param_distributions=param_grid,
                    n_iter=n_iter,
                    cv=cv_folds,
                    scoring='neg_root_mean_squared_error',
                    n_jobs=-1,
                    random_state=42,
                    verbose=1
                )
            else:
                logger.info(_("Iniciando búsqueda EXHAUSTIVA (Grid Search)"))
                # Calculate total combinations
                total_combos = 1
                for v in param_grid.values():
                    total_combos *= len(v)
                logger.debug(_("Total de combinaciones a evaluar: %d"), total_combos)
                
                search = GridSearchCV(
                    estimator=self.base_estimator,
                    param_grid=param_grid,
                    cv=cv_folds,
                    scoring='neg_root_mean_squared_error',
                    n_jobs=-1,
                    verbose=1
                )
            
            logger.info(_("Ejecutando búsqueda de hiperparámetros..."))
            search.fit(X_scaled, y)
            
            self.model = search.best_estimator_
            self.best_params = search.best_params_
            self.cv_results = search.cv_results_
            
            logger.info(_("✓ Optimización completada. Mejor score: %.4f"), -search.best_score_)
            logger.info(_("Mejores hiperparámetros: %s"), self.best_params)
            
            # Store optimization metadata
            self.metadata['best_params'] = self.best_params
            self.metadata['best_cv_score'] = float(-search.best_score_)
            
        except Exception as e:
            logger.error(_("Error durante búsqueda de hiperparámetros: %s"), str(e))
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Applies the predictive model to new observations while enforcing safety checks.
        
        Before emitting a prediction, this method subjects the incoming data to two 
        crucial transformations: validation against the training boundaries to prevent 
        dangerous clinical extrapolation, and normalization using the scale parameters 
        learned during the training phase.
        
        Args:
            X (pd.DataFrame): Data from patients for whom the paper test score is unknown.
            
        Returns:
            np.ndarray: A one-dimensional array containing the estimation in the original 
                scale of the paper-based test.
                        
        Raises:
            PipelineNotTrainedError: If this method is called before invoking the `train` method.
            ValueError: If input features don't match training features
        """
        self._check_trained()
        if self.feature_names is None:
            raise PipelineNotTrainedError(_("No hay nombres de features disponibles. El modelo no está listo para inferencia."))
        
        # Validate feature compatibility
        if list(X.columns) != self.feature_names:
            logger.warning(_("Advertencia: El orden o nombre de features no coincide con el entrenamiento"))
            X = X[self.feature_names]  # Reorder to match training
        
        # Check boundaries before prediction
        boundary_violations = self.check_boundaries(X)
        if boundary_violations:
            logger.warning(_("⚠️ Se detectaron %d features con valores fuera de rango"), len(boundary_violations))
        
        # Scale and predict
        logger.debug(_("Escalando datos de entrada para predicción..."))
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        logger.debug(_("Predicción completada: %d muestras procesadas"), len(predictions))
        return predictions
    
    def get_feature_importance(self, top_n: int = 10) -> Optional[pd.DataFrame]:
        """Extracts feature importance if available (for tree-based models).
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importances, or None if model doesn't support it
        """
        self._check_trained()
        
        # Check if model has feature_importances_ attribute (tree-based models)
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            logger.info(_("Importancias de features (top %d):"), top_n)
            for idx, row in feature_importance_df.head(top_n).iterrows():
                logger.info(_("  %s: %.4f"), row['feature'], row['importance'])
            
            return feature_importance_df.head(top_n)
        else:
            logger.debug(_("Modelo %s no soporta feature importance"), self._estimator_name)
            return None




class CountErrorClassifierCalibrator(BaseModelCalibrator):
    """Classifier-based calibrator for sparse clinical error counts."""

    def __init__(self, random_state: int = 42):
        super().__init__()
        self.scaler = StandardScaler()
        self.feature_ranges: Dict[str, Tuple[float, float]] = {}
        self.random_state = random_state
        self.classifier: Any = None
        self.class_value_map_: Dict[int, float] = {}
        self.target_name_: str = 'target'
        self.model: Any = None
        self._estimator_name = 'CountErrorClassifier'

    def _build_classifier(self) -> Any:
        return RandomForestClassifier(
            n_estimators=250,
            max_depth=6,
            min_samples_leaf=2,
            random_state=self.random_state,
            class_weight='balanced_subsample',
            n_jobs=-1,
        )

    @staticmethod
    def _fallback_representative(label: int) -> float:
        fallback = {
            0: 0.0,
            1: 1.0,
            2: 2.0,
            3: 4.0,
        }
        return fallback.get(int(label), float(label))

    @staticmethod
    def _resolve_target_name(y: pd.Series | pd.DataFrame) -> str:
        return str(getattr(y, 'name', 'target') or 'target')

    def _encode_error_classes(self, y_values: np.ndarray, target_name: str) -> Tuple[np.ndarray, Dict[int, float]]:
        rounded = np.rint(np.asarray(y_values, dtype=float)).astype(int)
        class_value_map: Dict[int, float] = {}

        if target_name.endswith('errors_a'):
            labels = (rounded > 0).astype(int)
            class_value_map[0] = 0.0
            positive_mask = labels == 1
            if positive_mask.any():
                class_value_map[1] = float(max(1, int(np.rint(np.median(rounded[positive_mask])))))
            else:
                class_value_map[1] = 1.0
            return labels, class_value_map

        labels = np.zeros_like(rounded, dtype=int)
        labels[rounded == 1] = 1
        labels[(rounded >= 2) & (rounded <= 3)] = 2
        labels[rounded >= 4] = 3

        for label in np.unique(labels):
            mask = labels == label
            if mask.any():
                class_value_map[int(label)] = float(max(0, int(np.rint(np.median(rounded[mask])))))

        for label in (0, 1, 2, 3):
            class_value_map.setdefault(label, self._fallback_representative(label))

        return labels, class_value_map

    def _fit_classifier(self, X: np.ndarray, labels: np.ndarray) -> Any:
        if np.unique(labels).size < 2:
            classifier = DummyClassifier(strategy='constant', constant=int(labels[0]) if len(labels) else 0)
        else:
            classifier = self._build_classifier()
        classifier.fit(X, labels)
        return classifier

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series | pd.DataFrame,
        param_grid: Optional[Dict[str, list]] = None,
        cv_folds: Any = 3,
        search_strategy: str = 'grid',
        n_iter: int = 10,
    ) -> None:
        del param_grid, cv_folds, search_strategy, n_iter
        if isinstance(y, pd.DataFrame):
            raise ValueError(_("El clasificador de errores solo soporta targets single-output."))
        super().train(X, y)

        logger.info(_("▶️ Iniciando fase de entrenamiento para modelo: %s"), self._estimator_name)
        self.feature_ranges = {col: (float(X[col].min()), float(X[col].max())) for col in X.columns}
        X_scaled = self.scaler.fit_transform(X)
        y_values = np.asarray(y, dtype=float).reshape(-1)
        self.target_name_ = self._resolve_target_name(y)
        labels, self.class_value_map_ = self._encode_error_classes(y_values, self.target_name_)
        self.classifier = self._fit_classifier(X_scaled, labels)

        self.model = {
            'classifier': self.classifier,
            'class_value_map': self.class_value_map_,
            'target_name': self.target_name_,
        }
        self.metadata = {
            'estimator_type': self._estimator_name,
            'n_features': X.shape[1],
            'n_samples': X.shape[0],
            'training_date': datetime.now().isoformat(),
            'target_columns': [self.target_name_],
            'n_targets': 1,
            'class_distribution': {
                str(label): int((labels == label).sum()) for label in np.unique(labels)
            },
            'class_value_map': {str(label): float(value) for label, value in self.class_value_map_.items()},
        }
        self._is_trained = True
        logger.info(_("✓ Entrenamiento completado exitosamente"))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_trained()
        if self.feature_names is None:
            raise PipelineNotTrainedError(_("No hay nombres de features disponibles. El modelo no está listo para inferencia."))
        if list(X.columns) != self.feature_names:
            logger.warning(_("Advertencia: El orden o nombre de features no coincide con el entrenamiento"))
            X = X[self.feature_names]

        boundary_violations = self.check_boundaries(X)
        if boundary_violations:
            logger.warning(_("⚠️ Se detectaron %d features con valores fuera de rango"), len(boundary_violations))

        X_scaled = self.scaler.transform(X)
        predicted_labels = np.asarray(self.classifier.predict(X_scaled), dtype=int).reshape(-1)
        predictions = np.array(
            [self.class_value_map_.get(int(label), self._fallback_representative(int(label))) for label in predicted_labels],
            dtype=float,
        )
        return predictions


class StackingEnsembleCalibrator(BaseModelCalibrator):
    """Clinical stacking ensemble combining Ridge, Random Forest and XGBoost."""

    def __init__(self, random_state: int = 42):
        super().__init__()
        self.scaler = StandardScaler()
        self.feature_ranges: Dict[str, Tuple[float, float]] = {}
        self.random_state = random_state
        self.base_estimators: Dict[str, Any] = {}
        self.meta_model: Optional[Ridge] = None
        self.model: Any = None
        self._estimator_name = 'StackingEnsemble'

    def _make_base_estimators(self, multi_output: bool) -> Dict[str, Any]:
        ridge = Ridge(alpha=1.0, random_state=self.random_state)
        rf = RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1,
        )
        xgb_estimator = xgb.XGBRegressor(
            random_state=self.random_state,
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective='reg:squarederror',
            n_jobs=-1,
        )
        if multi_output:
            xgb_estimator = MultiOutputRegressor(xgb_estimator)

        return {
            'ridge': ridge,
            'rf': rf,
            'xgboost': xgb_estimator,
        }

    @staticmethod
    def _to_prediction_matrix(values: Any) -> np.ndarray:
        predictions = np.asarray(values, dtype=float)
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        return predictions

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series | pd.DataFrame,
        param_grid: Optional[Dict[str, list]] = None,
        cv_folds: Any = 3,
        search_strategy: str = 'grid',
        n_iter: int = 10,
    ) -> None:
        del param_grid, search_strategy, n_iter
        super().train(X, y)

        logger.info(_("▶️ Iniciando fase de entrenamiento para modelo: %s"), self._estimator_name)
        self.feature_ranges = {
            col: (float(X[col].min()), float(X[col].max())) for col in X.columns
        }
        X_scaled = self.scaler.fit_transform(X)
        y_frame = y if isinstance(y, pd.DataFrame) else pd.DataFrame({'target': y})
        y_values = y_frame.to_numpy(dtype=float)
        multi_output = y_values.shape[1] > 1

        if isinstance(cv_folds, int):
            raise ValueError(_("El ensemble requiere particiones explícitas de validación cruzada agrupada."))
        splits = list(cv_folds)
        if len(splits) < 2:
            raise ValueError(_("El ensemble requiere al menos 2 folds para construir predicciones out-of-fold."))

        self.base_estimators = self._make_base_estimators(multi_output=multi_output)
        oof_predictions = {
            name: np.full((len(X), y_values.shape[1]), np.nan, dtype=float)
            for name in self.base_estimators
        }

        for fold_idx, (train_idx, valid_idx) in enumerate(splits, start=1):
            logger.debug(_("Entrenando fold %d/%d del ensemble"), fold_idx, len(splits))
            X_fold_train = X_scaled[train_idx]
            X_fold_valid = X_scaled[valid_idx]
            if multi_output:
                y_fold_train = y_values[train_idx]
            else:
                y_fold_train = y_values[train_idx, 0]

            for name, estimator in self.base_estimators.items():
                fold_estimator = clone(estimator)
                fold_estimator.fit(X_fold_train, y_fold_train)
                fold_predictions = self._to_prediction_matrix(fold_estimator.predict(X_fold_valid))
                oof_predictions[name][valid_idx] = fold_predictions

        stacked_blocks = [oof_predictions[name] for name in self.base_estimators]
        meta_X = np.hstack(stacked_blocks)
        if np.isnan(meta_X).any():
            raise ValueError(_("El ensemble produjo predicciones out-of-fold incompletas. Revise los folds agrupados."))

        self.meta_model = Ridge(alpha=1.0, random_state=self.random_state)
        self.meta_model.fit(meta_X, y_values)

        for name, estimator in self.base_estimators.items():
            if multi_output:
                estimator.fit(X_scaled, y_values)
            else:
                estimator.fit(X_scaled, y_values[:, 0])

        self.model = {
            'base_estimators': self.base_estimators,
            'meta_model': self.meta_model,
        }
        target_columns = y_frame.columns.astype(str).tolist()
        self.metadata = {
            'estimator_type': self._estimator_name,
            'n_features': X.shape[1],
            'n_samples': X.shape[0],
            'training_date': datetime.now().isoformat(),
            'target_columns': target_columns,
            'n_targets': int(y_values.shape[1]),
            'base_estimators': list(self.base_estimators.keys()),
            'stacking_cv_folds': len(splits),
        }
        self._is_trained = True
        logger.info(_("✓ Entrenamiento completado exitosamente"))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_trained()
        if self.feature_names is None:
            raise PipelineNotTrainedError(_("No hay nombres de features disponibles. El modelo no está listo para inferencia."))
        if list(X.columns) != self.feature_names:
            logger.warning(_("Advertencia: El orden o nombre de features no coincide con el entrenamiento"))
            X = X[self.feature_names]

        boundary_violations = self.check_boundaries(X)
        if boundary_violations:
            logger.warning(_("⚠️ Se detectaron %d features con valores fuera de rango"), len(boundary_violations))

        X_scaled = self.scaler.transform(X)
        stacked_blocks = []
        for estimator in self.base_estimators.values():
            stacked_blocks.append(self._to_prediction_matrix(estimator.predict(X_scaled)))
        meta_X = np.hstack(stacked_blocks)
        predictions = self.meta_model.predict(meta_X)
        predictions = np.asarray(predictions, dtype=float)
        if predictions.ndim == 2 and predictions.shape[1] == 1:
            return predictions.ravel()
        return predictions


class SplineGAMCalibrator(BaseModelCalibrator):
    """Approximate GAM calibrator using spline bases plus Ridge regularization."""

    def __init__(
        self,
        random_state: int = 42,
        n_knots: int = 5,
        degree: int = 3,
        alpha: float = 1.0,
        min_unique_for_spline: int = 6,
    ):
        super().__init__()
        self.scaler = StandardScaler()
        self.feature_ranges: Dict[str, Tuple[float, float]] = {}
        self.random_state = random_state
        self.n_knots = n_knots
        self.degree = degree
        self.alpha = alpha
        self.min_unique_for_spline = min_unique_for_spline
        self.model: Any = None
        self.best_params: Dict = {}
        self.cv_results: Dict = {}
        self._estimator_name = 'SplineGAM'
        self._spline_columns: List[str] = []
        self._linear_columns: List[str] = []

    def _split_feature_roles(self, X: pd.DataFrame) -> Tuple[List[str], List[str]]:
        spline_columns: List[str] = []
        linear_columns: List[str] = []
        for column in X.columns:
            series = pd.Series(X[column]).dropna()
            unique_values = int(series.nunique())
            if unique_values >= self.min_unique_for_spline:
                spline_columns.append(column)
            else:
                linear_columns.append(column)
        if not spline_columns:
            linear_columns = X.columns.astype(str).tolist()
        return spline_columns, linear_columns

    def _build_estimator(self, X: pd.DataFrame) -> Pipeline:
        self._spline_columns, self._linear_columns = self._split_feature_roles(X)
        transformers = []
        if self._spline_columns:
            transformers.append((
                'spline',
                SplineTransformer(
                    n_knots=self.n_knots,
                    degree=self.degree,
                    include_bias=False,
                ),
                self._spline_columns,
            ))
        if self._linear_columns:
            transformers.append(('linear', 'passthrough', self._linear_columns))

        features = ColumnTransformer(transformers=transformers, remainder='drop')
        return Pipeline([
            ('features', features),
            ('ridge', Ridge(alpha=self.alpha, random_state=self.random_state)),
        ])

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series | pd.DataFrame,
        param_grid: Optional[Dict[str, list]] = None,
        cv_folds: Any = 3,
        search_strategy: str = "grid",
        n_iter: int = 10
    ) -> None:
        super().train(X, y)

        logger.info(_("▶️ Iniciando fase de entrenamiento para modelo: %s"), self._estimator_name)
        self.feature_ranges = {
            col: (float(X[col].min()), float(X[col].max())) for col in X.columns
        }
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index,
        )
        estimator = self._build_estimator(X_scaled)

        target_columns: List[str]
        n_targets: int
        if isinstance(y, pd.DataFrame):
            target_columns = y.columns.astype(str).tolist()
            n_targets = int(y.shape[1])
        else:
            target_columns = [str(getattr(y, "name", "target"))]
            n_targets = 1

        self.metadata = {
            'estimator_type': self._estimator_name,
            'n_features': X.shape[1],
            'n_samples': X.shape[0],
            'training_date': datetime.now().isoformat(),
            'target_columns': target_columns,
            'n_targets': n_targets,
            'spline_columns': self._spline_columns,
            'linear_columns': self._linear_columns,
        }

        if param_grid:
            translated_param_grid = {
                key if '__' in key else f'ridge__{key}': value
                for key, value in param_grid.items()
            }
            if search_strategy.lower() == "random":
                search = RandomizedSearchCV(
                    estimator=estimator,
                    param_distributions=translated_param_grid,
                    n_iter=n_iter,
                    cv=cv_folds,
                    scoring='neg_root_mean_squared_error',
                    n_jobs=-1,
                    random_state=self.random_state,
                    verbose=1
                )
            else:
                search = GridSearchCV(
                    estimator=estimator,
                    param_grid=translated_param_grid,
                    cv=cv_folds,
                    scoring='neg_root_mean_squared_error',
                    n_jobs=-1,
                    verbose=1
                )
            search.fit(X_scaled, y)
            self.model = search.best_estimator_
            self.best_params = search.best_params_
            self.cv_results = search.cv_results_
            self.metadata['best_params'] = self.best_params
            self.metadata['best_cv_score'] = float(-search.best_score_)
        else:
            estimator.fit(X_scaled, y)
            self.model = estimator

        self._is_trained = True
        logger.info(_("✓ Entrenamiento completado exitosamente"))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_trained()
        if self.feature_names is None:
            raise PipelineNotTrainedError(_("No hay nombres de features disponibles. El modelo no está listo para inferencia."))
        if list(X.columns) != self.feature_names:
            logger.warning(_("Advertencia: El orden o nombre de features no coincide con el entrenamiento"))
            X = X[self.feature_names]

        boundary_violations = self.check_boundaries(X)
        if boundary_violations:
            logger.warning(_("⚠️ Se detectaron %d features con valores fuera de rango"), len(boundary_violations))

        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index,
        )
        predictions = self.model.predict(X_scaled)
        predictions = np.asarray(predictions, dtype=float)
        if predictions.ndim == 2 and predictions.shape[1] == 1:
            return predictions.ravel()
        return predictions


def get_model(model_type: str, multi_output: bool = False, **kwargs) -> BaseModelCalibrator:
    """Implementation of the Factory pattern for dynamic instantiation of regressors."""
    logger.info(_("Creando modelo factory para tipo: %s"), model_type.upper())

    model_type_lower = model_type.lower()
    if model_type_lower == 'ensemble':
        return StackingEnsembleCalibrator(random_state=int(kwargs.pop('random_state', 42)))
    if model_type_lower == 'error_classifier':
        return CountErrorClassifierCalibrator(random_state=int(kwargs.pop('random_state', 42)))
    if model_type_lower == 'gam':
        return SplineGAMCalibrator(
            random_state=int(kwargs.pop('random_state', 42)),
            n_knots=int(kwargs.pop('n_knots', 5)),
            degree=int(kwargs.pop('degree', 3)),
            alpha=float(kwargs.pop('alpha', 1.0)),
            min_unique_for_spline=int(kwargs.pop('min_unique_for_spline', 6)),
        )

    models = {
        "linear": LinearRegression(**kwargs),
        "logistic": LogisticRegression(random_state=42, **kwargs),
        "ridge": Ridge(random_state=42, **kwargs),
        "svm": SVR(**kwargs),
        "rf": RandomForestRegressor(random_state=42, n_jobs=-1, **kwargs),
        "xgboost": xgb.XGBRegressor(random_state=42, n_jobs=-1, objective='reg:squarederror', **kwargs),
    }

    if model_type_lower not in models:
        available = ", ".join(list(models.keys()) + ['gam', 'ensemble', 'error_classifier'])
        raise ValueError(
            _("Identificador de modelo '%s' no reconocido. Opciones disponibles: %s") % (model_type, available)
        )

    estimator = models[model_type_lower]
    if multi_output and not isinstance(estimator, (LinearRegression, Ridge, RandomForestRegressor)):
        estimator = MultiOutputRegressor(estimator)
    logger.debug(_("Estimador %s creado exitosamente"), type(estimator).__name__)

    return StandardCalibrator(estimator)
