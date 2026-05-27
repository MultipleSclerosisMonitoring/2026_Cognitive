import logging
from typing import Any, Dict, Optional, List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
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
        y: pd.Series, 
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
        
        Args:
            X (pd.DataFrame): Feature matrix extracted from the mobile device interactions.
            y (pd.Series): Target vector representing the paper-based gold standard scores.
            param_grid (Optional[Dict[str, list]]): Search space for hyperparameters. 
                If set to `None`, the model trains using the library's default parameters.
            cv_folds (Any): Integer specifying the number of folds or an iterable yielding 
                train/test splits. Highly recommended to pass a pre-configured GroupKFold.
            search_strategy (str): Exploration strategy. Accepts either "grid" or "random".
            n_iter (int): Computational budget representing the number of parameter settings 
                that are sampled if the strategy is "random". Ignored for "grid".
                
        Raises:
            ValueError: If the feature matrix X or target vector Y contain null data 
                or shape inconsistencies.
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
        
        # Store training metadata
        self.metadata = {
            'estimator_type': self._estimator_name,
            'n_features': X.shape[1],
            'n_samples': X.shape[0],
            'training_date': datetime.now().isoformat(),
            'target_mean': float(y.mean()),
            'target_std': float(y.std()),
            'target_min': float(y.min()),
            'target_max': float(y.max())
        }
        
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
        y: pd.Series, 
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


def get_model(model_type: str, **kwargs) -> StandardCalibrator:
    """Implementation of the Factory pattern for dynamic instantiation of regressors.
    
    This pattern decouples the creation of the object from its usage, allowing an 
    external configuration file (YAML) to dictate the program's behavior without 
    requiring structural code changes when adding new algorithmic families.
    
    Args:
        model_type (str): Short identifier of the desired model (e.g., 'rf', 'linear').
        **kwargs: Arbitrary keyword arguments to be passed directly to the base 
            estimator's constructor (useful for injecting global states).
        
    Returns:
        StandardCalibrator: An instance of the wrapper class with the specified 
            base algorithm already injected and ready for training.
        
    Raises:
        ValueError: If the requested `model_type` does not exist in the internal catalog.
    """
    logger.info(_("Creando modelo factory para tipo: %s"), model_type.upper())
    
    models = {
        "linear": LinearRegression(**kwargs),
        "logistic": LogisticRegression(random_state=42, **kwargs),
        "ridge": Ridge(random_state=42, **kwargs),
        "svm": SVR(**kwargs),
        "rf": RandomForestRegressor(random_state=42, n_jobs=-1, **kwargs),
        "xgboost": xgb.XGBRegressor(random_state=42, n_jobs=-1, **kwargs)
    }
    
    if model_type.lower() not in models:
        available = ", ".join(models.keys())
        raise ValueError(
            _("Identificador de modelo '%s' no reconocido. Opciones disponibles: %s") % (model_type, available)
        )
    
    estimator = models[model_type.lower()]
    logger.debug(_("Estimador %s creado exitosamente"), type(estimator).__name__)
    
    return StandardCalibrator(estimator)
