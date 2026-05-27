import logging
import gettext
from typing import Any, Dict, Optional, List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from .base import BaseModelCalibrator

logger = logging.getLogger(__name__)
translation = gettext.translation('messages', localedir='locales', fallback=True)
_ = translation.gettext

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
        Firstly, it explores the training dataset to record safety boundaries for all features.
        Secondly, it fits and applies standard scaling exclusively on the training data.
        Thirdly, it evaluates whether hyperparameter tuning is required based on the presence 
        of a parameter grid.
        Fourthly, if required, it branches between a randomized search for computational 
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
        logger.info(_("Iniciando fase de entrenamiento y pre-procesamiento de datos."))
        
        self.feature_ranges = {
            col: (float(X[col].min()), float(X[col].max())) for col in X.columns
        }
        
        X_scaled = self.scaler.fit_transform(X)
        
        if param_grid:
            if search_strategy.lower() == "random":
                logger.info(_("Iniciando búsqueda ALEATORIA en espacio de parámetros con %d intentos máximos."), n_iter)
                search = RandomizedSearchCV(
                    estimator=self.base_estimator,
                    param_distributions=param_grid,
                    n_iter=n_iter,
                    cv=cv_folds,
                    scoring='neg_root_mean_squared_error',
                    n_jobs=-1,
                    random_state=42
                )
            else:
                logger.info(_("Iniciando búsqueda EXHAUSTIVA (Grid) en el espacio de parámetros."))
                search = GridSearchCV(
                    estimator=self.base_estimator,
                    param_grid=param_grid,
                    cv=cv_folds,
                    scoring='neg_root_mean_squared_error',
                    n_jobs=-1
                )
            
            search.fit(X_scaled, y)
            
            self.model = search.best_estimator_
            logger.info(_("Optimización concluida. Mejores hiperparámetros: %s"), search.best_params_)
            
        else:
            logger.info(_("No se detectó malla de parámetros. Entrenando modelo de forma determinista con valores por defecto."))
            self.base_estimator.fit(X_scaled, y)
            self.model = self.base_estimator
            
        logger.info(_("El estimador predictivo se ha ajustado exitosamente y está listo para inferencia."))

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
            RuntimeError: If this method is called before invoking the `train` method.
        """
        if self.model is None:
            raise RuntimeError(_("No se puede inferir: el modelo aún no ha sido ajustado. Llame a train() primero."))
            
        self.check_boundaries(X)
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict(X_scaled)


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
    models = {
        "linear": LinearRegression(**kwargs),
        "logistic": LogisticRegression(random_state=42, **kwargs),
        "ridge": Ridge(random_state=42, **kwargs),
        "svm": SVR(**kwargs),
        "rf": RandomForestRegressor(random_state=42, n_jobs=-1, **kwargs),
        "xgboost": xgb.XGBRegressor(random_state=42, n_jobs=-1, **kwargs)
    }
    
    if model_type not in models:
        raise ValueError(_("Identificador de modelo no válido: '%s'. Opciones disponibles: %s") % 
                         (model_type, list(models.keys())))
        
    return StandardCalibrator(models[model_type])
