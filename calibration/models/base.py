import joblib
import logging
from abc import ABC, abstractmethod
from pydantic import BaseModel, ValidationError
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any, Optional
from pathlib import Path
from calibration.i18n import get_translator

# Configuración del registrador local para este submódulo
logger = logging.getLogger(__name__)

# Configuración de traducción con soporte de contingencia (fallback)
_ = get_translator()


class ModelMetrics(BaseModel):
    """Data validation model for regression metrics using Pydantic.

    This class ensures strict typing and structural validation for the 
    performance metrics calculated after model evaluation. Using Pydantic 
    guarantees that down-stream reporting tools receive clean, predictable data.

    Attributes:
        rmse (float): Root Mean Squared Error. Indicates the absolute fit
            of the model to the data (how close the observed data points
            are to the model's predicted values).
        r2_score (float): Coefficient of determination. Represents the
            proportion of the variance in the dependent variable that is
            predictable from the independent variables.
        mae (float, optional): Mean Absolute Error.
        mape (float, optional): Mean Absolute Percentage Error.
    """
    rmse: float
    r2_score: float
    mae: Optional[float] = None
    mape: Optional[float] = None


class PipelineNotTrainedError(RuntimeError):
    """Exception raised when attempting operations on untrained pipeline."""
    pass


class BaseModelCalibrator(ABC):
    """Abstract Base Class defining the contract for all clinical calibrators.

    This class establishes the fundamental architecture for any predictive
    model incorporated into the system, whether it is a classical Machine
    Learning algorithm (like scikit-learn) or a Deep Learning model (like
    PyTorch). It enforces the implementation of core methods and provides
    shared utilities for safe clinical inference and model persistence.

    Attributes:
        model (Any): The underlying predictive estimator. It is initialized
            as None and must be populated during the `train` method execution.
        scaler (Any): The pre-processing object used to normalize features.
        feature_ranges (Dict[str, Tuple[float, float]]): A dictionary mapping
            feature names to their observed (min, max) values during training.
            Crucial for preventing silent extrapolation during inference.
        metadata (Dict): Training metadata including model type, date, feature names
        n_features (int): Number of features the model was trained on
        feature_names (List[str]): Names of features for validation
    """

    def __init__(self) -> None:
        """Initializes the base attributes of the calibrator.
        
        The internal state is kept empty until the specific implementation 
        injects the algorithm and processes the training data.
        """
        self.model: Any = None
        self.scaler: Any = None
        self.feature_ranges: Dict[str, Tuple[float, float]] = {}
        self.metadata: Dict[str, Any] = {}
        self.n_features: int = 0
        self.feature_names: Optional[list] = None
        self._is_trained: bool = False

    def _check_trained(self) -> None:
        """Validates that the model has been trained.
        
        Raises:
            PipelineNotTrainedError: If model is not trained
        """
        if not self._is_trained or self.model is None or self.scaler is None:
            raise PipelineNotTrainedError(
                _("El modelo no puede ser utilizado porque el pipeline de entrenamiento no ha finalizado.")
            )

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs: Any) -> None:
        """Trains the underlying estimator and prepares the pre-processing pipeline.

        This is an abstract method. Any class inheriting from BaseModelCalibrator
        MUST implement this method. It is expected to handle feature scaling,
        hyperparameter optimization, and the final fitting of the model.

        Args:
            X (pd.DataFrame): Training feature matrix.
            y (pd.Series): Training target vector representing the gold standard.
            **kwargs (Any): Additional dynamic arguments such as parameter grids,
                cross-validation strategies, search iterations, etc.

        Raises:
            NotImplementedError: If the child class fails to implement this method.
            ValueError: If input data is invalid
        """
        # Validate input
        if X.empty or y.empty:
            raise ValueError(_("Las matrices X e y no pueden estar vacías"))
        
        if len(X) != len(y):
            raise ValueError(_("X e y deben tener la misma cantidad de muestras"))
        
        if X.isnull().any().any():
            raise ValueError(_("X contiene valores nulos. Pre-procese los datos antes del entrenamiento"))
        
        # Store feature information
        self.feature_names = X.columns.tolist()
        self.n_features = X.shape[1]
        logger.info(_("Inicializando entrenamiento con %d features: %s"), self.n_features, self.feature_names[:3])
        pass

    def check_boundaries(self, X: pd.DataFrame) -> Dict[str, list]:
        """Validates that incoming data falls within historically safe limits.

        To ensure clinical safety, this method inspects each feature of the
        incoming dataset. If any value exceeds the minimum or maximum values
        observed during the training phase, it triggers a warning mechanism.
        This prevents the model from making silent, potentially erroneous
        predictions on data distributions it has never seen (extrapolation).

        Args:
            X (pd.DataFrame): The new feature matrix to be evaluated prior to inference.

        Returns:
            Dict with feature names and out-of-range values

        Raises:
            ValueError: If feature count doesn't match training
        """
        if X.shape[1] != self.n_features:
            raise ValueError(
                _("Número de features inconsistente. Esperado: %d, Recibido: %d") % (self.n_features, X.shape[1])
            )
        
        out_of_range_features = {}
        
        for col in X.columns:
            if col in self.feature_ranges:
                min_val, max_val = self.feature_ranges[col]
                
                # Check boundaries
                below_min = X[col] < min_val
                above_max = X[col] > max_val
                
                if below_min.any() or above_max.any():
                    out_of_range_features[col] = {
                        'min_expected': min_val,
                        'max_expected': max_val,
                        'min_observed': float(X[col].min()),
                        'max_observed': float(X[col].max()),
                        'n_violations': int((below_min | above_max).sum())
                    }
                    
                    logger.warning(
                        _("⚠️ Alerta clinica: Valores fuera del rango de entrenamiento en '%s'. Entrenamiento: [%.2f, %.2f], Observado: [%.2f, %.2f] (%d violaciones)"),
                        col, min_val, max_val, X[col].min(), X[col].max(), (below_min | above_max).sum()
                    )
            else:
                logger.debug(_("Caracteristica '%s' no encontrada en los rangos de entrenamiento base."), col)
        
        return out_of_range_features

    def save(self, filepath: str, include_metadata: bool = True) -> None:
        """Serializes and saves the complete model pipeline to disk.

        This method exports not just the final mathematical model, but also
        the fitted scaler, safety boundaries dictionary, and metadata. This ensures
        that when the model is loaded in the future, it retains exact knowledge
        of how to pre-process new data and when to issue extrapolation warnings.

        Args:
            filepath (str): The destination path (including filename) where
                the binary object will be saved.
            include_metadata (bool): Whether to save metadata alongside the model

        Raises:
            PipelineNotTrainedError: If attempting to save an untrained model
            IOError: If there are permission or path issues writing the file to disk
        """
        self._check_trained()
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        logger.info(_("Guardando pipeline completo (modelo, escalador y limites de seguridad) en: %s"), filepath)
        
        # Pack all necessary components into a single dictionary for persistence
        pipeline_state = {
            'model': self.model,
            'scaler': self.scaler,
            'ranges': self.feature_ranges,
            'n_features': self.n_features,
            'feature_names': self.feature_names,
            'metadata': self.metadata
        }
        
        try:
            joblib.dump(pipeline_state, filepath)
            logger.info(_("✓ Pipeline guardado exitosamente"))
        except IOError as e:
            logger.error(_("Error al guardar pipeline: %s"), str(e))
            raise

    @classmethod
    def load(cls, filepath: str) -> 'BaseModelCalibrator':
        """Loads a previously trained model pipeline from disk.

        Args:
            filepath (str): Path to the saved pipeline file

        Returns:
            BaseModelCalibrator: Loaded model instance

        Raises:
            FileNotFoundError: If file does not exist
            RuntimeError: If pipeline file is corrupted
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(_("Archivo de modelo no encontrado: %s") % filepath)
        
        logger.info(_("Cargando pipeline desde: %s"), filepath)
        
        try:
            pipeline_state = joblib.load(filepath)
            logger.info(_("✓ Pipeline cargado exitosamente"))
            return pipeline_state
        except Exception as e:
            logger.error(_("Error al cargar pipeline: %s"), str(e))
            raise RuntimeError(_("No se pudo cargar el archivo de modelo. Archivo corrupto.")) from e
