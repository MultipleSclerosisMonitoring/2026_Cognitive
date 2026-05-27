import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GroupKFold
import logging
import gettext
from typing import Tuple, Any

logger = logging.getLogger(__name__)
translation = gettext.translation('messages', localedir='locales', fallback=True)
_ = translation.gettext

class DataProcessor:
    """Handles the extraction, merging, and partitioning of clinical datasets.

    This class is responsible for connecting to the PostgreSQL database to retrieve
    the digital biomarker features and merging them with the traditional clinical
    scores (gold standard) stored in an external Excel or CSV file. It also ensures
    that the data is correctly split for training and testing, implementing grouped
    cross-validation strategies to prevent data leakage across different test
    sessions belonging to the same patient.

    Attributes:
        db_uri (str): The connection string for the PostgreSQL database.
        excel_path (str): The file path to the clinical demographics and paper scores.
        engine (sqlalchemy.engine.Engine): The active database connection engine.
    """

    def __init__(self, db_uri: str, excel_path: str) -> None:
        """Initializes the DataProcessor with database and file credentials.

        Args:
            db_uri (str): Connection URI for PostgreSQL (e.g., 'postgresql://user:pass@host/db').
            excel_path (str): Path to the Excel file containing the paper-based test scores.
        """
        self.db_uri = db_uri
        self.excel_path = excel_path
        self.engine = create_engine(self.db_uri)

    def load_and_merge(self, test_type: str) -> pd.DataFrame:
        """Retrieves and merges digital and clinical data based on patient IDs.

        This method attempts to fetch the digital test data directly from the
        database. If the connection fails, it implements a fallback mechanism to
        read from a local CSV dump. It then standardizes the patient identification
        codes across both the digital and clinical datasets to perform an inner join,
        ensuring only complete records are kept for the modeling phase.

        Args:
            test_type (str): The specific cognitive test to process (e.g., 'sdmt' or 'tmt').
                             This determines which database table to query.

        Returns:
            pd.DataFrame: A unified pandas DataFrame containing both the digital
                          features and the clinical gold standard targets.
        """
        logger.info(_("Conectando a la base de datos PostgreSQL para el test %s..."), test_type)
        
        try:
            df_digital = pd.read_sql_table(test_type.lower(), self.engine)
        except Exception as e:
            logger.warning(_("No se pudo conectar a la BD, intentando leer desde CSV local como respaldo: %s"), str(e))
            df_digital = pd.read_csv(f"{test_type.lower()}.csv")
        
        logger.info(_("Cargando datos clinicos..."))
        try:
            df_clinical = pd.read_excel(self.excel_path)
        except Exception:
            df_clinical = pd.read_csv(self.excel_path)
            
        df_digital['patient_id'] = df_digital['codeid'].astype(str).str.strip().str.upper()
        
        if 'Código' in df_clinical.columns:
            df_clinical['patient_id'] = df_clinical['Código'].astype(str).str.strip().str.upper()
        else:
            df_clinical['patient_id'] = df_clinical.iloc[:, 0].astype(str).str.strip().str.upper()
            
        logger.info(_("Fusionando conjuntos de datos por identificador de paciente..."))
        df_merged = pd.merge(df_digital, df_clinical, on='patient_id', how='inner')
        
        return df_merged

    def prepare_splits(self, df: pd.DataFrame, test_type: str, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Cleans the dataset and performs the initial train-test partitioning.

        This method identifies the appropriate target column depending on the test
        type, drops missing values in the target, and removes non-predictive metadata
        columns (like dates, IDs, and timestamps). It then splits the data into a
        training set for model optimization and a completely blind test set for
        final evaluation.

        Args:
            df (pd.DataFrame): The merged dataset containing features and targets.
            test_type (str): The test identifier to dynamically select columns.
            test_size (float, optional): The proportion of the dataset to include
                                         in the test split. Defaults to 0.2.

        Returns:
            Tuple containing:
                - X_train (pd.DataFrame): Features for the training set.
                - X_test (pd.DataFrame): Features for the hold-out test set.
                - y_train (pd.Series): Target values for the training set.
                - y_test (pd.Series): Target values for the test set.
                - groups_train (pd.Series): Patient IDs corresponding to the training
                  set, necessary for grouped cross-validation.
        """
        logger.info(_("Preparando particiones de entrenamiento y prueba..."))
        
        if test_type.lower() == 'sdmt':
            target_col_hint = 'Día 1 SDMT Papel score'
            features_to_drop = ['id', 'codeid', 'f_nacim', 'fecha', 'ts_created', 'ts_updated', 'patient_id', 'sset']
        else:
            target_col_hint = 'DIA 1       TMT    papel       (tiempo)             A            B  '
            features_to_drop = ['id', 'codeid', 'f_nacimiento', 'date_data', 'ts_created', 'ts_updated', 'patient_id', 'time_complete_a', 'time_complete_b']
            
        target_col = [col for col in df.columns if target_col_hint.strip() in col.strip() or 'Papel' in col or 'papel' in col][0]
            
        df_clean = df.dropna(subset=[target_col])
        y = df_clean[target_col]
        
        X = df_clean.select_dtypes(include=['number']).drop(columns=[target_col], errors='ignore')
        available_columns = X.columns.tolist()
        cols_to_drop = [col for col in features_to_drop if col in available_columns]
        X = X.drop(columns=cols_to_drop, errors='ignore')
        
        groups = df_clean['patient_id']
        
        indices = range(len(df_clean))
        train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=42)
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_train = groups.iloc[train_idx]
        
        return X_train, X_test, y_train, y_test, groups_train

    def get_cv_folds(self, groups_train: pd.Series, n_splits: int = 3) -> Any:
        """Generates grouped cross-validation splits to ensure strict clinical isolation.

        Standard K-Fold cross-validation can inadvertently place data from the same
        patient in both the training and validation sets, causing the model to
        memorize the patient rather than generalizing to the disease patterns. This
        method utilizes GroupKFold to guarantee that all records belonging to a
        specific patient are strictly confined to either the train or validation
        split during any single iteration.

        Args:
            groups_train (pd.Series): An array representing the group identifier
                                      (patient ID) for each sample in the training set.
            n_splits (int, optional): The number of folds to generate. Defaults to 3.

        Returns:
            Any: An iterator yielding train and validation indices for each fold.
        """
        logger.info(_("Generando particiones de validacion cruzada agrupada..."))
        gkf = GroupKFold(n_splits=n_splits)
        return gkf.split(X=groups_train, y=groups_train, groups=groups_train)
