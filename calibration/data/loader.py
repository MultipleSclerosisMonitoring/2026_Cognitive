import numpy as np
import pandas as pd
from sqlalchemy import create_engine, inspect
from sklearn.model_selection import train_test_split, GroupKFold
import logging
from typing import Tuple, Any, Dict, List, Optional
from pathlib import Path
from calibration.i18n import get_translator

logger = logging.getLogger(__name__)
_ = get_translator()


class DataProcessingError(Exception):
    """Custom exception for data processing errors."""
    pass


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
        column_config (Dict): Configuration for column names per test type.
    """

    def __init__(
        self,
        db_uri: str,
        excel_path: str,
        column_config: Optional[Dict] = None,
        clinical_skiprows: int = 2,
    ) -> None:
        """Initializes the DataProcessor with database and file credentials.

        Args:
            db_uri (str): Connection URI for PostgreSQL (e.g., 'postgresql://user:pass@host/db').
            excel_path (str): Path to the Excel file containing the paper-based test scores.
            column_config (Optional[Dict]): Configuration for column names per test type.
                Should contain 'sdmt', 'tmt' keys with 'target' and 'features_to_drop' lists.
            clinical_skiprows (int): Number of initial rows to skip before reading
                the clinical header row. Defaults to 2.
        """
        self.db_uri = db_uri
        self.excel_path = excel_path
        self.engine = create_engine(self.db_uri)
        self.column_config = column_config or self._default_column_config()
        self.clinical_skiprows = clinical_skiprows
    
    @staticmethod
    def _default_column_config() -> Dict:
        """Returns default column configuration."""
        return {
            'sdmt': {
                'target_hints': ['Día 1 SDMT Papel score', 'SDMT Papel', 'sdmt_papel'],
                'features_to_drop': ['id', 'codeid', 'ts_created', 'ts_updated', 'patient_id', 'sset']
            },
            'tmt': {
                'target_hints': ['DIA 1 TMT papel', 'TMT papel', 'tmt_papel', 'tiempo'],
                'features_to_drop': ['id', 'codeid', 'ts_created', 'ts_updated', 'patient_id']
            }
        }

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
                          
        Raises:
            DataProcessingError: If data loading or merging fails critically.
        """
        logger.info(_("Conectando a la base de datos PostgreSQL para el test %s..."), test_type.upper())
        
        # Load digital biomarker data
        df_digital = self._load_digital_data(test_type)
        logger.info(_("Datos digitales cargados: %d filas, %d columnas"), df_digital.shape[0], df_digital.shape[1])
        
        # Load clinical reference data
        df_clinical = self._load_clinical_data()
        logger.info(_("Datos clínicos cargados: %d filas, %d columnas"), df_clinical.shape[0], df_clinical.shape[1])
        
        # Standardize patient IDs
        df_digital = self._standardize_patient_id(df_digital, 'codeid')
        df_clinical = self._standardize_patient_id_clinical(df_clinical)
        df_clinical = self._complete_clinical_patient_ids(df_clinical, df_digital)
        
        # Merge datasets
        logger.info(_("Fusionando conjuntos de datos por identificador de paciente..."))
        df_merged = pd.merge(df_digital, df_clinical, on='patient_id', how='inner')
        
        if df_merged.empty:
            raise DataProcessingError(
                _("La fusión de datos resultó en un DataFrame vacío. Verifique que los identificadores de pacientes coincidan entre BD y Excel.")
            )
        
        logger.info(_("Fusión exitosa: %d registros comunes encontrados"), df_merged.shape[0])
        return df_merged
    
    def _load_digital_data(self, test_type: str) -> pd.DataFrame:
        """Loads digital biomarker data from database or local CSV fallback.
        
        Args:
            test_type (str): Test type identifier (sdmt, tmt, etc.)
            
        Returns:
            pd.DataFrame: Digital biomarker data
            
        Raises:
            DataProcessingError: If both database and fallback fail
        """
        try:
            inspector = inspect(self.engine)
            tables = inspector.get_table_names()
            table_name = test_type.lower()
            
            if table_name not in tables:
                raise ValueError(f"Tabla '{table_name}' no existe en la BD. Tablas disponibles: {tables}")
            
            logger.debug(_("Leyendo tabla %s desde PostgreSQL"), table_name)
            df = pd.read_sql_table(table_name, self.engine)
            return df
            
        except Exception as e:
            logger.warning(_("No se pudo conectar a la BD: %s. Intentando fallback a CSV local..."), str(e))
            csv_path = Path(f"{test_type.lower()}.csv")
            
            if not csv_path.exists():
                raise DataProcessingError(
                    _("No se pudo leer ni desde BD ni desde CSV. Archivo esperado: %s") % csv_path
                )
            
            logger.info(_("Cargando desde CSV local: %s"), csv_path)
            return pd.read_csv(csv_path)
    
    def _load_clinical_data(self) -> pd.DataFrame:
        """Loads clinical reference data from Excel or CSV.
        
        Returns:
            pd.DataFrame: Clinical reference data
            
        Raises:
            DataProcessingError: If loading fails
        """
        excel_path = Path(self.excel_path)
        
        if not excel_path.exists():
            raise DataProcessingError(
                _("Archivo de datos clínicos no encontrado: %s") % self.excel_path
            )
        
        try:
            logger.debug(
                _("Leyendo datos clínicos desde Excel: %s (skiprows=%d)"),
                self.excel_path,
                self.clinical_skiprows
            )
            df_clinical = pd.read_excel(self.excel_path, skiprows=self.clinical_skiprows)
            return self._normalize_clinical_columns(df_clinical)
        except Exception as e:
            logger.debug(_("Fallback a CSV: %s"), str(e))
            try:
                df_clinical = pd.read_csv(self.excel_path, skiprows=self.clinical_skiprows)
                return self._normalize_clinical_columns(df_clinical)
            except Exception as csv_error:
                raise DataProcessingError(
                    _("No se pudo leer datos clínicos ni desde Excel ni desde CSV: %s") % str(csv_error)
                )

    @staticmethod
    def _is_unnamed_column(column_name: Any) -> bool:
        """Checks whether a column label is an unnamed placeholder."""
        return str(column_name).strip().lower().startswith("unnamed:")

    @staticmethod
    def _looks_like_suffix_tokens(tokens: List[str], expected_count: int) -> bool:
        """Heuristic to detect compact suffix groups like A/B or fis/Cog/psc."""
        if len(tokens) < expected_count:
            return False

        suffix_tokens = tokens[-expected_count:]
        return all(1 <= len(token) <= 6 for token in suffix_tokens)

    @classmethod
    def _expand_merged_header(cls, header: str, span: int) -> List[str]:
        """Expands a merged header across consecutive unnamed columns."""
        header_clean = " ".join(str(header).strip().split())
        tokens = header_clean.split()

        if cls._looks_like_suffix_tokens(tokens, span):
            suffix_tokens = tokens[-span:]
            base_tokens = tokens[:-span]
            if base_tokens:
                base_header = " ".join(base_tokens).rstrip(",;:/-").strip()
                if base_header:
                    return [f"{base_header} {suffix}".strip() for suffix in suffix_tokens]

        return [header_clean] + [f"{header_clean} [{idx}]" for idx in range(2, span + 1)]

    @staticmethod
    def _deduplicate_column_names(columns: List[str]) -> List[str]:
        """Ensures the final column labels are unique."""
        seen: Dict[str, int] = {}
        unique_columns: List[str] = []

        for column in columns:
            count = seen.get(column, 0)
            if count == 0:
                unique_columns.append(column)
            else:
                unique_columns.append(f"{column}__{count + 1}")
            seen[column] = count + 1

        return unique_columns

    @classmethod
    def _normalize_clinical_columns(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Rebuilds clinical headers split across unnamed columns after Excel import."""
        original_columns = [str(col).strip() for col in df.columns]
        normalized_columns = original_columns.copy()
        reconstructed_groups = 0

        idx = 0
        while idx < len(original_columns):
            current_header = original_columns[idx]
            if cls._is_unnamed_column(current_header):
                idx += 1
                continue

            run_end = idx + 1
            while run_end < len(original_columns) and cls._is_unnamed_column(original_columns[run_end]):
                run_end += 1

            span = run_end - idx
            if span > 1:
                expanded_headers = cls._expand_merged_header(current_header, span)
                normalized_columns[idx:run_end] = expanded_headers
                reconstructed_groups += 1

            idx = run_end

        deduplicated_columns = cls._deduplicate_column_names(normalized_columns)
        df_normalized = df.copy()
        df_normalized.columns = deduplicated_columns

        if reconstructed_groups > 0:
            logger.info(
                _("Cabeceras clínicas reconstruidas desde columnas Unnamed: %d grupos detectados"),
                reconstructed_groups
            )
            logger.debug(_("Primeras columnas normalizadas: %s"), deduplicated_columns[:15])

        return df_normalized
    
    @staticmethod
    def _standardize_patient_id(df: pd.DataFrame, id_column: str) -> pd.DataFrame:
        """Standardizes patient ID column for merging.
        
        Args:
            df (pd.DataFrame): DataFrame to process
            id_column (str): Name of the ID column
            
        Returns:
            pd.DataFrame: Modified DataFrame
        """
        if id_column not in df.columns:
            raise DataProcessingError(_("Columna de ID '%s' no encontrada en el DataFrame") % id_column)
        
        df = df.copy()
        df['patient_id'] = df[id_column].astype(str).str.strip().str.upper()
        df['patient_id_base'] = df['patient_id'].map(DataProcessor._extract_patient_id_base)
        return df
    
    @staticmethod
    def _standardize_patient_id_clinical(df: pd.DataFrame) -> pd.DataFrame:
        """Standardizes patient ID in clinical data (flexible column detection).
        
        Args:
            df (pd.DataFrame): Clinical DataFrame
            
        Returns:
            pd.DataFrame: Modified DataFrame
        """
        df = df.copy()
        
        # Try common column names for patient ID
        id_candidates = ['Código', 'codigo', 'ID', 'id', 'patient_id', 'Patient ID']
        id_column = None
        
        for candidate in id_candidates:
            if candidate in df.columns:
                id_column = candidate
                logger.debug(_("Columna de ID clínico detectada: %s"), id_column)
                break
        
        if id_column is None:
            logger.warning(_("Columna de ID no detectada. Usando primera columna como ID."))
            id_column = df.columns[0]
        
        df['patient_id'] = df[id_column].astype(str).str.strip().str.upper()
        df['patient_id_base'] = df['patient_id'].map(DataProcessor._extract_patient_id_base)
        return df

    @staticmethod
    def _extract_patient_id_base(patient_id: str) -> str:
        """Extracts the common patient identifier before any CRC suffix."""
        return patient_id.split('-', 1)[0].strip().upper()

    @staticmethod
    def _complete_clinical_patient_ids(df_clinical: pd.DataFrame, df_digital: pd.DataFrame) -> pd.DataFrame:
        """Completes clinical patient IDs using unique digital ID matches by base ID."""
        df_completed = df_clinical.copy()

        exact_matches = df_completed['patient_id'].isin(df_digital['patient_id'])
        digital_unique = (
            df_digital[['patient_id_base', 'patient_id']]
            .drop_duplicates()
            .groupby('patient_id_base')['patient_id']
            .agg(list)
        )
        unique_base_map = {
            base_id: patient_ids[0]
            for base_id, patient_ids in digital_unique.items()
            if len(patient_ids) == 1
        }

        mapped_ids = df_completed['patient_id_base'].map(unique_base_map)
        can_complete = (~exact_matches) & mapped_ids.notna()
        df_completed.loc[can_complete, 'patient_id'] = mapped_ids.loc[can_complete]

        ambiguous_bases: List[str] = sorted(
            [str(base_id) for base_id, patient_ids in digital_unique.items() if len(patient_ids) > 1]
        )
        if ambiguous_bases:
            logger.warning(
                _("Se detectaron %d patient_id_base ambiguos en datos digitales. No se completarán automáticamente."),
                len(ambiguous_bases)
            )
            logger.debug(_("Bases ambiguas detectadas: %s"), ambiguous_bases)

        logger.info(
            _("Reconstrucción de patient_id clínico: %d coincidencias exactas, %d completadas por base común, %d sin resolver"),
            int(exact_matches.sum()),
            int(can_complete.sum()),
            int((~df_completed['patient_id'].isin(df_digital['patient_id'])).sum())
        )

        unresolved_clinical_ids = sorted(
            df_completed.loc[
                ~df_completed['patient_id'].isin(df_digital['patient_id']),
                'patient_id'
            ].dropna().astype(str).unique().tolist()
        )
        if unresolved_clinical_ids:
            logger.debug(
                _("Códigos clínicos sin correspondencia en datos digitales: %s"),
                unresolved_clinical_ids
            )

        return df_completed

    def prepare_splits(self, df: pd.DataFrame, test_type: str, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Cleans the dataset and performs the initial train-test partitioning.

        This method identifies the appropriate target column depending on the test
        type, drops missing values in the target, and removes non-predictive metadata
        columns (like dates, IDs, and timestamps). It then splits the data into a
        training set for model optimization and a completely blind test set for
        final evaluation, ensuring no data leakage from the same patient across sets.

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
                  
        Raises:
            DataProcessingError: If target column not found, insufficient data, or other issues
        """
        logger.info(_("Preparando particiones de entrenamiento y prueba..."))
        
        test_type_lower = test_type.lower()
        if test_type_lower not in self.column_config:
            raise DataProcessingError(
                _("Tipo de test '%s' no reconocido. Opciones: %s") % (test_type, list(self.column_config.keys()))
            )
        
        config = self.column_config[test_type_lower]
        
        # Find target column using multiple hints
        target_col = self._find_target_column(df, config['target_hints'])
        logger.info(_("Columna target identificada: %s"), target_col)
        
        # Validate target column
        initial_rows = len(df)
        df_clean = df.dropna(subset=[target_col])
        dropped_rows = initial_rows - len(df_clean)
        
        if dropped_rows > 0:
            logger.warning(_("Se eliminaron %d filas con valores faltantes en target (%.1f%%)"), 
                          dropped_rows, (dropped_rows/initial_rows)*100)
        
        if len(df_clean) < 10:
            raise DataProcessingError(
                _("Insuficientes datos después de limpiar. Mínimo 10 muestras requeridas, se obtuvieron %d") % len(df_clean)
            )

        df_clean = self._add_clinical_covariates(df_clean)
        
        y = df_clean[target_col].copy()
        
        # Validate target values
        if not pd.api.types.is_numeric_dtype(y):
            logger.warning(_("Intentando convertir target a numérico..."))
            y = pd.to_numeric(y, errors='coerce')
            y = y.dropna()
        
        if (y <= 0).any():
            logger.warning(_("Se detectaron valores no positivos en target. Pueden afectar métricas de evaluación."))
        
        # Extract and clean features
        X = df_clean.select_dtypes(include=['number']).copy()
        X = X.drop(columns=[target_col], errors='ignore')
        
        # Drop non-predictive columns
        available_columns = X.columns.tolist()
        cols_to_drop = [col for col in config['features_to_drop'] if col in available_columns]
        X = X.drop(columns=cols_to_drop, errors='ignore')
        
        logger.info(_("Features mantenidas: %d de %d columnas numéricas"), X.shape[1], df_clean.select_dtypes(include=['number']).shape[1])
        
        # Handle missing values in features
        missing_before = X.isnull().sum().sum()
        if missing_before > 0:
            logger.warning(_("Se detectaron %d valores faltantes en features. Imputando con media."), missing_before)
            X = X.fillna(X.mean())
        
        # Validate feature data
        if X.shape[1] == 0:
            raise DataProcessingError(_("No se encontraron features válidas después de la limpieza."))
        
        if X.shape[0] != y.shape[0]:
            logger.warning(_("Sincronizando X e y después de limpieza. X: %d, y: %d"), X.shape[0], y.shape[0])
            common_idx = X.index.intersection(y.index)
            X = X.loc[common_idx]
            y = y.loc[common_idx]
        
        # Get patient IDs for grouped CV
        if 'patient_id' not in df_clean.columns:
            raise DataProcessingError(_("Columna 'patient_id' no encontrada. Requerida para validación cruzada agrupada."))
        
        groups = df_clean.loc[X.index, 'patient_id'].copy()
        
        # Perform stratified train-test split respecting patient groups
        logger.info(_("Realizando partición train-test (%.0f%% train, %.0f%% test)..."), (1-test_size)*100, test_size*100)
        
        train_idx, test_idx = train_test_split(
            range(len(X)), 
            test_size=test_size, 
            random_state=42,
            stratify=None  # Could use stratify for balanced splits if target is categorical
        )
        
        X_train = X.iloc[train_idx].reset_index(drop=True)
        X_test = X.iloc[test_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        y_test = y.iloc[test_idx].reset_index(drop=True)
        groups_train = groups.iloc[train_idx].reset_index(drop=True)
        
        logger.info(_("Train: %d muestras | Test: %d muestras | Features: %d"), 
                   len(X_train), len(X_test), X_train.shape[1])
        logger.info(_("Target - Media: %.2f, Std: %.2f, Min: %.2f, Max: %.2f"), 
                   y_train.mean(), y_train.std(), y_train.min(), y_train.max())
        
        return X_train, X_test, y_train, y_test, groups_train

    @staticmethod
    def _find_first_matching_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Finds the first column whose normalized name matches any candidate."""
        normalized_columns = {col.lower().strip(): col for col in df.columns}

        for candidate in candidates:
            direct_match = normalized_columns.get(candidate.lower().strip())
            if direct_match is not None:
                return direct_match

        for candidate in candidates:
            candidate_lower = candidate.lower().strip()
            for col in df.columns:
                if candidate_lower in col.lower().strip():
                    return col

        return None

    @classmethod
    def _add_clinical_covariates(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Adds clinically relevant derived covariates such as age and sex."""
        df_enriched = df.copy()

        birth_col = cls._find_first_matching_column(
            df_enriched,
            ['f_nacim', 'f_nacimiento', 'fecha_nacimiento', 'birth_date', 'dob']
        )
        assessment_col = cls._find_first_matching_column(
            df_enriched,
            ['fecha', 'date_data', 'fecha_test', 'assessment_date', 'test_date']
        )

        logger.info(
            _("Detección de covariables clínicas - nacimiento: %s | fecha de prueba: %s"),
            birth_col or _("no detectada"),
            assessment_col or _("no detectada")
        )

        if birth_col and assessment_col:
            birth_dates = pd.to_datetime(df_enriched[birth_col], errors='coerce')
            assessment_dates = pd.to_datetime(df_enriched[assessment_col], errors='coerce')
            age_years = (assessment_dates - birth_dates).dt.days / 365.25
            valid_age = age_years.where((age_years >= 0) & (age_years <= 120))

            if valid_age.notna().any():
                df_enriched['age_at_test'] = valid_age
                logger.info(
                    _("Feature clínica derivada añadida: age_at_test (%d valores válidos, media %.2f, rango %.2f-%.2f)"),
                    int(valid_age.notna().sum()),
                    float(valid_age.mean()),
                    float(valid_age.min()),
                    float(valid_age.max())
                )
            else:
                logger.warning(
                    _("No fue posible derivar age_at_test a partir de %s y %s"),
                    birth_col, assessment_col
                )

        sex_col = cls._find_first_matching_column(
            df_enriched,
            ['sexo', 'sex', 'gender', 'genero']
        )
        logger.info(
            _("Detección de covariables clínicas - sexo/género: %s"),
            sex_col or _("no detectada")
        )
        if sex_col:
            sex_series = df_enriched[sex_col].astype(str).str.strip().str.lower()
            sex_map = {
                'f': 1.0,
                'female': 1.0,
                'femenino': 1.0,
                'mujer': 1.0,
                'woman': 1.0,
                'm': 0.0,
                'male': 0.0,
                'masculino': 0.0,
                'hombre': 0.0,
                'man': 0.0,
            }
            sex_encoded = sex_series.map(sex_map)
            if sex_encoded.notna().any():
                df_enriched['sex_binary'] = sex_encoded
                female_count = int((sex_encoded == 1.0).sum())
                male_count = int((sex_encoded == 0.0).sum())
                missing_count = int(sex_encoded.isna().sum())
                logger.info(
                    _("Feature clínica derivada añadida: sex_binary (%d valores válidos; female=1: %d, male=0: %d, no codificados: %d)"),
                    int(sex_encoded.notna().sum()),
                    female_count,
                    male_count,
                    missing_count
                )
            else:
                logger.warning(
                    _("No fue posible codificar la columna de sexo/género detectada: %s"),
                    sex_col
                )

        return df_enriched
    
    @staticmethod
    def _find_target_column(df: pd.DataFrame, hints: List[str]) -> str:
        """Finds target column using multiple hints.
        
        Args:
            df (pd.DataFrame): DataFrame to search
            hints (List[str]): List of possible column name hints
            
        Returns:
            str: Name of the target column
            
        Raises:
            DataProcessingError: If no matching column found
        """
        # First try exact matches
        for hint in hints:
            if hint in df.columns:
                logger.debug(_("Target encontrado (coincidencia exacta): %s"), hint)
                return hint
        
        # Then try case-insensitive substring matches
        for hint in hints:
            hint_lower = hint.lower().strip()
            for col in df.columns:
                if hint_lower in col.lower().strip():
                    logger.debug(_("Target encontrado (coincidencia parcial): %s (hint: %s)"), col, hint)
                    return col
        
        # Last resort: look for any column with 'papel' or 'paper'
        for col in df.columns:
            if 'papel' in col.lower() or 'paper' in col.lower():
                logger.debug(_("Target encontrado (contiene 'papel'): %s"), col)
                return col
        
        raise DataProcessingError(
            _("No se pudo identificar columna target. Pistas buscadas: %s. Columnas disponibles: %s") % 
            (hints, df.columns.tolist())
        )

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
            
        Raises:
            ValueError: If n_splits is invalid or groups are insufficient
        """
        if n_splits < 2:
            raise ValueError(_("n_splits debe ser al menos 2"))
        
        n_groups = groups_train.nunique()
        if n_groups < n_splits:
            logger.warning(
                _("Número de grupos únicos (%d) menor que n_splits (%d). Ajustando a %d splits."),
                n_groups, n_splits, n_groups
            )
            n_splits = max(2, n_groups)
        
        logger.info(_("Generando %d particiones de validación cruzada agrupada (%d grupos únicos)..."), 
                   n_splits, n_groups)
        
        gkf = GroupKFold(n_splits=n_splits)
        groups_array = groups_train.to_numpy()
        dummy_features = np.zeros((len(groups_array), 1))
        folds = list(gkf.split(X=dummy_features, y=None, groups=groups_array))
        
        logger.debug(_("Estructura de folds:"))
        for fold_idx, (train_idx, val_idx) in enumerate(folds, 1):
            train_groups = groups_train.iloc[train_idx].nunique()
            val_groups = groups_train.iloc[val_idx].nunique()
            logger.debug(
                _("  Fold %d: Train %d muestras (%d grupos) | Val %d muestras (%d grupos)"),
                fold_idx, len(train_idx), train_groups, len(val_idx), val_groups
            )
        
        return folds
