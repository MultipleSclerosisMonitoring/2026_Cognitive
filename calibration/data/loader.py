import numpy as np
import pandas as pd
from pandas import CategoricalDtype
from sqlalchemy import create_engine, inspect
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
import logging
from typing import Tuple, Any, Dict, List, Optional
from pathlib import Path
import re
import unicodedata
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
        feature_filter_config: Optional[Dict[str, Any]] = None,
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
        self.feature_filter_config = feature_filter_config or self._default_feature_filter_config()
        self.digital_source_columns: set[str] = set()
        self.last_feature_audit: Dict[str, Any] = {}
    

    @staticmethod
    def _default_feature_filter_config() -> Dict[str, Any]:
        """Returns default feature filtering settings for clinical robustness."""
        return {
            'allowed_derived_features': ['age_at_test', 'sex_binary', 'delta_dias_digital_papel', 'disease_duration_years'],
            'allowed_categorical_features': ['clinical_group'],
            'max_train_missing_rate': 0.35,
            'min_train_unique_values': 2,
            'min_train_std': 1e-8,
            'tmt_use_advanced_kinematics': True,
            'tmt_transform_strategy': 'log1p',
            'tmt_feature_selection_enabled': True,
            'tmt_include_demographic_covariates': False,
            'tmt_keep_education_band': True,
            'tmt_feature_selection_max_features': 24,
            'tmt_feature_selection_min_features': 8,
            'tmt_feature_selection_step_ratio': 0.2,
            'sdmt_fatigue_features_enabled': True,
        }

    @staticmethod
    def _match_any_pattern(column_name: str, patterns: List[str]) -> bool:
        """Checks whether a normalized feature name matches any regex pattern."""
        normalized = DataProcessor._normalize_lookup_text(column_name)
        return any(re.search(pattern, normalized) for pattern in patterns)

    @staticmethod
    def _tmt_priority_patterns() -> Dict[str, List[str]]:
        """Defines heuristic patterns for clinically relevant TMT kinematics."""
        return {
            'inside_circle_time': [
                r'(inside|dwell|stay|within|in)\s*(circle|circulo)',
                r'(circle|circulo).*(inside|dwell|stay|within|in)',
                r'average\s*time\s*inside\s*circles',
            ],
            'flight_time': [
                r'(flight|vuelo|hover|air)\s*(time|tiempo)',
                r'(transition|inter|between).*(time|tiempo)',
                r'average\s*time\s*between\s*circles',
            ],
            'inside_circle_rate': [
                r'average\s*rate\s*inside\s*circles',
            ],
            'between_circle_rate': [
                r'average\s*rate\s*between\s*circles',
            ],
            'pressure_mean': [
                r'(press|pressure|presion).*(mean|avg|media)',
                r'(mean|avg|media).*(press|pressure|presion)',
                r'average\s*total\s*pressure',
            ],
            'preselect_letter_time': [
                r'average\s*time\s*before\s*letters',
            ],
            'preselect_number_time': [
                r'average\s*time\s*before\s*numbers',
            ],
            'preselect_letter_rate': [
                r'average\s*rate\s*before\s*letters',
            ],
            'preselect_number_rate': [
                r'average\s*rate\s*before\s*numbers',
            ],
            'finger_lift_rate': [
                r'(finger|dedo|touch|pen).*(lift|up|levant)',
                r'(lift|up|levant).*(rate|ratio|freq|frecuencia|count|conteo)',
                r'(air|hover).*(rate|ratio|freq|frecuencia|count|conteo)',
                r'average\s*lift',
                r'number\s*lifts',
            ],
            'error_count': [
                r'number\s*errors',
            ],
        }

    @staticmethod
    def _tmt_global_time_patterns() -> List[str]:
        """Patterns used to demote overly global TMT summaries."""
        return [
            r'(total|overall|global|completion|complete|duracion|duration).*(time|tiempo)',
            r'(time|tiempo).*(total|overall|global|completion|complete)',
        ]

    @staticmethod
    def _tmt_transform_patterns() -> List[str]:
        """Patterns for temporal and count features that benefit from log transforms."""
        return [
            r'(time|tiempo|latency|latencia|duration|duracion|delay|dwell|flight|hover|air)',
            r'(error|errores|mistake|count|conteo|n_?)',
        ]

    def _derive_tmt_kinematic_features(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        """Builds ratio-style TMT features from the most relevant kinematic signals."""
        X_enriched = X.copy()
        patterns = self._tmt_priority_patterns()
        matched_columns: Dict[str, List[str]] = {}

        for feature_family, regexes in patterns.items():
            matched_columns[feature_family] = [
                column for column in X_enriched.columns
                if self._match_any_pattern(column, regexes)
            ]

        inside_circle_cols = matched_columns.get('inside_circle_time', [])
        flight_time_cols = matched_columns.get('flight_time', [])
        if inside_circle_cols and flight_time_cols:
            inside_col = inside_circle_cols[0]
            flight_col = flight_time_cols[0]
            denominator = X_enriched[flight_col].replace(0.0, np.nan)
            X_enriched['tmt_kinematic_circle_to_flight_ratio'] = X_enriched[inside_col] / denominator
            X_enriched['tmt_kinematic_circle_minus_flight'] = X_enriched[inside_col] - X_enriched[flight_col]

        inside_rate_cols = matched_columns.get('inside_circle_rate', [])
        between_rate_cols = matched_columns.get('between_circle_rate', [])
        if inside_rate_cols and between_rate_cols:
            inside_rate_col = inside_rate_cols[0]
            between_rate_col = between_rate_cols[0]
            denominator = X_enriched[between_rate_col].replace(0.0, np.nan)
            X_enriched['tmt_kinematic_inside_to_between_rate_ratio'] = X_enriched[inside_rate_col] / denominator
            X_enriched['tmt_kinematic_inside_minus_between_rate'] = X_enriched[inside_rate_col] - X_enriched[between_rate_col]

        letter_time_cols = matched_columns.get('preselect_letter_time', [])
        number_time_cols = matched_columns.get('preselect_number_time', [])
        if letter_time_cols and number_time_cols:
            letter_time_col = letter_time_cols[0]
            number_time_col = number_time_cols[0]
            denominator = X_enriched[number_time_col].replace(0.0, np.nan)
            X_enriched['tmt_kinematic_letter_to_number_pretime_ratio'] = X_enriched[letter_time_col] / denominator
            X_enriched['tmt_kinematic_letter_minus_number_pretime'] = X_enriched[letter_time_col] - X_enriched[number_time_col]

        letter_rate_cols = matched_columns.get('preselect_letter_rate', [])
        number_rate_cols = matched_columns.get('preselect_number_rate', [])
        if letter_rate_cols and number_rate_cols:
            letter_rate_col = letter_rate_cols[0]
            number_rate_col = number_rate_cols[0]
            denominator = X_enriched[number_rate_col].replace(0.0, np.nan)
            X_enriched['tmt_kinematic_letter_to_number_prerate_ratio'] = X_enriched[letter_rate_col] / denominator
            X_enriched['tmt_kinematic_letter_minus_number_prerate'] = X_enriched[letter_rate_col] - X_enriched[number_rate_col]

        pressure_cols = matched_columns.get('pressure_mean', [])
        lift_cols = matched_columns.get('finger_lift_rate', [])
        if pressure_cols and lift_cols:
            pressure_col = pressure_cols[0]
            lift_col = lift_cols[0]
            denominator = X_enriched[lift_col].replace(0.0, np.nan)
            X_enriched['tmt_kinematic_pressure_per_lift'] = X_enriched[pressure_col] / denominator
            X_enriched['tmt_kinematic_pressure_lift_interaction'] = X_enriched[pressure_col] * X_enriched[lift_col]

        if lift_cols and flight_time_cols:
            lift_col = lift_cols[0]
            flight_col = flight_time_cols[0]
            denominator = X_enriched[flight_col].replace(0.0, np.nan)
            X_enriched['tmt_kinematic_lift_per_flight'] = X_enriched[lift_col] / denominator

        error_cols = matched_columns.get('error_count', [])
        if error_cols and flight_time_cols:
            error_col = error_cols[0]
            flight_col = flight_time_cols[0]
            denominator = X_enriched[flight_col].replace(0.0, np.nan)
            X_enriched['tmt_kinematic_errors_per_between_time'] = X_enriched[error_col] / denominator

        created_columns = [
            column for column in X_enriched.columns
            if column.startswith('tmt_kinematic_')
        ]
        matched_columns['derived_features'] = created_columns
        return X_enriched, matched_columns

    def _derive_sdmt_fatigue_features(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Builds within-test fatigability markers from SDMT tercile counters."""
        X_enriched = X.copy()
        created_columns: List[str] = []

        digit_cols = [column for column in ['numdig1', 'numdig2', 'numdig3'] if column in X_enriched.columns]
        error_cols = [column for column in ['numerr1', 'numerr2', 'numerr3'] if column in X_enriched.columns]

        if len(digit_cols) == 3:
            first, second, third = (X_enriched[column] for column in digit_cols)
            first_safe = first.replace(0.0, np.nan)
            total = first + second + third
            X_enriched['sdmt_fatigue_correct_total_terciles'] = total
            X_enriched['sdmt_fatigue_correct_last_minus_first'] = third - first
            X_enriched['sdmt_fatigue_correct_last_over_first'] = third / first_safe
            X_enriched['sdmt_fatigue_correct_slope'] = (third - first) / 2.0
            X_enriched['sdmt_fatigue_correct_mid_drop'] = second - first
            X_enriched['sdmt_fatigue_correct_end_drop'] = third - second
            created_columns.extend([
                'sdmt_fatigue_correct_total_terciles',
                'sdmt_fatigue_correct_last_minus_first',
                'sdmt_fatigue_correct_last_over_first',
                'sdmt_fatigue_correct_slope',
                'sdmt_fatigue_correct_mid_drop',
                'sdmt_fatigue_correct_end_drop',
            ])

        if len(error_cols) == 3:
            first_err, second_err, third_err = (X_enriched[column] for column in error_cols)
            first_err_safe = first_err.replace(0.0, np.nan)
            total_err = first_err + second_err + third_err
            X_enriched['sdmt_fatigue_error_total_terciles'] = total_err
            X_enriched['sdmt_fatigue_error_last_minus_first'] = third_err - first_err
            X_enriched['sdmt_fatigue_error_last_over_first'] = third_err / first_err_safe
            X_enriched['sdmt_fatigue_error_slope'] = (third_err - first_err) / 2.0
            created_columns.extend([
                'sdmt_fatigue_error_total_terciles',
                'sdmt_fatigue_error_last_minus_first',
                'sdmt_fatigue_error_last_over_first',
                'sdmt_fatigue_error_slope',
            ])

        if len(digit_cols) == 3 and len(error_cols) == 3:
            first_eff_den = (X_enriched[digit_cols[0]] + X_enriched[error_cols[0]]).replace(0.0, np.nan)
            third_eff_den = (X_enriched[digit_cols[2]] + X_enriched[error_cols[2]]).replace(0.0, np.nan)
            first_eff = X_enriched[digit_cols[0]] / first_eff_den
            third_eff = X_enriched[digit_cols[2]] / third_eff_den
            X_enriched['sdmt_fatigue_accuracy_last_minus_first'] = third_eff - first_eff
            X_enriched['sdmt_fatigue_accuracy_last_over_first'] = third_eff / first_eff.replace(0.0, np.nan)
            created_columns.extend([
                'sdmt_fatigue_accuracy_last_minus_first',
                'sdmt_fatigue_accuracy_last_over_first',
            ])

        return X_enriched, created_columns

    def _apply_train_fitted_log_transforms(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        candidate_columns: List[str],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """Applies train-fitted log1p transforms to skewed non-negative predictors."""
        transformed_columns: List[str] = []
        for column in candidate_columns:
            if column not in X_train.columns or column not in X_test.columns:
                continue
            train_values = X_train[column]
            test_values = X_test[column]
            if train_values.isna().all():
                continue
            min_train = float(train_values.min())
            if min_train < 0.0:
                continue
            X_train[column] = np.log1p(train_values)
            X_test[column] = np.log1p(test_values.clip(lower=0.0))
            transformed_columns.append(column)

        return X_train, X_test, transformed_columns

    @staticmethod
    def _build_tmt_target_proxy(y_train: pd.DataFrame) -> pd.Series:
        """Aggregates the available TMT targets into a stable proxy for feature selection."""
        standardized_targets = []
        for column in y_train.columns:
            series = y_train[column].astype(float)
            mean = series.mean()
            std = series.std(ddof=0)
            if pd.isna(std) or std <= 1e-8:
                standardized_targets.append(series - mean)
            else:
                standardized_targets.append((series - mean) / std)

        proxy = pd.concat(standardized_targets, axis=1).mean(axis=1, skipna=True)
        return proxy

    def _select_tmt_features_via_rf_rfe(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """Runs a lightweight recursive Random Forest elimination for TMT."""
        if X_train.shape[1] <= 3:
            return X_train, X_test, X_train.columns.astype(str).tolist()

        selection_target = self._build_tmt_target_proxy(y_train)
        valid_rows = selection_target.notna()
        if valid_rows.sum() < 10:
            return X_train, X_test, X_train.columns.astype(str).tolist()

        max_features = int(self.feature_filter_config.get('tmt_feature_selection_max_features', 24))
        min_features = int(self.feature_filter_config.get('tmt_feature_selection_min_features', 8))
        step_ratio = float(self.feature_filter_config.get('tmt_feature_selection_step_ratio', 0.2))
        current_columns = X_train.columns.astype(str).tolist()

        max_features = max(min_features, min(max_features, len(current_columns)))
        while len(current_columns) > max_features:
            estimator = RandomForestRegressor(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            )
            estimator.fit(
                X_train.loc[valid_rows, current_columns],
                selection_target.loc[valid_rows],
            )
            importances = pd.Series(estimator.feature_importances_, index=current_columns).sort_values()
            step = max(1, int(len(current_columns) * step_ratio))
            removable = max(0, len(current_columns) - min_features)
            drop_count = min(step, removable)
            if drop_count <= 0:
                break
            to_drop = importances.head(drop_count).index.tolist()
            current_columns = [column for column in current_columns if column not in to_drop]

        return (
            X_train.loc[:, current_columns].copy(),
            X_test.loc[:, current_columns].copy(),
            current_columns,
        )

    @staticmethod
    def _default_column_config() -> Dict:
        """Returns default column configuration."""
        return {
            'sdmt': {
                'target_columns': [
                    {
                        'name': 'sdmt_paper_score',
                        'hints': ['Día 1 SDMT Papel score', 'SDMT Papel', 'sdmt_papel']
                    }
                ],
                'features_to_drop': ['id', 'codeid', 'ts_created', 'ts_updated', 'patient_id', 'sset']
            },
            'tmt': {
                'target_columns': [
                    {
                        'name': 'tmt_paper_score_a',
                        'group': 'score',
                        'hints': ['DIA 1 TMT papel (tiempo) A', 'TMT papel (tiempo) A', 'tmt papel tiempo a']
                    },
                    {
                        'name': 'tmt_paper_score_b',
                        'group': 'score',
                        'hints': ['DIA 1 TMT papel (tiempo) B', 'TMT papel (tiempo) B', 'tmt papel tiempo b']
                    },
                    {
                        'name': 'tmt_paper_errors_a',
                        'group': 'errors',
                        'hints': ['DIA 1 TMT papel (errores) A', 'TMT papel (errores) A', 'tmt papel errores a']
                    },
                    {
                        'name': 'tmt_paper_errors_b',
                        'group': 'errors',
                        'hints': ['DIA 1 TMT papel (errores) B', 'TMT papel (errores) B', 'tmt papel errores b']
                    }
                ],
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
        self.digital_source_columns = {str(column) for column in df_digital.columns}
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
            clinical_frames: List[pd.DataFrame] = []

            raw_workbook = pd.read_excel(self.excel_path, header=None, sheet_name=None)
            for sheet_name, raw_sheet in raw_workbook.items():
                header_candidates = raw_sheet.iloc[: self.clinical_skiprows + 3]
                header_rows = header_candidates.apply(
                    lambda row: row.astype('string').str.strip().eq('Código').any(),
                    axis=1,
                )
                header_matches = header_rows[header_rows].index.tolist()
                header_row = header_matches[0] if header_matches else self.clinical_skiprows
                df_sheet = pd.read_excel(
                    self.excel_path,
                    sheet_name=sheet_name,
                    header=header_row,
                )
                df_normalized = self._normalize_clinical_columns(df_sheet)
                df_normalized = self._normalize_clinical_dtypes(df_normalized)
                df_normalized = df_normalized.dropna(how='all')
                df_normalized["clinical_source_sheet"] = str(sheet_name)
                clinical_frames.append(df_normalized)

            if not clinical_frames:
                raise DataProcessingError(_("El libro clínico no contiene hojas legibles"))

            df_clinical = pd.concat(clinical_frames, ignore_index=True, sort=False)
            logger.info(
                _("Datos clínicos combinados desde %d hoja(s): %s"),
                len(clinical_frames),
                list(raw_workbook.keys())
            )
            return df_clinical
        except Exception as e:
            logger.debug(_("Fallback a CSV: %s"), str(e))
            try:
                df_clinical = pd.read_csv(self.excel_path, skiprows=self.clinical_skiprows)
                df_clinical = self._normalize_clinical_columns(df_clinical)
                return self._normalize_clinical_dtypes(df_clinical)
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
    def _is_date_like_column(column_name: str) -> bool:
        """Checks whether a column name likely contains date values."""
        column_lower = column_name.lower()
        date_hints = ["fecha", "date", "nacim", "birth", "dob"]
        return any(hint in column_lower for hint in date_hints)

    @staticmethod
    def _is_identifier_like_column(column_name: str) -> bool:
        """Checks whether a column is likely an identifier and should not be coerced."""
        column_lower = column_name.lower()
        id_hints = ["id", "codigo", "código", "code", "patient"]
        return any(hint in column_lower for hint in id_hints)

    @staticmethod
    def _normalize_lookup_text(value: Any) -> str:
        """Normalizes strings for resilient clinical column lookup."""
        text = str(value).strip().lower()
        text = unicodedata.normalize("NFKD", text)
        text = "".join(char for char in text if not unicodedata.combining(char))
        return " ".join(text.replace("_", " ").split())

    @staticmethod
    def _is_categorical_series(series: pd.Series) -> bool:
        """Checks whether a pandas Series uses categorical dtype."""
        return isinstance(series.dtype, CategoricalDtype)

    @staticmethod
    def _coerce_numeric_series(series: pd.Series) -> pd.Series:
        """Converts locale-dependent numeric strings into pandas numeric values."""
        normalized = series.astype("string").str.strip()
        has_comma = normalized.str.contains(",", na=False)
        has_dot = normalized.str.contains(".", regex=False, na=False)

        both_separators = has_comma & has_dot
        normalized = normalized.where(~both_separators, normalized.str.replace(".", "", regex=False))
        normalized = normalized.where(~has_comma, normalized.str.replace(",", ".", regex=False))
        return pd.to_numeric(normalized, errors="coerce")

    @classmethod
    def _normalize_clinical_dtypes(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Normalizes common Excel clinical types such as dates and numeric strings."""
        df_normalized = df.copy()
        converted_dates: List[str] = []
        converted_numeric: List[str] = []
        cleaned_strings: List[str] = []

        for column in df_normalized.columns:
            series = df_normalized[column]

            if pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_numeric_dtype(series):
                continue

            if not (
                pd.api.types.is_object_dtype(series)
                or pd.api.types.is_string_dtype(series)
                or cls._is_categorical_series(series)
            ):
                continue

            cleaned = series.astype("string").str.strip()
            missing_tokens = {"", "nan", "None", "none", "N/A", "n/a"}
            cleaned = cleaned.mask(cleaned.isin(missing_tokens), pd.NA)
            df_normalized[column] = cleaned
            cleaned_strings.append(str(column))

            non_null_count = int(cleaned.notna().sum())
            if non_null_count == 0:
                continue

            if cls._is_date_like_column(str(column)):
                parsed_dates = pd.to_datetime(cleaned, errors="coerce", dayfirst=True)
                parsed_count = int(parsed_dates.notna().sum())
                if parsed_count >= max(3, int(non_null_count * 0.5)):
                    df_normalized[column] = parsed_dates
                    converted_dates.append(str(column))
                    continue

            if cls._is_identifier_like_column(str(column)):
                continue

            numeric_candidate = cls._coerce_numeric_series(cleaned)
            numeric_count = int(numeric_candidate.notna().sum())
            if numeric_count >= max(3, int(non_null_count * 0.8)):
                df_normalized[column] = numeric_candidate
                converted_numeric.append(str(column))

        if converted_dates:
            logger.info(_("Columnas clínicas convertidas a fecha: %s"), converted_dates)
        if converted_numeric:
            logger.info(_("Columnas clínicas convertidas a numérico: %s"), converted_numeric[:20])
            if len(converted_numeric) > 20:
                logger.debug(_("Conversión numérica adicional (%d columnas más)"), len(converted_numeric) - 20)
        logger.debug(_("Columnas clínicas limpiadas como string: %d"), len(cleaned_strings))

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

    def prepare_splits(
        self,
        df: pd.DataFrame,
        test_type: str,
        test_size: float = 0.2
    ) -> Tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.Series,
        pd.Series,
        pd.DataFrame,
        pd.DataFrame,
        Dict[str, List[str]],
    ]:
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
            A tuple containing the training and hold-out feature matrices, target
            frames, patient-group series, metadata frames, and target-group mapping.
                  
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
        
        target_mapping = self._resolve_target_columns(df, config)
        target_columns = list(target_mapping.keys())
        source_target_columns = list(target_mapping.values())
        target_groups = self._build_target_groups(config, target_mapping)
        logger.info(_("Columnas target identificadas: %s"), target_mapping)
        
        # Validate target coverage
        initial_rows = len(df)
        df_clean = df.dropna(subset=source_target_columns, how='all')
        dropped_rows = initial_rows - len(df_clean)
        
        if dropped_rows > 0:
            logger.warning(_("Se eliminaron %d filas sin ningún target disponible (%.1f%%)"), 
                          dropped_rows, (dropped_rows/initial_rows)*100)
        
        if len(df_clean) < 10:
            raise DataProcessingError(
                _("Insuficientes datos después de limpiar. Mínimo 10 muestras requeridas, se obtuvieron %d") % len(df_clean)
            )

        df_clean = self._add_clinical_covariates(df_clean)
        df_clean = self._encode_low_cardinality_categoricals(
            df_clean,
            exclude_columns=source_target_columns + ['patient_id', 'patient_id_base'],
            include_columns=self.feature_filter_config.get('allowed_categorical_features', ['clinical_group'])
        )
        
        y = df_clean[source_target_columns].copy()
        y.columns = target_columns
        
        # Validate target values
        for column in y.columns:
            if not pd.api.types.is_numeric_dtype(y[column]):
                logger.warning(_("Intentando convertir target '%s' a numérico..."), column)
                y[column] = pd.to_numeric(y[column], errors='coerce')
        
        if (y <= 0).any().any():
            logger.warning(_("Se detectaron valores no positivos en target. Pueden afectar métricas de evaluación."))
        
        # Extract and clean features prioritizing digital biomarkers plus stable covariates
        numeric_df = df_clean.select_dtypes(include=['number']).copy()
        numeric_df.columns = numeric_df.columns.map(str)
        numeric_df = numeric_df.astype(float)

        allowed_digital_features = {str(column) for column in self.digital_source_columns}
        allowed_derived_features = set(self.feature_filter_config.get('allowed_derived_features', []))
        allowed_categorical_features = self.feature_filter_config.get('allowed_categorical_features', ['clinical_group'])
        allowed_dummy_prefixes = tuple(f"{column}_" for column in allowed_categorical_features)

        selected_feature_names = [
            column for column in numeric_df.columns
            if (
                column in allowed_digital_features
                or column in allowed_derived_features
                or column.startswith(allowed_dummy_prefixes)
            )
        ]

        X = numeric_df.loc[:, selected_feature_names].copy()
        X = X.drop(columns=source_target_columns, errors='ignore')

        if test_type_lower == 'tmt' and not self.feature_filter_config.get('tmt_include_demographic_covariates', False):
            tmt_covariates_to_drop = []
            for column in X.columns:
                if column in allowed_derived_features or column.startswith(allowed_dummy_prefixes):
                    if self.feature_filter_config.get('tmt_keep_education_band', True) and (column == 'education_band' or column.startswith('education_band_')):
                        continue
                    tmt_covariates_to_drop.append(column)
            X = X.drop(columns=tmt_covariates_to_drop, errors='ignore')

        tmt_priority_matches: Dict[str, List[str]] = {}
        sdmt_fatigue_features: List[str] = []
        if test_type_lower == 'tmt':
            X, tmt_priority_matches = self._derive_tmt_kinematic_features(X)
        elif test_type_lower == 'sdmt' and self.feature_filter_config.get('sdmt_fatigue_features_enabled', True):
            X, sdmt_fatigue_features = self._derive_sdmt_fatigue_features(X)

        available_columns = X.columns.tolist()
        cols_to_drop = [col for col in config['features_to_drop'] if col in available_columns]
        X = X.drop(columns=cols_to_drop, errors='ignore')

        removed_global_tmt_features: List[str] = []
        if test_type_lower == 'tmt' and self.feature_filter_config.get('tmt_use_advanced_kinematics', True):
            priority_columns = sorted({
                column
                for columns in tmt_priority_matches.values()
                for column in columns
                if column in X.columns
            })
            global_time_columns = [
                column for column in X.columns
                if self._match_any_pattern(column, self._tmt_global_time_patterns())
            ]
            covariate_columns = [
                column for column in X.columns
                if column in allowed_derived_features or column.startswith(allowed_dummy_prefixes)
            ]
            if priority_columns:
                keep_columns = sorted(set(priority_columns + covariate_columns))
                removed_global_tmt_features = [
                    column for column in X.columns
                    if column not in keep_columns and column in global_time_columns
                ]
                X = X.loc[:, keep_columns].copy()
            elif global_time_columns:
                removed_global_tmt_features = global_time_columns
                X = X.drop(columns=global_time_columns, errors='ignore')

        excluded_non_digital = [
            column for column in numeric_df.columns
            if column not in selected_feature_names and column not in source_target_columns
        ]
        self.last_feature_audit = {
            'test_type': test_type_lower,
            'selected_features_initial': X.columns.astype(str).tolist(),
            'removed_by_reason': {
                'non_digital_or_non_stable_clinical': excluded_non_digital,
                'config_features_to_drop': cols_to_drop,
                'tmt_global_time_removed': removed_global_tmt_features,
            },
            'tmt_priority_matches': tmt_priority_matches,
            'sdmt_fatigue_features': sdmt_fatigue_features,
        }

        logger.info(
            _("Features mantenidas: %d de %d columnas numéricas tras filtrar biomarcadores digitales y covariables clínicas estables"),
            X.shape[1],
            numeric_df.shape[1]
        )
        
        # Validate feature data
        if X.shape[1] == 0:
            raise DataProcessingError(_("No se encontraron features válidas después de la limpieza."))
        
        if X.shape[0] != y.shape[0]:
            logger.warning(_("Sincronizando X e y después de limpieza. X: %d, y: %d"), X.shape[0], y.shape[0])
            common_idx = X.index.intersection(y.index)
            X = X.loc[common_idx]
            y = y.loc[common_idx]
        
        metadata_columns = [
            column for column in [
                'patient_id',
                'clinical_group',
                'age_at_test',
                'age_band',
                'sex_binary',
                'delta_dias_digital_papel',
                'disease_duration_years',
                'education_band',
                'edss_band',
                'cognitive_burden_band',
                'physical_impact_band',
                'disease_duration_band',
            ]
            if column in df_clean.columns
        ]
        metadata = df_clean.loc[X.index, metadata_columns].copy() if metadata_columns else pd.DataFrame(index=X.index)

        # Get patient IDs for grouped CV
        if 'patient_id' not in df_clean.columns:
            raise DataProcessingError(_("Columna 'patient_id' no encontrada. Requerida para validación cruzada agrupada."))
        
        groups = df_clean.loc[X.index, 'patient_id'].copy()
        
        # Perform hold-out split by patient to avoid leakage across sessions
        logger.info(_("Realizando partición train-test (%.0f%% train, %.0f%% test)..."), (1-test_size)*100, test_size*100)

        unique_groups = groups.nunique()
        if unique_groups < 2:
            raise DataProcessingError(
                _("Se requieren al menos 2 pacientes distintos para separar train y test. Detectados: %d") % unique_groups
            )

        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y=None, groups=groups))

        X_train = X.iloc[train_idx].reset_index(drop=True)
        X_test = X.iloc[test_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        y_test = y.iloc[test_idx].reset_index(drop=True)
        groups_train = groups.iloc[train_idx].reset_index(drop=True)
        groups_test = groups.iloc[test_idx].reset_index(drop=True)
        metadata_train = metadata.iloc[train_idx].reset_index(drop=True)
        metadata_test = metadata.iloc[test_idx].reset_index(drop=True)

        all_nan_train_cols = X_train.columns[X_train.isna().all()].tolist()
        if all_nan_train_cols:
            logger.warning(
                _("Se eliminan %d features sin ningún valor observado en train antes de imputar."),
                len(all_nan_train_cols)
            )
            X_train = X_train.drop(columns=all_nan_train_cols)
            X_test = X_test.drop(columns=all_nan_train_cols, errors='ignore')
            self.last_feature_audit['removed_by_reason']['all_nan_in_train'] = all_nan_train_cols

        max_missing_rate = float(self.feature_filter_config.get('max_train_missing_rate', 0.35))
        high_missing_cols = X_train.columns[X_train.isna().mean() > max_missing_rate].tolist()
        if high_missing_cols:
            logger.warning(
                _("Se eliminan %d features con tasa de nulos en train superior a %.0f%%."),
                len(high_missing_cols),
                max_missing_rate * 100.0
            )
            X_train = X_train.drop(columns=high_missing_cols)
            X_test = X_test.drop(columns=high_missing_cols, errors='ignore')
            self.last_feature_audit['removed_by_reason']['high_missing_rate_in_train'] = high_missing_cols

        min_unique_values = int(self.feature_filter_config.get('min_train_unique_values', 2))
        low_unique_cols = X_train.columns[X_train.nunique(dropna=True) < min_unique_values].tolist()
        if low_unique_cols:
            logger.warning(
                _("Se eliminan %d features con variación insuficiente en train."),
                len(low_unique_cols)
            )
            X_train = X_train.drop(columns=low_unique_cols)
            X_test = X_test.drop(columns=low_unique_cols, errors='ignore')
            self.last_feature_audit['removed_by_reason']['low_unique_values_in_train'] = low_unique_cols

        min_train_std = float(self.feature_filter_config.get('min_train_std', 1e-8))
        low_variance_cols = X_train.columns[X_train.std(ddof=0).fillna(0.0) <= min_train_std].tolist()
        if low_variance_cols:
            logger.warning(
                _("Se eliminan %d features con desviación estándar casi nula en train."),
                len(low_variance_cols)
            )
            X_train = X_train.drop(columns=low_variance_cols)
            X_test = X_test.drop(columns=low_variance_cols, errors='ignore')
            self.last_feature_audit['removed_by_reason']['low_variance_in_train'] = low_variance_cols

        missing_train = int(X_train.isnull().sum().sum())
        missing_test = int(X_test.isnull().sum().sum())
        if missing_train or missing_test:
            logger.warning(
                _("Se detectaron valores faltantes en features tras el split. Train: %d | Test: %d. Imputando con medias de train."),
                missing_train,
                missing_test
            )
            train_means = X_train.mean()
            X_train = X_train.fillna(train_means)
            X_test = X_test.fillna(train_means)

        if test_type_lower == 'tmt' and self.feature_filter_config.get('tmt_transform_strategy', 'log1p') == 'log1p':
            transform_candidates = [
                column for column in X_train.columns
                if self._match_any_pattern(column, self._tmt_transform_patterns())
            ]
            X_train, X_test, transformed_columns = self._apply_train_fitted_log_transforms(
                X_train,
                X_test,
                transform_candidates,
            )
            self.last_feature_audit['transformed_columns'] = transformed_columns

        if test_type_lower == 'tmt' and self.feature_filter_config.get('tmt_feature_selection_enabled', True):
            X_train, X_test, selected_columns = self._select_tmt_features_via_rf_rfe(
                X_train,
                X_test,
                y_train,
            )
            removed_after_selection = [
                column for column in self.last_feature_audit.get('selected_features_initial', [])
                if column not in selected_columns
            ]
            self.last_feature_audit['removed_by_reason']['rf_rfe_elimination'] = removed_after_selection
            self.last_feature_audit['rf_rfe_selected_features'] = selected_columns

        logger.info(_("Train: %d muestras | Test: %d muestras | Features: %d"), 
                   len(X_train), len(X_test), X_train.shape[1])
        for target_name in y_train.columns:
            valid_target = y_train[target_name].dropna()
            logger.info(_("Target %s - muestras válidas train: %d"), target_name, len(valid_target))
            if not valid_target.empty:
                logger.info(
                    _("Target %s - Media: %.2f, Std: %.2f, Min: %.2f, Max: %.2f"),
                    target_name,
                    valid_target.mean(),
                    valid_target.std(),
                    valid_target.min(),
                    valid_target.max()
                )
        
        self.last_feature_audit['selected_features_final'] = X_train.columns.astype(str).tolist()
        self.last_feature_audit['n_features_final'] = int(X_train.shape[1])

        return X_train, X_test, y_train, y_test, groups_train, groups_test, metadata_train, metadata_test, target_groups

    @staticmethod
    def _find_first_matching_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Finds the first column whose normalized name matches any candidate."""
        normalized_columns = {
            DataProcessor._normalize_lookup_text(col): col for col in df.columns
        }

        for candidate in candidates:
            direct_match = normalized_columns.get(DataProcessor._normalize_lookup_text(candidate))
            if direct_match is not None:
                return direct_match

        for candidate in candidates:
            candidate_lower = DataProcessor._normalize_lookup_text(candidate)
            for col in df.columns:
                normalized_col = DataProcessor._normalize_lookup_text(col)
                if candidate_lower in normalized_col:
                    return col

        return None

    @classmethod
    def _resolve_target_columns(cls, df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, str]:
        """Resolves logical target names to actual dataframe columns."""
        target_defs = config.get('target_columns')
        if target_defs:
            resolved_targets: Dict[str, str] = {}
            for target_def in target_defs:
                resolved_targets[target_def['name']] = cls._find_target_column(df, target_def['hints'])
            return resolved_targets

        target_col = cls._find_target_column(df, config['target_hints'])
        return {config.get('target_name', 'target'): target_col}

    @staticmethod
    def _build_target_groups(config: Dict[str, Any], target_mapping: Dict[str, str]) -> Dict[str, List[str]]:
        """Builds target groups for optional grouped multi-output training."""
        target_defs = config.get('target_columns')
        if not target_defs:
            return {'default': list(target_mapping.keys())}

        groups: Dict[str, List[str]] = {}
        for target_def in target_defs:
            group_name = target_def.get('group', 'default')
            target_name = target_def['name']
            if target_name in target_mapping:
                groups.setdefault(group_name, []).append(target_name)

        return groups or {'default': list(target_mapping.keys())}

    @classmethod
    def _add_clinical_covariates(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Adds clinically relevant derived covariates such as age, sex and timing."""
        df_enriched = df.copy()

        birth_col = cls._find_first_matching_column(
            df_enriched,
            ['fecha nacimiento', 'f_nacim', 'f_nacimiento', 'fecha_nacimiento', 'birth_date', 'dob']
        )
        assessment_col = cls._find_first_matching_column(
            df_enriched,
            ['dia1 monitzacio fecha', 'dia1 monitorizacion fecha', 'dia1 monitorizacio fecha', 'fecha monitorizacion', 'fecha monitoriz.', 'fecha test', 'assessment_date', 'test_date', 'date_data']
        )
        digital_date_col = cls._find_first_matching_column(
            df_enriched,
            ['ts_created', 'created_at', 'digital_date', 'digital assessment date', 'date_data', 'fecha app']
        )

        logger.info(
            _("Detección de covariables clínicas - nacimiento: %s | fecha de prueba clínica: %s | fecha digital: %s"),
            birth_col or _("no detectada"),
            assessment_col or _("no detectada"),
            digital_date_col or _("no detectada")
        )

        if 'clinical_source_sheet' in df_enriched.columns:
            source_series = df_enriched['clinical_source_sheet'].astype(str).str.strip()
            normalized_source = source_series.str.lower()
            clinical_group = pd.Series(pd.NA, index=df_enriched.index, dtype='object')
            clinical_group = clinical_group.mask(normalized_source.str.contains('control', na=False), 'Controles')
            clinical_group = clinical_group.mask(normalized_source.str.fullmatch('em', na=False), 'EM')
            clinical_group = clinical_group.mask(clinical_group.isna(), source_series.where(source_series != 'nan', pd.NA))
            if clinical_group.notna().any():
                df_enriched['clinical_group'] = clinical_group
                logger.info(
                    _("Feature clínica derivada añadida: clinical_group (%d valores válidos)"),
                    int(clinical_group.notna().sum())
                )

        education_col = cls._find_first_matching_column(
            df_enriched,
            ['nivel_educ', 'educa', 'education_level', 'education', 'nivel estudios', 'estudios']
        )
        if education_col:
            education_series = df_enriched[education_col].astype(str).str.strip().str.upper()
            education_map = {
                '1': 'Primary',
                'PRIMARIA': 'Primary',
                'PRIMARY': 'Primary',
                '2': 'Secondary',
                'SECUNDARIA': 'Secondary',
                'SECONDARY': 'Secondary',
                'BACHILLERATO': 'Secondary',
                'G': 'University',
                'GRADO': 'University',
                'UNIVERSITARIO': 'University',
                'UNIVERSIDAD': 'University',
                'M': 'University',
                'MASTER': 'University',
                'D': 'University',
                'DOCTORADO': 'University',
                'PHD': 'University',
            }
            education_band = education_series.map(education_map)
            if education_band.notna().any():
                df_enriched['education_band'] = pd.Series(education_band, index=df_enriched.index, dtype='string')

        assessment_dates = None
        if assessment_col:
            assessment_dates = pd.to_datetime(df_enriched[assessment_col], errors='coerce', utc=True).dt.tz_localize(None)

        duration_col = cls._find_first_matching_column(
            df_enriched,
            [
                'tiempo evolucion io estudio',
                'tiempo evolución io estudio',
                'tiempo evolucion',
                'tiempo evolución',
                'disease duration',
                'years with disease',
                'anos evolucion',
                'años evolucion',
            ]
        )
        if duration_col:
            duration_values = pd.to_numeric(df_enriched[duration_col], errors='coerce')
            valid_duration = duration_values.where((duration_values >= 0.0) & (duration_values <= 80.0))
            if valid_duration.notna().any():
                df_enriched['disease_duration_years'] = valid_duration
                duration_band = pd.cut(
                    valid_duration,
                    bins=[-0.001, 5.0, 15.0, 80.0],
                    labels=['Disease_short', 'Disease_mid', 'Disease_long'],
                    include_lowest=True,
                )
                if duration_band.notna().any():
                    df_enriched['disease_duration_band'] = duration_band.astype('string')
                logger.info(
                    _("Feature clínica derivada añadida: disease_duration_years (%d valores válidos, media %.2f, rango %.2f-%.2f)"),
                    int(valid_duration.notna().sum()),
                    float(valid_duration.mean()),
                    float(valid_duration.min()),
                    float(valid_duration.max())
                )

        if birth_col and assessment_col:
            birth_dates = pd.to_datetime(df_enriched[birth_col], errors='coerce', utc=True).dt.tz_localize(None)
            age_years = (assessment_dates - birth_dates).dt.days / 365.25
            valid_age = age_years.where((age_years >= 0) & (age_years <= 120))

            if valid_age.notna().any():
                df_enriched['age_at_test'] = valid_age
                age_band = pd.cut(
                    valid_age,
                    bins=[0.0, 39.999, 54.999, 120.0],
                    labels=['<40', '40-54', '55+'],
                    include_lowest=True
                )
                if age_band.notna().any():
                    df_enriched['age_band'] = age_band.astype('string')
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

            if digital_date_col:
                digital_dates = pd.to_datetime(df_enriched[digital_date_col], errors='coerce', utc=True).dt.tz_localize(None)
                delta_days = (assessment_dates - digital_dates).dt.total_seconds() / 86400.0
                valid_delta = delta_days.where(delta_days.abs() <= 365)
                if valid_delta.notna().any():
                    df_enriched['delta_dias_digital_papel'] = valid_delta
                    logger.info(
                        _("Feature clínica derivada añadida: delta_dias_digital_papel (%d valores válidos, media %.2f, rango %.2f-%.2f)"),
                        int(valid_delta.notna().sum()),
                        float(valid_delta.mean()),
                        float(valid_delta.min()),
                        float(valid_delta.max())
                    )
                else:
                    logger.warning(
                        _("No fue posible derivar delta_dias_digital_papel a partir de %s y %s"),
                        assessment_col, digital_date_col
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

        edss_col = cls._find_first_matching_column(df_enriched, ['edss'])
        if edss_col:
            edss_values = pd.to_numeric(df_enriched[edss_col], errors='coerce')
            edss_band = pd.cut(
                edss_values,
                bins=[-0.001, 2.5, 4.5, 10.0],
                labels=['EDSS_mild', 'EDSS_moderate', 'EDSS_advanced'],
                include_lowest=True,
            )
            if edss_band.notna().any():
                df_enriched['edss_band'] = edss_band.astype('string')

        mfis_cog_col = cls._find_first_matching_column(df_enriched, ['m-fis score cog', 'mfis score cog', 'cogn'])
        if mfis_cog_col:
            mfis_cog_values = pd.to_numeric(df_enriched[mfis_cog_col], errors='coerce')
            cognitive_band = pd.cut(
                mfis_cog_values,
                bins=[-0.001, 17.0, 30.0, 60.0],
                labels=['Cog_low', 'Cog_moderate', 'Cog_high'],
                include_lowest=True,
            )
            if cognitive_band.notna().any():
                df_enriched['cognitive_burden_band'] = cognitive_band.astype('string')

        msis_phys_col = cls._find_first_matching_column(df_enriched, ['msis-29 score impacto fis', 'msis 29 score impacto fis', 'impacto fis'])
        if msis_phys_col:
            msis_phys_values = pd.to_numeric(df_enriched[msis_phys_col], errors='coerce')
            physical_band = pd.cut(
                msis_phys_values,
                bins=[-0.001, 29.0, 50.0, 100.0],
                labels=['Phys_low', 'Phys_moderate', 'Phys_high'],
                include_lowest=True,
            )
            if physical_band.notna().any():
                df_enriched['physical_impact_band'] = physical_band.astype('string')

        return df_enriched

    @classmethod
    def _encode_low_cardinality_categoricals(
        cls,
        df: pd.DataFrame,
        exclude_columns: List[str],
        include_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Encodes compatible low-cardinality categoricals into numeric dummy columns."""
        df_encoded = df.copy()
        encoded_columns: List[str] = []

        for column in df.columns:
            if column in exclude_columns:
                continue
            if include_columns is not None and column not in include_columns:
                continue
            if cls._is_date_like_column(str(column)) or cls._is_identifier_like_column(str(column)):
                continue
            if any(token in str(column).lower() for token in ["sexo", "sex", "gender", "genero"]):
                continue

            series = df[column]
            if pd.api.types.is_bool_dtype(series):
                df_encoded[column] = series.astype(float)
                encoded_columns.append(str(column))
                continue

            if not (
                pd.api.types.is_object_dtype(series)
                or pd.api.types.is_string_dtype(series)
                or cls._is_categorical_series(series)
            ):
                continue

            non_null = series.dropna()
            nunique = int(non_null.nunique())
            if nunique < 2 or nunique > 6:
                continue

            dummies = pd.get_dummies(non_null.astype("string"), prefix=str(column), dtype=float)
            if dummies.empty:
                continue

            dummies = dummies.reindex(df.index, fill_value=0.0)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            encoded_columns.extend(dummies.columns.astype(str).tolist())

        if encoded_columns:
            logger.info(
                _("Columnas categóricas codificadas a dummies: %d nuevas features"),
                len(encoded_columns)
            )
            logger.debug(_("Primeras columnas categóricas codificadas: %s"), encoded_columns[:20])

        return df_encoded
    
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
        normalized_columns = {
            DataProcessor._normalize_lookup_text(col): col for col in df.columns
        }

        # First try exact normalized matches
        for hint in hints:
            normalized_hint = DataProcessor._normalize_lookup_text(hint)
            if normalized_hint in normalized_columns:
                match = normalized_columns[normalized_hint]
                logger.debug(_("Target encontrado (coincidencia exacta): %s"), match)
                return match
        
        # Then try normalized substring matches
        for hint in hints:
            hint_lower = DataProcessor._normalize_lookup_text(hint)
            for col in df.columns:
                normalized_col = DataProcessor._normalize_lookup_text(col)
                if hint_lower in normalized_col:
                    logger.debug(_("Target encontrado (coincidencia parcial): %s (hint: %s)"), col, hint)
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
