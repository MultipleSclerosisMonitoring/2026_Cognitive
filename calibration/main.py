import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.model_selection import GroupShuffleSplit

from calibration.i18n import configure_language, get_translator, normalize_lang

def load_environment() -> None:
    """Loads environment variables from the repo root or legacy calibration path."""
    repo_root = Path(__file__).resolve().parents[1]
    env_candidates = [
        repo_root / '.env',
        repo_root / 'calibration' / '.env',
    ]

    for env_path in env_candidates:
        if env_path.exists():
            load_dotenv(env_path, override=False)


load_environment()

_ = get_translator()


def setup_logging(verbosity: int) -> None:
    """Configures the application's logging system based on user input."""
    levels = [logging.CRITICAL, logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG]
    level = levels[min(verbosity, 4)]
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')


def _quantile_higher(values: np.ndarray, quantile: float) -> float:
    """Returns a conservative empirical quantile using the 'higher' rule."""
    cleaned = np.asarray(values, dtype=float)
    cleaned = cleaned[np.isfinite(cleaned)]
    if cleaned.size == 0:
        return float('nan')

    quantile = float(min(max(quantile, 0.0), 1.0))
    try:
        return float(np.quantile(cleaned, quantile, method='higher'))
    except TypeError:
        return float(np.quantile(cleaned, quantile, interpolation='higher'))


def _split_conformal_subset(
    X_train: pd.DataFrame,
    y_train: pd.Series | pd.DataFrame,
    groups_train: pd.Series,
    calibration_size: float,
    random_state: int,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Creates a grouped train/calibration split for split-conformal inference."""
    unique_groups = groups_train.nunique()
    if unique_groups < 4:
        return None

    splitter = GroupShuffleSplit(n_splits=1, test_size=calibration_size, random_state=random_state)
    fit_idx, cal_idx = next(splitter.split(X_train, y=None, groups=groups_train))

    fit_groups = groups_train.iloc[fit_idx]
    cal_groups = groups_train.iloc[cal_idx]
    if fit_groups.nunique() < 2 or cal_groups.nunique() < 1:
        return None
    if len(fit_idx) < 10 or len(cal_idx) < 5:
        return None

    return fit_idx, cal_idx


def _fit_model(
    processor: Any,
    model_type: str,
    multi_output: bool,
    X_train: pd.DataFrame,
    y_train: pd.Series | pd.DataFrame,
    groups_train: pd.Series,
    cv_folds_num: int,
    param_grid: Optional[Dict[str, List[Any]]],
    search_strategy: str,
    n_iter: int,
):
    """Fits a calibrator using grouped cross-validation."""
    from calibration.models.algoritmos import get_model

    calibrator = get_model(model_type, multi_output=multi_output)
    cv_splits = list(processor.get_cv_folds(groups_train.reset_index(drop=True), n_splits=cv_folds_num))
    calibrator.train(
        X=X_train.reset_index(drop=True),
        y=y_train.reset_index(drop=True),
        param_grid=param_grid,
        cv_folds=cv_splits,
        search_strategy=search_strategy,
        n_iter=n_iter,
    )
    return calibrator






def _resolve_output_path(path_value: str) -> str:
    """Resolves output paths relative to the repository root."""
    candidate = Path(path_value)
    if candidate.is_absolute():
        return str(candidate)
    repo_root = Path(__file__).resolve().parents[1]
    return str((repo_root / candidate).resolve())
def _resolve_existing_path(path_value: str, config_path: str) -> str:
    """Resolves a path trying config-relative and repo-relative locations."""
    candidate = Path(path_value)
    if candidate.is_absolute():
        return str(candidate)

    config_dir = Path(config_path).resolve().parent
    repo_root = Path(__file__).resolve().parents[1]
    search_order = [
        config_dir / candidate,
        repo_root / candidate,
        Path.cwd() / candidate,
    ]
    for resolved in search_order:
        if resolved.exists():
            return str(resolved)

    return str((config_dir / candidate).resolve())

def _fit_predict_with_optional_conformal(
    processor: Any,
    model_type: str,
    multi_output: bool,
    X_train: pd.DataFrame,
    y_train: pd.Series | pd.DataFrame,
    groups_train: pd.Series,
    X_test: pd.DataFrame,
    cv_folds_num: int,
    param_grid: Optional[Dict[str, List[Any]]],
    search_strategy: str,
    n_iter: int,
    uncertainty_cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Trains the model and optionally estimates split-conformal prediction intervals."""
    conformal_enabled = bool(uncertainty_cfg.get('enabled', True))
    alpha = float(uncertainty_cfg.get('conformal_alpha', 0.10))
    calibration_size = float(uncertainty_cfg.get('calibration_size', 0.20))
    random_state = int(uncertainty_cfg.get('random_state', 42))

    split_indices = None
    if conformal_enabled:
        split_indices = _split_conformal_subset(
            X_train=X_train,
            y_train=y_train,
            groups_train=groups_train,
            calibration_size=calibration_size,
            random_state=random_state,
        )
        if split_indices is None:
            logging.warning(
                _(
                    'No hay suficientes pacientes para activar split-conformal de forma robusta. Se entrenará sin intervalos para este ajuste.'
                )
            )

    if split_indices is None:
        calibrator = _fit_model(
            processor=processor,
            model_type=model_type,
            multi_output=multi_output,
            X_train=X_train,
            y_train=y_train,
            groups_train=groups_train,
            cv_folds_num=cv_folds_num,
            param_grid=param_grid,
            search_strategy=search_strategy,
            n_iter=n_iter,
        )
        predictions = calibrator.predict(X_test.reset_index(drop=True))
        return predictions, None, None

    fit_idx, cal_idx = split_indices
    X_fit = X_train.iloc[fit_idx].reset_index(drop=True)
    y_fit = y_train.iloc[fit_idx].reset_index(drop=True)
    groups_fit = groups_train.iloc[fit_idx].reset_index(drop=True)
    X_cal = X_train.iloc[cal_idx].reset_index(drop=True)
    y_cal = y_train.iloc[cal_idx].reset_index(drop=True)

    calibrator = _fit_model(
        processor=processor,
        model_type=model_type,
        multi_output=multi_output,
        X_train=X_fit,
        y_train=y_fit,
        groups_train=groups_fit,
        cv_folds_num=cv_folds_num,
        param_grid=param_grid,
        search_strategy=search_strategy,
        n_iter=n_iter,
    )

    cal_predictions = np.asarray(calibrator.predict(X_cal), dtype=float)
    test_predictions = np.asarray(calibrator.predict(X_test.reset_index(drop=True)), dtype=float)
    cal_truth = np.asarray(y_cal, dtype=float)

    if cal_predictions.ndim == 1:
        cal_predictions = cal_predictions.reshape(-1, 1)
        test_predictions = test_predictions.reshape(-1, 1)
        cal_truth = cal_truth.reshape(-1, 1)

    qhat_values: List[float] = []
    lower_bounds = np.empty_like(test_predictions, dtype=float)
    upper_bounds = np.empty_like(test_predictions, dtype=float)

    for col_idx in range(cal_predictions.shape[1]):
        scores = np.abs(cal_truth[:, col_idx] - cal_predictions[:, col_idx])
        quantile_level = min(1.0, np.ceil((len(scores) + 1) * (1.0 - alpha)) / max(len(scores), 1))
        qhat = _quantile_higher(scores, quantile_level)
        qhat_values.append(qhat)
        lower_bounds[:, col_idx] = test_predictions[:, col_idx] - qhat
        upper_bounds[:, col_idx] = test_predictions[:, col_idx] + qhat

    logging.info(
        _('Intervalos conformales calculados. Alpha: %.3f | qhat por target: %s'),
        alpha,
        ', '.join(f'{value:.4f}' for value in qhat_values),
    )

    if test_predictions.shape[1] == 1:
        return test_predictions.ravel(), lower_bounds.ravel(), upper_bounds.ravel()

    return test_predictions, lower_bounds, upper_bounds


def _fit_predict_with_optional_stratification(
    processor: Any,
    model_type: str,
    multi_output: bool,
    X_train: pd.DataFrame,
    y_train: pd.Series | pd.DataFrame,
    groups_train: pd.Series,
    metadata_train: pd.DataFrame,
    X_test: pd.DataFrame,
    metadata_test: pd.DataFrame,
    cv_folds_num: int,
    param_grid: Optional[Dict[str, List[Any]]],
    search_strategy: str,
    n_iter: int,
    uncertainty_cfg: Dict[str, Any],
    stratify_by: Optional[str] = None,
    min_train_samples_per_stratum: int = 10,
    min_unique_groups_per_stratum: int = 2,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Routes calibration through subgroup-specific models with a global fallback."""
    if not stratify_by:
        return _fit_predict_with_optional_conformal(
            processor=processor,
            model_type=model_type,
            multi_output=multi_output,
            X_train=X_train,
            y_train=y_train,
            groups_train=groups_train,
            X_test=X_test,
            cv_folds_num=cv_folds_num,
            param_grid=param_grid,
            search_strategy=search_strategy,
            n_iter=n_iter,
            uncertainty_cfg=uncertainty_cfg,
        )

    if stratify_by not in metadata_train.columns or stratify_by not in metadata_test.columns:
        logging.warning(_("No se encontró la columna de estratificación '%s'. Se usará el modelo global."), stratify_by)
        return _fit_predict_with_optional_conformal(
            processor=processor,
            model_type=model_type,
            multi_output=multi_output,
            X_train=X_train,
            y_train=y_train,
            groups_train=groups_train,
            X_test=X_test,
            cv_folds_num=cv_folds_num,
            param_grid=param_grid,
            search_strategy=search_strategy,
            n_iter=n_iter,
            uncertainty_cfg=uncertainty_cfg,
        )

    train_strata = metadata_train[stratify_by].astype('string').reset_index(drop=True)
    test_strata = metadata_test[stratify_by].astype('string').reset_index(drop=True)
    if train_strata.notna().sum() == 0 or test_strata.notna().sum() == 0:
        logging.warning(_("La estratificación '%s' no contiene valores suficientes. Se usará el modelo global."), stratify_by)
        return _fit_predict_with_optional_conformal(
            processor=processor,
            model_type=model_type,
            multi_output=multi_output,
            X_train=X_train,
            y_train=y_train,
            groups_train=groups_train,
            X_test=X_test,
            cv_folds_num=cv_folds_num,
            param_grid=param_grid,
            search_strategy=search_strategy,
            n_iter=n_iter,
            uncertainty_cfg=uncertainty_cfg,
        )

    subgroup_results: List[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]] = []
    fallback_positions: List[int] = []

    for subgroup in pd.unique(test_strata.dropna()):
        subgroup_train_mask = train_strata.eq(subgroup)
        subgroup_test_mask = test_strata.eq(subgroup)
        subgroup_positions = np.flatnonzero(subgroup_test_mask.to_numpy())
        subgroup_train_n = int(subgroup_train_mask.sum())
        subgroup_group_count = int(groups_train.loc[subgroup_train_mask].nunique()) if subgroup_train_n else 0

        if subgroup_positions.size == 0:
            continue

        if subgroup_train_n < min_train_samples_per_stratum or subgroup_group_count < min_unique_groups_per_stratum:
            logging.info(
                _(
                    "Subgrupo %s en '%s' se servirá con fallback global. Train=%d | Pacientes=%d"
                ),
                subgroup,
                stratify_by,
                subgroup_train_n,
                subgroup_group_count,
            )
            fallback_positions.extend(subgroup_positions.tolist())
            continue

        predictions, interval_low, interval_high = _fit_predict_with_optional_conformal(
            processor=processor,
            model_type=model_type,
            multi_output=multi_output,
            X_train=X_train.loc[subgroup_train_mask].reset_index(drop=True),
            y_train=y_train.loc[subgroup_train_mask].reset_index(drop=True),
            groups_train=groups_train.loc[subgroup_train_mask].reset_index(drop=True),
            X_test=X_test.loc[subgroup_test_mask].reset_index(drop=True),
            cv_folds_num=cv_folds_num,
            param_grid=param_grid,
            search_strategy=search_strategy,
            n_iter=n_iter,
            uncertainty_cfg=uncertainty_cfg,
        )
        subgroup_results.append((subgroup_positions, np.asarray(predictions, dtype=float), interval_low, interval_high))

    missing_positions = np.flatnonzero(test_strata.isna().to_numpy())
    if missing_positions.size:
        fallback_positions.extend(missing_positions.tolist())

    if fallback_positions:
        fallback_positions = sorted(set(fallback_positions))
        predictions, interval_low, interval_high = _fit_predict_with_optional_conformal(
            processor=processor,
            model_type=model_type,
            multi_output=multi_output,
            X_train=X_train.reset_index(drop=True),
            y_train=y_train.reset_index(drop=True),
            groups_train=groups_train.reset_index(drop=True),
            X_test=X_test.iloc[fallback_positions].reset_index(drop=True),
            cv_folds_num=cv_folds_num,
            param_grid=param_grid,
            search_strategy=search_strategy,
            n_iter=n_iter,
            uncertainty_cfg=uncertainty_cfg,
        )
        subgroup_results.append((np.asarray(fallback_positions, dtype=int), np.asarray(predictions, dtype=float), interval_low, interval_high))

    if not subgroup_results:
        return _fit_predict_with_optional_conformal(
            processor=processor,
            model_type=model_type,
            multi_output=multi_output,
            X_train=X_train,
            y_train=y_train,
            groups_train=groups_train,
            X_test=X_test,
            cv_folds_num=cv_folds_num,
            param_grid=param_grid,
            search_strategy=search_strategy,
            n_iter=n_iter,
            uncertainty_cfg=uncertainty_cfg,
        )

    first_predictions = np.asarray(subgroup_results[0][1], dtype=float)
    predictions_shape = (len(X_test),) if first_predictions.ndim == 1 else (len(X_test), first_predictions.shape[1])
    combined_predictions = np.full(predictions_shape, np.nan, dtype=float)

    intervals_available = all(low is not None and high is not None for _, _, low, high in subgroup_results)
    combined_low = np.full(predictions_shape, np.nan, dtype=float) if intervals_available else None
    combined_high = np.full(predictions_shape, np.nan, dtype=float) if intervals_available else None

    for positions, predictions, interval_low, interval_high in subgroup_results:
        combined_predictions[positions] = predictions
        if intervals_available and combined_low is not None and combined_high is not None:
            combined_low[positions] = np.asarray(interval_low, dtype=float)
            combined_high[positions] = np.asarray(interval_high, dtype=float)

    if np.isnan(combined_predictions).any():
        raise ValueError(_("La estratificación produjo predicciones incompletas. Revise la cobertura de subgrupos."))

    if combined_predictions.ndim == 2 and combined_predictions.shape[1] == 1:
        combined_predictions = combined_predictions.ravel()
        if combined_low is not None and combined_high is not None:
            combined_low = combined_low.ravel()
            combined_high = combined_high.ravel()

    return combined_predictions, combined_low, combined_high


def main() -> None:
    """Main entry point for orchestrating the clinical calibration pipeline."""
    global _

    parser = argparse.ArgumentParser(description=_("Sistema de calibracion de tests cognitivos SDMT y TMT para dispositivos moviles."))
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help=_("Ruta absoluta o relativa al archivo YAML que contiene la configuracion del experimento."),
    )
    parser.add_argument(
        "--verbose",
        type=int,
        choices=[0, 1, 2, 3, 4],
        default=2,
        help=_("Nivel de verbosidad (0=Critico, 4=Depuracion maxima). Por defecto es 2 (Avisos)."),
    )
    parser.add_argument(
        "--lang",
        type=str,
        default=None,
        help=_("Idioma de la interfaz (por ejemplo: es, en, fr). Si no se indica, usa APP_LANG o LANG."),
    )
    args = parser.parse_args()

    setup_logging(args.verbose)

    selected_lang = normalize_lang(args.lang or os.environ.get("APP_LANG") or os.environ.get("LANG"))
    translation = configure_language(selected_lang)
    _ = translation.gettext

    from calibration.data.loader import DataProcessor, DataProcessingError
    from calibration.utils.reporting import ReportGenerator
    from calibration.utils.clinical_discussion import (
        build_clinical_discussion_workbook,
        save_feature_audit,
        save_stratified_metrics,
    )

    config_path = os.path.abspath(args.config)

    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        logging.critical(_("Fallo critico al leer el archivo de configuracion YAML: %s"), str(e))
        return

    if "language" in config:
        logging.warning(
            _(
                "La clave 'language' en config.yaml esta obsoleta y ya no tiene efecto. Use --lang o APP_LANG en su archivo .env."
            )
        )

    logging.info(_("Iniciando el proceso de calibracion con configuracion: %s"), config_path)
    logging.info(_("Idioma activo: %s"), selected_lang)

    output_config = config.get("output", {})
    output_excel = _resolve_output_path(output_config.get("excel_report", "resultados.xlsx"))
    output_html = _resolve_output_path(output_config.get("html_plot", "graficos.html"))
    reporter = ReportGenerator(output_excel=output_excel, output_html=output_html)

    data_config = config.get("data", {})
    uncertainty_cfg = config.get("uncertainty", {})

    db_host = os.environ.get("DB_HOST", data_config.get("db_host", "localhost"))
    db_port = os.environ.get("DB_PORT", data_config.get("db_port", "5432"))
    db_user = os.environ.get("DB_USER", data_config.get("db_user"))
    db_password = os.environ.get("DB_PASSWORD", data_config.get("db_password"))
    db_name = os.environ.get("DB_NAME", data_config.get("db_name"))

    if not all([db_user, db_password, db_name]):
        logging.critical(
            _(
                "Credenciales de base de datos incompletas. Configure las variables de entorno DB_USER, DB_PASSWORD y DB_NAME o actualize el archivo config.yaml"
            )
        )
        return

    db_uri = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

    excel_path_value = os.environ.get("EXCEL_DATA_PATH") or data_config.get("excel_path") or "datos_papel.xlsx"
    if not isinstance(excel_path_value, str):
        logging.critical(_("La ruta del archivo clínico debe ser una cadena válida. Revise EXCEL_DATA_PATH o data.excel_path."))
        return

    excel_path = _resolve_existing_path(excel_path_value, config_path)
    test_type = data_config.get("test_type", "sdmt")
    column_config = data_config.get("column_mapping", None)
    clinical_skiprows = data_config.get("clinical_skiprows", 2)

    logging.info(_("Procesando extraccion de datos para el test objetivo: %s"), test_type.upper())
    processor = DataProcessor(
        db_uri=db_uri,
        excel_path=excel_path,
        column_config=column_config,
        clinical_skiprows=clinical_skiprows,
        feature_filter_config=data_config.get('feature_filters'),
    )

    try:
        df_merged = processor.load_and_merge(test_type)
    except DataProcessingError as e:
        logging.critical(_("Error critico en procesamiento de datos: %s"), str(e))
        return
    except Exception as e:
        logging.critical(_("Error inesperado durante carga de datos: %s"), str(e))
        return

    try:
        X_train, X_test, y_train, y_test, groups_train, groups_test, metadata_train, metadata_test, target_groups = processor.prepare_splits(df_merged, test_type)
    except DataProcessingError as e:
        logging.critical(_("Error en preparación de datos: %s"), str(e))
        return

    save_feature_audit(output_excel, test_type, processor.last_feature_audit)

    models_config: List[Dict[str, Any]] = config.get("models", [{"type": "linear", "cv_folds": 3}])
    bootstrap_iterations = int(uncertainty_cfg.get('bootstrap_iterations', 500))
    bootstrap_confidence = float(uncertainty_cfg.get('bootstrap_confidence', 0.95))
    conformal_alpha = float(uncertainty_cfg.get('conformal_alpha', 0.10))
    bootstrap_random_state = int(uncertainty_cfg.get('random_state', 42))

    logging.info(_("Se han detectado %d modelo(s) en la cola de ejecucion."), len(models_config))

    for i, model_cfg in enumerate(models_config, 1):
        model_type = model_cfg.get("type", "linear")
        cv_folds_num = model_cfg.get("cv_folds", 3)
        search_strategy = model_cfg.get("search_strategy", "grid")
        n_iter = model_cfg.get("n_iter", 10)
        param_grid = model_cfg.get("param_grid", None)
        target_mode = model_cfg.get("target_mode", "single_output")
        stratify_by = model_cfg.get("stratify_by")
        min_train_samples_per_stratum = int(model_cfg.get("min_train_samples_per_stratum", 10))
        min_unique_groups_per_stratum = int(model_cfg.get("min_unique_groups_per_stratum", 2))

        logging.info(_("--- [Iteracion %d/%d] Iniciando pipeline para modelo: %s ---"), i, len(models_config), model_type.upper())

        try:
            grouped_available = any(len(target_names) >= 2 for target_names in target_groups.values())
            if target_mode == "grouped_multi_output":
                eligible_groups = [
                    (group_name, target_names)
                    for group_name, target_names in target_groups.items()
                    if len(target_names) >= 2
                ]
                if not eligible_groups:
                    logging.info(_("No hay grupos multi-output elegibles para este test. Se omite este bloque para evitar duplicar el entrenamiento single-output."))

                for group_name, target_names in eligible_groups:
                    train_mask = y_train[target_names].notna().all(axis=1)
                    test_mask = y_test[target_names].notna().all(axis=1)
                    if train_mask.sum() < 10 or test_mask.sum() < 2:
                        logging.warning(
                            _(
                                "Grupo %s omitido por falta de muestras completas. Train válidas: %d | Test válidas: %d"
                            ),
                            group_name,
                            int(train_mask.sum()),
                            int(test_mask.sum()),
                        )
                        continue

                    run_name = f"{model_type}_{group_name}_multioutput"
                    group_X_train = X_train.loc[train_mask].reset_index(drop=True)
                    group_y_train = y_train.loc[train_mask, target_names].reset_index(drop=True)
                    group_groups_train = groups_train.loc[train_mask].reset_index(drop=True)
                    group_X_test = X_test.loc[test_mask].reset_index(drop=True)
                    group_y_test = y_test.loc[test_mask, target_names].reset_index(drop=True)
                    group_groups_test = groups_test.loc[test_mask].reset_index(drop=True)
                    group_metadata_test = metadata_test.loc[test_mask].reset_index(drop=True)

                    predictions, interval_low, interval_high = _fit_predict_with_optional_stratification(
                        processor=processor,
                        model_type=model_type,
                        multi_output=True,
                        X_train=group_X_train,
                        y_train=group_y_train,
                        groups_train=group_groups_train,
                        metadata_train=metadata_train.loc[train_mask].reset_index(drop=True),
                        X_test=group_X_test,
                        metadata_test=group_metadata_test,
                        cv_folds_num=cv_folds_num,
                        param_grid=param_grid,
                        search_strategy=search_strategy,
                        n_iter=n_iter,
                        uncertainty_cfg=uncertainty_cfg,
                        stratify_by=stratify_by,
                        min_train_samples_per_stratum=min_train_samples_per_stratum,
                        min_unique_groups_per_stratum=min_unique_groups_per_stratum,
                    )

                    predictions_df = pd.DataFrame(predictions, columns=target_names)
                    lower_df = pd.DataFrame(interval_low, columns=target_names) if interval_low is not None else None
                    upper_df = pd.DataFrame(interval_high, columns=target_names) if interval_high is not None else None

                    for target_name in target_names:
                        eval_name = f"{run_name}_{target_name}"
                        metrics = reporter.evaluate_and_save(
                            y_true=group_y_test[target_name],
                            y_pred=predictions_df[target_name],
                            model_name=eval_name,
                            groups=group_groups_test,
                            prediction_interval_low=None if lower_df is None else lower_df[target_name],
                            prediction_interval_high=None if upper_df is None else upper_df[target_name],
                            interval_alpha=conformal_alpha,
                            bootstrap_iterations=bootstrap_iterations,
                            bootstrap_confidence=bootstrap_confidence,
                            bootstrap_random_state=bootstrap_random_state,
                        )
                        save_stratified_metrics(
                            output_excel=output_excel,
                            model_name=eval_name,
                            y_true=group_y_test[target_name],
                            y_pred=predictions_df[target_name],
                            metadata=group_metadata_test,
                            prediction_interval_low=None if lower_df is None else lower_df[target_name],
                            prediction_interval_high=None if upper_df is None else upper_df[target_name],
                            interval_alpha=conformal_alpha,
                        )
                        reporter.generate_scatter_plot(y_true=group_y_test[target_name], y_pred=predictions_df[target_name], model_name=eval_name)
                        reporter.generate_residuals_plot(y_true=group_y_test[target_name], y_pred=predictions_df[target_name], model_name=eval_name)
                        reporter.generate_bland_altman_plot(y_true=group_y_test[target_name], y_pred=predictions_df[target_name], model_name=eval_name)
                        logging.info(_("Modelo %s evaluado con exito. RMSE: %.4f | R2: %.4f"), eval_name, metrics["RMSE"], metrics["R2_Score"])

            if target_mode != "grouped_multi_output":
                configured_targets = model_cfg.get('targets')
                allowed_targets = set(configured_targets) if configured_targets else None
                for target_name in y_train.columns:
                    if allowed_targets is not None and target_name not in allowed_targets:
                        continue
                    train_mask = y_train[target_name].notna()
                    test_mask = y_test[target_name].notna()
                    if train_mask.sum() < 10 or test_mask.sum() < 2:
                        logging.warning(
                            _("Target %s omitido por falta de muestras válidas. Train: %d | Test: %d"),
                            target_name,
                            int(train_mask.sum()),
                            int(test_mask.sum()),
                        )
                        continue

                    run_name = f"{model_type}_{target_name}"
                    target_X_train = X_train.loc[train_mask].reset_index(drop=True)
                    target_y_train = y_train.loc[train_mask, target_name].reset_index(drop=True)
                    target_groups_train = groups_train.loc[train_mask].reset_index(drop=True)
                    target_X_test = X_test.loc[test_mask].reset_index(drop=True)
                    target_y_test = y_test.loc[test_mask, target_name].reset_index(drop=True)
                    target_groups_test = groups_test.loc[test_mask].reset_index(drop=True)
                    target_metadata_test = metadata_test.loc[test_mask].reset_index(drop=True)

                    predictions, interval_low, interval_high = _fit_predict_with_optional_stratification(
                        processor=processor,
                        model_type=model_type,
                        multi_output=False,
                        X_train=target_X_train,
                        y_train=target_y_train,
                        groups_train=target_groups_train,
                        metadata_train=metadata_train.loc[train_mask].reset_index(drop=True),
                        X_test=target_X_test,
                        metadata_test=target_metadata_test,
                        cv_folds_num=cv_folds_num,
                        param_grid=param_grid,
                        search_strategy=search_strategy,
                        n_iter=n_iter,
                        uncertainty_cfg=uncertainty_cfg,
                        stratify_by=stratify_by,
                        min_train_samples_per_stratum=min_train_samples_per_stratum,
                        min_unique_groups_per_stratum=min_unique_groups_per_stratum,
                    )

                    logging.info(_("Generando reportes y metricas clinicas..."))
                    metrics = reporter.evaluate_and_save(
                        y_true=target_y_test,
                        y_pred=predictions,
                        model_name=run_name,
                        groups=target_groups_test,
                        prediction_interval_low=interval_low,
                        prediction_interval_high=interval_high,
                        interval_alpha=conformal_alpha,
                        bootstrap_iterations=bootstrap_iterations,
                        bootstrap_confidence=bootstrap_confidence,
                        bootstrap_random_state=bootstrap_random_state,
                    )
                    save_stratified_metrics(
                        output_excel=output_excel,
                        model_name=run_name,
                        y_true=target_y_test,
                        y_pred=predictions,
                        metadata=target_metadata_test,
                        prediction_interval_low=interval_low,
                        prediction_interval_high=interval_high,
                        interval_alpha=conformal_alpha,
                    )
                    reporter.generate_scatter_plot(y_true=target_y_test, y_pred=predictions, model_name=run_name)
                    reporter.generate_residuals_plot(y_true=target_y_test, y_pred=predictions, model_name=run_name)
                    reporter.generate_bland_altman_plot(y_true=target_y_test, y_pred=predictions, model_name=run_name)
                    logging.info(_("Modelo %s evaluado con exito. RMSE: %.4f | R2: %.4f"), run_name, metrics["RMSE"], metrics["R2_Score"])

        except ValueError as e:
            logging.error(_("Error de configuracion para el modelo %s: %s"), model_type, str(e))
        except Exception as e:
            logging.error(_("Error critico e inesperado ejecutando el modelo %s: %s"), model_type, str(e))

    build_clinical_discussion_workbook(
        output_excel=_resolve_output_path('comparativa_clinica_final.xlsx'),
        source_workbooks=[
            _resolve_output_path('resultados.xlsx'),
            _resolve_output_path('resultados_tmt.xlsx'),
        ]
    )
    logging.info(_("Ejecucion global completada. Todos los modelos han sido procesados y reportados."))


if __name__ == "__main__":
    main()
