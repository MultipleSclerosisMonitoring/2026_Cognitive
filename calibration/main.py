import argparse
import yaml
import logging
import os
import pandas as pd
from typing import Any, Dict, List
from dotenv import load_dotenv
from calibration.i18n import configure_language, get_translator, normalize_lang

# Load environment variables from .env file
load_dotenv()

_ = get_translator()

def setup_logging(verbosity: int) -> None:
    """Configures the application's logging system based on user input.

    This function maps an integer provided by the user via the Command Line
    Interface (CLI) to the standard logging levels available in Python's
    built-in logging library. It allows for a seamless transition from a
    completely silent execution (batch mode) to a deep debugging state.

    Args:
        verbosity (int): Desired level of detail. Must be an integer in the 
            range [0, 4], where 0 represents CRITICAL (only fatal errors), 
            1 is ERROR, 2 is WARNING, 3 is INFO, and 4 is DEBUG (maximum detail).

    Returns:
        None: This function does not return any value; it modifies the global 
        logging configuration.
    """
    levels = [logging.CRITICAL, logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG]
    level = levels[min(verbosity, 4)]
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')

def main() -> None:
    """Main entry point for orchestrating the clinical calibration pipeline.

    The execution flow of this method is divided into the following phases:
    Firstly, it parses command-line arguments using a preliminary i18n setup.
    Secondly, it loads and validates the external configuration from a YAML file.
    Thirdly, it configures the internationalization (i18n) engine based on the
    language specified via CLI or environment variables.
    Fourthly, it extracts and merges data from PostgreSQL databases and Excel files.
    Fifthly, it generates cross-validation splits using a grouped approach to 
    prevent data leakage across different patient records.
    Sixthly, it iterates over a battery of models defined in the configuration,
    executing training, hyperparameter optimization, and generating blind predictions.
    Finally, it invokes the reporting module to calculate performance metrics,
    save results to an Excel spreadsheet, and generate interactive Plotly HTML graphs.

    Raises:
        FileNotFoundError: If the specified YAML configuration file does not exist.
        yaml.YAMLError: If the configuration file contains syntax errors.
    """
    global _

    parser = argparse.ArgumentParser(description=_("Sistema de calibracion de tests cognitivos SDMT y TMT para dispositivos moviles."))
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help=_("Ruta absoluta o relativa al archivo YAML que contiene la configuracion del experimento.")
    )
    parser.add_argument(
        "--verbose", 
        type=int, 
        choices=[0, 1, 2, 3, 4], 
        default=2, 
        help=_("Nivel de verbosidad (0=Critico, 4=Depuracion maxima). Por defecto es 2 (Avisos).")
    )
    parser.add_argument(
        "--lang",
        type=str,
        default=None,
        help=_("Idioma de la interfaz (por ejemplo: es, en, fr). Si no se indica, usa APP_LANG o LANG.")
    )
    args = parser.parse_args()

    setup_logging(args.verbose)

    selected_lang = normalize_lang(args.lang or os.environ.get("APP_LANG") or os.environ.get("LANG"))
    translation = configure_language(selected_lang)
    _ = translation.gettext

    from calibration.models.algoritmos import get_model
    from calibration.data.loader import DataProcessor, DataProcessingError
    from calibration.utils.reporting import ReportGenerator
    
    try:
        with open(args.config, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        logging.critical(_("Fallo critico al leer el archivo de configuracion YAML: %s"), str(e))
        return

    if "language" in config:
        logging.warning(
            _("La clave 'language' en config.yaml esta obsoleta y ya no tiene efecto. Use --lang o APP_LANG en su archivo .env.")
        )

    logging.info(_("Iniciando el proceso de calibracion con configuracion: %s"), args.config)
    logging.info(_("Idioma activo: %s"), selected_lang)
    
    # Configuración de Rutas de Salida y Generador de Reportes
    output_config = config.get("output", {})
    output_excel = output_config.get("excel_report", "resultados.xlsx")
    output_html = output_config.get("html_plot", "graficos.html")
    reporter = ReportGenerator(output_excel=output_excel, output_html=output_html)
    
    # Procesamiento de Datos
    data_config = config.get("data", {})
    
    # Build database URI from environment variables (preferred) or config file (fallback)
    db_host = os.environ.get("DB_HOST", data_config.get("db_host", "localhost"))
    db_port = os.environ.get("DB_PORT", data_config.get("db_port", "5432"))
    db_user = os.environ.get("DB_USER", data_config.get("db_user"))
    db_password = os.environ.get("DB_PASSWORD", data_config.get("db_password"))
    db_name = os.environ.get("DB_NAME", data_config.get("db_name"))
    
    if not all([db_user, db_password, db_name]):
        logging.critical(_("Credenciales de base de datos incompletas. Configure las variables de entorno DB_USER, DB_PASSWORD y DB_NAME o actualize el archivo config.yaml"))
        return
    
    db_uri = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    
    excel_path_value = os.environ.get("EXCEL_DATA_PATH") or data_config.get("excel_path") or "datos_papel.xlsx"
    if not isinstance(excel_path_value, str):
        logging.critical(_("La ruta del archivo clínico debe ser una cadena válida. Revise EXCEL_DATA_PATH o data.excel_path."))
        return

    excel_path = excel_path_value
    test_type = data_config.get("test_type", "sdmt")
    column_config = data_config.get("column_mapping", None)
    clinical_skiprows = data_config.get("clinical_skiprows", 2)
    
    logging.info(_("Procesando extraccion de datos para el test objetivo: %s"), test_type.upper())
    processor = DataProcessor(
        db_uri=db_uri,
        excel_path=excel_path,
        column_config=column_config,
        clinical_skiprows=clinical_skiprows,
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
        X_train, X_test, y_train, y_test, groups_train, target_groups = processor.prepare_splits(df_merged, test_type)
    except DataProcessingError as e:
        logging.critical(_("Error en preparación de datos: %s"), str(e))
        return
    
    models_config: List[Dict[str, Any]] = config.get("models", [{"type": "linear", "cv_folds": 3}])
    
    logging.info(_("Se han detectado %d modelo(s) en la cola de ejecucion."), len(models_config))

    # Bucle de Entrenamiento y Evaluación
    for i, model_cfg in enumerate(models_config, 1):
        model_type = model_cfg.get("type", "linear")
        cv_folds_num = model_cfg.get("cv_folds", 3)
        search_strategy = model_cfg.get("search_strategy", "grid")
        n_iter = model_cfg.get("n_iter", 10)
        param_grid = model_cfg.get("param_grid", None)
        target_mode = model_cfg.get("target_mode", "single_output")
        
        logging.info(_("--- [Iteracion %d/%d] Iniciando pipeline para modelo: %s ---"), i, len(models_config), model_type.upper())

        try:
            if target_mode == "grouped_multi_output":
                eligible_groups = [(group_name, target_names) for group_name, target_names in target_groups.items() if len(target_names) >= 2]
                if not eligible_groups:
                    logging.info(_("No hay grupos multi-output elegibles para este test. Fallback a entrenamiento por target."))
                for group_name, target_names in eligible_groups:
                    train_mask = y_train[target_names].notna().all(axis=1)
                    test_mask = y_test[target_names].notna().all(axis=1)
                    if train_mask.sum() < 10 or test_mask.sum() < 2:
                        logging.warning(
                            _("Grupo %s omitido por falta de muestras completas. Train válidas: %d | Test válidas: %d"),
                            group_name,
                            int(train_mask.sum()),
                            int(test_mask.sum())
                        )
                        continue

                    run_name = f"{model_type}_{group_name}_multioutput"
                    calibrador = get_model(model_type, multi_output=True)
                    logging.debug(_("Instancia multi-output de %s generada para grupo %s."), model_type, group_name)
                    group_X_train = X_train.loc[train_mask].reset_index(drop=True)
                    group_y_train = y_train.loc[train_mask, target_names].reset_index(drop=True)
                    group_groups_train = groups_train.loc[train_mask].reset_index(drop=True)
                    group_X_test = X_test.loc[test_mask].reset_index(drop=True)
                    group_y_test = y_test.loc[test_mask, target_names].reset_index(drop=True)
                    cv_splits = list(processor.get_cv_folds(group_groups_train, n_splits=cv_folds_num))

                    calibrador.train(
                        X=group_X_train,
                        y=group_y_train,
                        param_grid=param_grid,
                        cv_folds=cv_splits,
                        search_strategy=search_strategy,
                        n_iter=n_iter
                    )

                    logging.info(_("Entrenamiento finalizado para %s. Obteniendo predicciones ciegas..."), run_name)
                    predicciones = calibrador.predict(group_X_test)
                    predicciones_df = pd.DataFrame(predicciones, columns=target_names)

                    for target_name in target_names:
                        eval_name = f"{run_name}_{target_name}"
                        metrics = reporter.evaluate_and_save(
                            y_true=group_y_test[target_name],
                            y_pred=predicciones_df[target_name],
                            model_name=eval_name
                        )
                        reporter.generate_scatter_plot(y_true=group_y_test[target_name], y_pred=predicciones_df[target_name], model_name=eval_name)
                        reporter.generate_residuals_plot(y_true=group_y_test[target_name], y_pred=predicciones_df[target_name], model_name=eval_name)
                        logging.info(_("Modelo %s evaluado con exito. RMSE: %.4f | R2: %.4f"), eval_name, metrics["RMSE"], metrics["R2_Score"])
            if target_mode != "grouped_multi_output" or not any(len(target_names) >= 2 for target_names in target_groups.values()):
                for target_name in y_train.columns:
                    train_mask = y_train[target_name].notna()
                    test_mask = y_test[target_name].notna()
                    if train_mask.sum() < 10 or test_mask.sum() < 2:
                        logging.warning(
                            _("Target %s omitido por falta de muestras válidas. Train: %d | Test: %d"),
                            target_name,
                            int(train_mask.sum()),
                            int(test_mask.sum())
                        )
                        continue

                    run_name = f"{model_type}_{target_name}"
                    calibrador = get_model(model_type)
                    logging.debug(_("Instancia de %s generada correctamente para target %s."), model_type, target_name)
                    target_X_train = X_train.loc[train_mask].reset_index(drop=True)
                    target_y_train = y_train.loc[train_mask, target_name].reset_index(drop=True)
                    target_groups_train = groups_train.loc[train_mask].reset_index(drop=True)
                    target_X_test = X_test.loc[test_mask].reset_index(drop=True)
                    target_y_test = y_test.loc[test_mask, target_name].reset_index(drop=True)
                    cv_splits = list(processor.get_cv_folds(target_groups_train, n_splits=cv_folds_num))

                    calibrador.train(
                        X=target_X_train,
                        y=target_y_train,
                        param_grid=param_grid,
                        cv_folds=cv_splits,
                        search_strategy=search_strategy,
                        n_iter=n_iter
                    )

                    logging.info(_("Entrenamiento finalizado para %s. Obteniendo predicciones ciegas..."), run_name)
                    predicciones = calibrador.predict(target_X_test)

                    logging.info(_("Generando reportes y metricas clinicas..."))
                    metrics = reporter.evaluate_and_save(y_true=target_y_test, y_pred=predicciones, model_name=run_name)
                    reporter.generate_scatter_plot(y_true=target_y_test, y_pred=predicciones, model_name=run_name)
                    reporter.generate_residuals_plot(y_true=target_y_test, y_pred=predicciones, model_name=run_name)
                    logging.info(_("Modelo %s evaluado con exito. RMSE: %.4f | R2: %.4f"), run_name, metrics["RMSE"], metrics["R2_Score"])
            
        except ValueError as e:
            logging.error(_("Error de configuracion para el modelo %s: %s"), model_type, str(e))
        except Exception as e:
            logging.error(_("Error critico e inesperado ejecutando el modelo %s: %s"), model_type, str(e))
            
    logging.info(_("Ejecucion global completada. Todos los modelos han sido procesados y reportados."))

if __name__ == "__main__":
    main()
