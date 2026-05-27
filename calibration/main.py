import argparse
import yaml
import logging
import gettext
import os
import pandas as pd
from typing import Any, Dict, List
from calibracion_cognitiva.models.algoritmos import get_model
from calibracion_cognitiva.data.loader import DataProcessor
from calibracion_cognitiva.utils.reporting import ReportGenerator

default_lang = os.environ.get('LANG', 'es').split('_')[0]
temp_translation = gettext.translation('messages', localedir='locales', languages=[default_lang], fallback=True)
_ = temp_translation.gettext

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
    Thirdly, it reconfigures the internationalization (i18n) engine based on the
    language specified in the configuration file.
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
    args = parser.parse_args()

    setup_logging(args.verbose)
    
    try:
        with open(args.config, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        logging.critical(_("Fallo critico al leer el archivo de configuracion YAML: %s"), str(e))
        return

    lang = config.get("language", "es")
    translation = gettext.translation('messages', localedir='locales', languages=[lang], fallback=True)
    translation.install()
    
    global _ 
    _ = translation.gettext

    logging.info(_("Iniciando el proceso de calibracion con configuracion: %s"), args.config)
    
    # Configuración de Rutas de Salida y Generador de Reportes
    output_config = config.get("output", {})
    output_excel = output_config.get("excel_report", "resultados.xlsx")
    output_html = output_config.get("html_plot", "graficos.html")
    reporter = ReportGenerator(output_excel=output_excel, output_html=output_html)
    
    # Procesamiento de Datos
    data_config = config.get("data", {})
    db_uri = data_config.get("db_uri", "postgresql://user:pass@localhost/dbname")
    excel_path = data_config.get("excel_path", "datos_papel.xlsx")
    test_type = data_config.get("test_type", "sdmt")
    
    logging.info(_("Procesando extraccion de datos para el test objetivo: %s"), test_type.upper())
    processor = DataProcessor(db_uri=db_uri, excel_path=excel_path)
    df_merged = processor.load_and_merge(test_type)
    
    X_train, X_test, y_train, y_test, groups_train = processor.prepare_splits(df_merged, test_type)
    
    models_config: List[Dict[str, Any]] = config.get("models", [{"type": "linear", "cv_folds": 3}])
    
    logging.info(_("Se han detectado %d modelo(s) en la cola de ejecucion."), len(models_config))

    # Bucle de Entrenamiento y Evaluación
    for i, model_cfg in enumerate(models_config, 1):
        model_type = model_cfg.get("type", "linear")
        cv_folds_num = model_cfg.get("cv_folds", 3)
        search_strategy = model_cfg.get("search_strategy", "grid")
        n_iter = model_cfg.get("n_iter", 10)
        param_grid = model_cfg.get("param_grid", None)
        
        logging.info(_("--- [Iteracion %d/%d] Iniciando pipeline para modelo: %s ---"), i, len(models_config), model_type.upper())
        
        cv_splits = list(processor.get_cv_folds(groups_train, n_splits=cv_folds_num))
        
        try:
            calibrador = get_model(model_type)
            logging.debug(_("Instancia de %s generada correctamente. Iniciando ajuste."), model_type)
            
            calibrador.train(
                X=X_train, 
                y=y_train, 
                param_grid=param_grid, 
                cv_folds=cv_splits, 
                search_strategy=search_strategy, 
                n_iter=n_iter
            )
            
            logging.info(_("Entrenamiento finalizado para %s. Obteniendo predicciones ciegas..."), model_type)
            predicciones = calibrador.predict(X_test)
            
            # Generación de Reportes y Gráficos
            logging.info(_("Generando reportes y metricas clinicas..."))
            metrics = reporter.evaluate_and_save(y_true=y_test, y_pred=predicciones, model_name=model_type)
            reporter.generate_scatter_plot(y_true=y_test, y_pred=predicciones, model_name=model_type)
            
            logging.info(_("Modelo %s evaluado con exito. RMSE: %.4f | R2: %.4f"), model_type, metrics["RMSE"], metrics["R2_Score"])
            
        except ValueError as e:
            logging.error(_("Error de configuracion para el modelo %s: %s"), model_type, str(e))
        except Exception as e:
            logging.error(_("Error critico e inesperado ejecutando el modelo %s: %s"), model_type, str(e))
            
    logging.info(_("Ejecucion global completada. Todos los modelos han sido procesados y reportados."))

if __name__ == "__main__":
    main()
