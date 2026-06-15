import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)

from calibration.i18n import get_translator

logger = logging.getLogger(__name__)
_ = get_translator()

_BOOTSTRAP_KEYS = [
    'RMSE',
    'MAE',
    'MedianAE',
    'R2_Score',
    'CCC_Lin',
    'ICC_A1',
    'BlandAltman_Bias',
    'BlandAltman_LoA_Low',
    'BlandAltman_LoA_High',
    'PI_Coverage',
    'PI_Mean_Width',
]


def _to_1d_array(values: pd.Series | Any) -> np.ndarray:
    """Converts a vector-like input into a flat numpy array."""
    return np.asarray(values, dtype=float).flatten()


def _lin_concordance_ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Lin's concordance correlation coefficient."""
    if len(y_true) < 2:
        return float('nan')

    mean_true = float(np.mean(y_true))
    mean_pred = float(np.mean(y_pred))
    var_true = float(np.var(y_true, ddof=1))
    var_pred = float(np.var(y_pred, ddof=1))
    covariance = float(np.cov(y_true, y_pred, ddof=1)[0, 1])
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    if denominator == 0:
        return float('nan')
    return float((2.0 * covariance) / denominator)


def _icc_absolute_agreement(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes ICC(A,1) for absolute agreement using two raters."""
    n_subjects = len(y_true)
    n_raters = 2
    if n_subjects < 2:
        return float('nan')

    ratings = np.column_stack([y_true, y_pred])
    mean_subject = ratings.mean(axis=1)
    mean_rater = ratings.mean(axis=0)
    grand_mean = ratings.mean()

    ss_subject = n_raters * np.sum((mean_subject - grand_mean) ** 2)
    ss_rater = n_subjects * np.sum((mean_rater - grand_mean) ** 2)
    ss_error = np.sum((ratings - mean_subject[:, None] - mean_rater + grand_mean) ** 2)

    ms_subject = ss_subject / (n_subjects - 1)
    ms_rater = ss_rater / (n_raters - 1)
    ms_error = ss_error / ((n_subjects - 1) * (n_raters - 1))

    denominator = ms_subject + (n_raters - 1) * ms_error + (n_raters * (ms_rater - ms_error) / n_subjects)
    if denominator == 0:
        return float('nan')

    return float((ms_subject - ms_error) / denominator)


def _bland_altman_stats(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes Bland-Altman summary statistics."""
    differences = y_pred - y_true
    bias = float(np.mean(differences))
    sd_diff = float(np.std(differences, ddof=1)) if len(differences) > 1 else float('nan')
    loa_low = float(bias - 1.96 * sd_diff) if not np.isnan(sd_diff) else float('nan')
    loa_high = float(bias + 1.96 * sd_diff) if not np.isnan(sd_diff) else float('nan')
    return {
        'BlandAltman_Bias': bias,
        'BlandAltman_SD': sd_diff,
        'BlandAltman_LoA_Low': loa_low,
        'BlandAltman_LoA_High': loa_high,
    }


def _calibration_line(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Fits a simple calibration line y_true ~ y_pred."""
    if len(y_true) < 2 or np.allclose(y_pred, y_pred[0]):
        return {
            'Calibration_Slope': float('nan'),
            'Calibration_Intercept': float('nan'),
        }

    slope, intercept = np.polyfit(y_pred, y_true, deg=1)
    return {
        'Calibration_Slope': float(slope),
        'Calibration_Intercept': float(intercept),
    }


def _prediction_interval_metrics(
    y_true: np.ndarray,
    interval_low: Optional[np.ndarray],
    interval_high: Optional[np.ndarray],
    interval_alpha: Optional[float],
) -> Dict[str, float]:
    """Computes summary metrics for prediction intervals."""
    if interval_low is None or interval_high is None:
        return {
            'PI_Alpha': float('nan'),
            'PI_Coverage': float('nan'),
            'PI_Mean_Width': float('nan'),
            'PI_Median_Width': float('nan'),
            'PI_Miscoverage': float('nan'),
        }

    widths = interval_high - interval_low
    covered = (y_true >= interval_low) & (y_true <= interval_high)
    coverage = float(np.mean(covered)) if len(covered) else float('nan')
    return {
        'PI_Alpha': float(interval_alpha) if interval_alpha is not None else float('nan'),
        'PI_Coverage': coverage,
        'PI_Mean_Width': float(np.mean(widths)),
        'PI_Median_Width': float(np.median(widths)),
        'PI_Miscoverage': float(1.0 - coverage) if not np.isnan(coverage) else float('nan'),
    }


def _compute_metric_bundle(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    interval_low: Optional[np.ndarray] = None,
    interval_high: Optional[np.ndarray] = None,
    interval_alpha: Optional[float] = None,
) -> Dict[str, float]:
    """Computes the full metric bundle used in reports and bootstrap."""
    rmse = float(root_mean_squared_error(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    try:
        mape = float(mean_absolute_percentage_error(y_true, y_pred))
    except Exception:
        mape = float('nan')

    residuals = y_true - y_pred
    residual_std = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else float('nan')
    residual_mean = float(np.mean(residuals))
    median_ae = float(np.median(np.abs(residuals)))
    ccc = _lin_concordance_ccc(y_true, y_pred)
    icc = _icc_absolute_agreement(y_true, y_pred)
    bland_altman = _bland_altman_stats(y_true, y_pred)
    calibration = _calibration_line(y_true, y_pred)
    interval_metrics = _prediction_interval_metrics(y_true, interval_low, interval_high, interval_alpha)

    metrics = {
        'N_Samples': len(y_true),
        'RMSE': rmse,
        'MAE': mae,
        'MedianAE': median_ae,
        'MSE': mse,
        'R2_Score': r2,
        'MAPE': mape,
        'Residual_Mean': residual_mean,
        'Residual_Std': residual_std,
        'CCC_Lin': ccc,
        'ICC_A1': icc,
        'Min_True': float(y_true.min()),
        'Max_True': float(y_true.max()),
        'Mean_True': float(y_true.mean()),
        'Std_True': float(y_true.std(ddof=1)) if len(y_true) > 1 else float('nan'),
        'Min_Pred': float(y_pred.min()),
        'Max_Pred': float(y_pred.max()),
        'Mean_Pred': float(y_pred.mean()),
        'Std_Pred': float(y_pred.std(ddof=1)) if len(y_pred) > 1 else float('nan'),
    }
    metrics.update(bland_altman)
    metrics.update(calibration)
    metrics.update(interval_metrics)
    return metrics


def _grouped_bootstrap_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: Optional[np.ndarray],
    n_bootstrap: int,
    confidence: float,
    random_state: int,
    interval_low: Optional[np.ndarray] = None,
    interval_high: Optional[np.ndarray] = None,
    interval_alpha: Optional[float] = None,
) -> Dict[str, float]:
    """Computes cluster-bootstrap confidence intervals by patient/group."""
    summary = {
        'Bootstrap_Iterations_Requested': int(n_bootstrap),
        'Bootstrap_Iterations_Valid': 0,
    }
    if groups is None or n_bootstrap <= 1:
        return summary

    group_array = np.asarray(groups).flatten()
    if len(group_array) != len(y_true):
        raise ValueError(_('Longitud de groups y y_true no coinciden'))

    unique_groups = pd.unique(group_array)
    if len(unique_groups) < 2:
        return summary

    alpha_tail = max(0.0, min((1.0 - confidence) / 2.0, 0.5))
    rng = np.random.default_rng(random_state)
    collected: Dict[str, list] = {key: [] for key in _BOOTSTRAP_KEYS}

    for _ in range(n_bootstrap):
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        sampled_indices = []
        for group in sampled_groups:
            group_idx = np.flatnonzero(group_array == group)
            if group_idx.size:
                sampled_indices.append(group_idx)
        if not sampled_indices:
            continue

        sample_idx = np.concatenate(sampled_indices)
        if sample_idx.size < 2:
            continue

        sample_metrics = _compute_metric_bundle(
            y_true=y_true[sample_idx],
            y_pred=y_pred[sample_idx],
            interval_low=None if interval_low is None else interval_low[sample_idx],
            interval_high=None if interval_high is None else interval_high[sample_idx],
            interval_alpha=interval_alpha,
        )
        for key in _BOOTSTRAP_KEYS:
            value = sample_metrics.get(key, float('nan'))
            if np.isfinite(value):
                collected[key].append(float(value))
        summary['Bootstrap_Iterations_Valid'] += 1

    for key, values in collected.items():
        if not values:
            continue
        values_array = np.asarray(values, dtype=float)
        summary[f'{key}_CI_Low'] = float(np.quantile(values_array, alpha_tail))
        summary[f'{key}_CI_High'] = float(np.quantile(values_array, 1.0 - alpha_tail))
        summary[f'{key}_Bootstrap_Median'] = float(np.quantile(values_array, 0.5))

    return summary


class ReportGenerator:
    """Utility class for generating performance reports and interactive visualizations."""

    def __init__(self, output_excel: str, output_html: str) -> None:
        self.output_excel = output_excel
        self.output_html = output_html
        self.metrics_history: Dict[str, Any] = {}

        Path(self.output_excel).parent.mkdir(parents=True, exist_ok=True)
        Path(self.output_html).parent.mkdir(parents=True, exist_ok=True)

        logger.info(
            _('Generador de reportes inicializado. Excel: %s | HTML: %s'),
            self.output_excel,
            self.output_html,
        )

    def evaluate_and_save(
        self,
        y_true: pd.Series,
        y_pred: Any,
        model_name: str,
        groups: Optional[pd.Series | Any] = None,
        prediction_interval_low: Optional[Any] = None,
        prediction_interval_high: Optional[Any] = None,
        interval_alpha: Optional[float] = None,
        bootstrap_iterations: int = 0,
        bootstrap_confidence: float = 0.95,
        bootstrap_random_state: int = 42,
    ) -> Dict[str, float]:
        y_true_array = _to_1d_array(y_true)
        y_pred_array = _to_1d_array(y_pred)
        interval_low_array = None if prediction_interval_low is None else _to_1d_array(prediction_interval_low)
        interval_high_array = None if prediction_interval_high is None else _to_1d_array(prediction_interval_high)
        group_array = None if groups is None else np.asarray(groups).flatten()

        if len(y_true_array) != len(y_pred_array):
            raise ValueError(
                _('Longitud de y_true (%d) y y_pred (%d) no coinciden')
                % (len(y_true_array), len(y_pred_array))
            )
        if interval_low_array is not None and len(interval_low_array) != len(y_true_array):
            raise ValueError(_('Longitud de prediction_interval_low y y_true no coinciden'))
        if interval_high_array is not None and len(interval_high_array) != len(y_true_array):
            raise ValueError(_('Longitud de prediction_interval_high y y_true no coinciden'))
        if group_array is not None and len(group_array) != len(y_true_array):
            raise ValueError(_('Longitud de groups y y_true no coinciden'))
        if len(y_true_array) < 2:
            raise ValueError(_('Se requieren al menos 2 muestras para calcular métricas'))

        logger.info(_('▶️ Calculando métricas de rendimiento para modelo: %s'), model_name)

        metrics = {
            'Model': model_name,
            'Timestamp': datetime.now().isoformat(),
        }
        metrics.update(
            _compute_metric_bundle(
                y_true=y_true_array,
                y_pred=y_pred_array,
                interval_low=interval_low_array,
                interval_high=interval_high_array,
                interval_alpha=interval_alpha,
            )
        )
        metrics.update(
            _grouped_bootstrap_summary(
                y_true=y_true_array,
                y_pred=y_pred_array,
                groups=group_array,
                n_bootstrap=bootstrap_iterations,
                confidence=bootstrap_confidence,
                random_state=bootstrap_random_state,
                interval_low=interval_low_array,
                interval_high=interval_high_array,
                interval_alpha=interval_alpha,
            )
        )

        logger.info(_('✓ Métricas de %s:'), model_name)
        logger.info(_('  • RMSE: %.4f | MAE: %.4f | R²: %.4f'), metrics['RMSE'], metrics['MAE'], metrics['R2_Score'])
        logger.info(_('  • CCC: %.4f | ICC(A,1): %.4f'), metrics['CCC_Lin'], metrics['ICC_A1'])
        logger.info(
            _('  • Bland-Altman bias: %.4f | LoA [%.4f, %.4f]'),
            metrics['BlandAltman_Bias'],
            metrics['BlandAltman_LoA_Low'],
            metrics['BlandAltman_LoA_High'],
        )
        if np.isfinite(metrics.get('PI_Coverage', float('nan'))):
            logger.info(
                _('  • Intervalos predictivos: cobertura %.4f | anchura media %.4f'),
                metrics['PI_Coverage'],
                metrics['PI_Mean_Width'],
            )

        df_new = pd.DataFrame([metrics])

        output_path = Path(self.output_excel)
        metrics_sheet = 'Metrics'
        if output_path.exists():
            try:
                df_existing = pd.read_excel(output_path, sheet_name=metrics_sheet)
                df_final = pd.concat([df_existing, df_new], ignore_index=True)
                logger.debug(_('Tabla de reportes existente encontrada. Agregando fila...'))
            except Exception:
                df_final = df_new

            with pd.ExcelWriter(output_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                df_final.to_excel(writer, sheet_name=metrics_sheet, index=False)
        else:
            logger.debug(_('Creando nueva tabla de reportes'))
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                df_new.to_excel(writer, sheet_name=metrics_sheet, index=False)
            df_final = df_new

        logger.info(_('✓ Resultados guardados en: %s'), self.output_excel)

        self.metrics_history[model_name] = metrics
        return metrics

    def generate_scatter_plot(
        self,
        y_true: pd.Series,
        y_pred: Any,
        model_name: str,
    ) -> None:
        logger.info(_('▶️ Generando visualización interactiva Plotly para %s'), model_name)

        y_true_array = _to_1d_array(y_true)
        y_pred_array = _to_1d_array(y_pred)
        r2 = r2_score(y_true_array, y_pred_array)
        rmse = root_mean_squared_error(y_true_array, y_pred_array)
        mae = mean_absolute_error(y_true_array, y_pred_array)
        ccc = _lin_concordance_ccc(y_true_array, y_pred_array)
        icc = _icc_absolute_agreement(y_true_array, y_pred_array)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=y_true_array,
                y=y_pred_array,
                mode='markers',
                name=_('Predicciones'),
                marker=dict(
                    size=8,
                    opacity=0.7,
                    color='rgba(31, 119, 180, 0.8)',
                    line=dict(width=1, color='rgba(31, 119, 180, 1)'),
                ),
                hovertemplate='<b>Real: %{x:.2f}</b><br>Predicho: %{y:.2f}<extra></extra>',
            )
        )

        min_val = min(y_true_array.min(), y_pred_array.min())
        max_val = max(y_true_array.max(), y_pred_array.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name=_('Predicción Perfecta'),
                line=dict(dash='dash', color='red', width=2),
            )
        )

        fig.update_layout(
            title=dict(
                text=_('Calibración de %s: Digital vs Papel') % model_name.upper(),
                x=0.5,
                xanchor='center',
            ),
            xaxis_title=_('Puntuación Real (Test en Papel)'),
            yaxis_title=_('Puntuación Predicha (Test Digital)'),
            template='plotly_white',
            hovermode='closest',
            annotations=[
                dict(
                    text=f"R² = {r2:.3f}<br>RMSE = {rmse:.3f}<br>MAE = {mae:.3f}<br>CCC = {ccc:.3f}<br>ICC = {icc:.3f}<br>N = {len(y_true_array)}",
                    xref='paper',
                    yref='paper',
                    x=0.02,
                    y=0.98,
                    showarrow=False,
                    bgcolor='rgba(255, 255, 255, 0.85)',
                    bordercolor='black',
                    borderwidth=1,
                )
            ],
        )
        fig.update_xaxes(zeroline=False, showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(zeroline=False, showgrid=True, gridwidth=1, gridcolor='LightGray')

        plot_path = str(Path(self.output_html).parent / f'calibration_{model_name}.html')
        fig.write_html(plot_path)
        logger.info(_('✓ Gráfico interactivo guardado: %s'), plot_path)

    def generate_residuals_plot(
        self,
        y_true: pd.Series,
        y_pred: Any,
        model_name: str,
    ) -> None:
        logger.info(_('Generando gráfico de residuos para %s'), model_name)

        y_true_array = _to_1d_array(y_true)
        y_pred_array = _to_1d_array(y_pred)
        residuals = y_true_array - y_pred_array

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(_('Residuos vs Predicción'), _('Distribución de Residuos')),
        )
        fig.add_trace(
            go.Scatter(
                x=y_pred_array,
                y=residuals,
                mode='markers',
                name=_('Residuos'),
                marker=dict(size=6, color='blue', opacity=0.6),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[float(y_pred_array.min()), float(y_pred_array.max())],
                y=[0.0, 0.0],
                mode='lines',
                line=dict(dash='dash', color='red'),
                name=_('Línea base'),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Histogram(
                x=residuals,
                nbinsx=30,
                name=_('Residuos'),
                marker=dict(color='green', opacity=0.6),
            ),
            row=1,
            col=2,
        )

        fig.update_xaxes(title_text=_('Predicción'), row=1, col=1)
        fig.update_yaxes(title_text=_('Residuo'), row=1, col=1)
        fig.update_xaxes(title_text=_('Residuo'), row=1, col=2)
        fig.update_yaxes(title_text=_('Frecuencia'), row=1, col=2)
        fig.update_layout(
            title_text=_('Diagnóstico de Residuos: %s') % model_name.upper(),
            showlegend=False,
            height=400,
        )

        plot_path = str(Path(self.output_html).parent / f'residuals_{model_name}.html')
        fig.write_html(plot_path)
        logger.info(_('✓ Gráfico de residuos guardado: %s'), plot_path)

    def generate_bland_altman_plot(
        self,
        y_true: pd.Series,
        y_pred: Any,
        model_name: str,
    ) -> None:
        """Creates a Bland-Altman agreement plot."""
        logger.info(_('Generando gráfico de Bland-Altman para %s'), model_name)

        y_true_array = _to_1d_array(y_true)
        y_pred_array = _to_1d_array(y_pred)
        averages = (y_true_array + y_pred_array) / 2.0
        differences = y_pred_array - y_true_array
        stats = _bland_altman_stats(y_true_array, y_pred_array)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=averages,
                y=differences,
                mode='markers',
                name=_('Diferencias'),
                marker=dict(size=7, color='rgba(44, 160, 44, 0.75)'),
                hovertemplate='Media: %{x:.2f}<br>Diferencia: %{y:.2f}<extra></extra>',
            )
        )

        for y_value, label, color in [
            (stats['BlandAltman_Bias'], _('Sesgo medio'), 'red'),
            (stats['BlandAltman_LoA_Low'], _('Límite inferior'), 'orange'),
            (stats['BlandAltman_LoA_High'], _('Límite superior'), 'orange'),
        ]:
            fig.add_trace(
                go.Scatter(
                    x=[float(averages.min()), float(averages.max())],
                    y=[y_value, y_value],
                    mode='lines',
                    name=label,
                    line=dict(color=color, dash='dash'),
                )
            )

        fig.update_layout(
            title=_('Bland-Altman de %s') % model_name.upper(),
            xaxis_title=_('Media de papel y predicción'),
            yaxis_title=_('Predicción - Papel'),
            template='plotly_white',
            showlegend=True,
        )

        plot_path = str(Path(self.output_html).parent / f'bland_altman_{model_name}.html')
        fig.write_html(plot_path)
        logger.info(_('✓ Gráfico Bland-Altman guardado: %s'), plot_path)
