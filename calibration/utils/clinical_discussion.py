from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

from calibration.utils.reporting import _compute_metric_bundle


_TARGET_SUFFIXES = [
    'sdmt_paper_score',
    'tmt_paper_score_a',
    'tmt_paper_score_b',
    'tmt_paper_errors_a',
    'tmt_paper_errors_b',
]


def _to_1d_array(values: pd.Series | Any) -> np.ndarray:
    return np.asarray(values, dtype=float).flatten()


def _append_sheet_rows(output_excel: str, sheet_name: str, new_rows: pd.DataFrame) -> None:
    path = Path(output_excel)
    if path.exists():
        try:
            existing = pd.read_excel(path, sheet_name=sheet_name)
            combined = pd.concat([existing, new_rows], ignore_index=True)
        except ValueError:
            combined = new_rows
        with pd.ExcelWriter(path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            combined.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            new_rows.to_excel(writer, sheet_name=sheet_name, index=False)


def save_feature_audit(output_excel: str, test_type: str, feature_audit: Dict[str, Any]) -> None:
    rows = []
    for reason, features in feature_audit.get('removed_by_reason', {}).items():
        for feature in features:
            rows.append({
                'Test_Type': test_type,
                'Reason': reason,
                'Feature': str(feature),
            })

    for feature in feature_audit.get('selected_features_final', []):
        rows.append({
            'Test_Type': test_type,
            'Reason': 'selected_feature_final',
            'Feature': str(feature),
        })

    if rows:
        _append_sheet_rows(output_excel, 'FeatureAudit', pd.DataFrame(rows))


def save_stratified_metrics(
    output_excel: str,
    model_name: str,
    y_true: pd.Series,
    y_pred: Any,
    metadata: Optional[pd.DataFrame],
    prediction_interval_low: Optional[Any] = None,
    prediction_interval_high: Optional[Any] = None,
    interval_alpha: Optional[float] = None,
    min_group_size: int = 3,
) -> None:
    if metadata is None or metadata.empty:
        return

    df = metadata.copy().reset_index(drop=True)
    df['y_true'] = _to_1d_array(y_true)
    df['y_pred'] = _to_1d_array(y_pred)
    if prediction_interval_low is not None:
        df['pi_low'] = _to_1d_array(prediction_interval_low)
    if prediction_interval_high is not None:
        df['pi_high'] = _to_1d_array(prediction_interval_high)

    if 'sex_binary' in df.columns:
        df['sex_group'] = df['sex_binary'].map({0.0: 'Male', 1.0: 'Female'})

    stratifiers = []
    for column in ['clinical_group', 'age_band', 'sex_group']:
        if column in df.columns:
            stratifiers.append(column)

    rows = []
    for stratifier in stratifiers:
        valid_df = df[df[stratifier].notna()].copy()
        for subgroup, subgroup_df in valid_df.groupby(stratifier):
            if len(subgroup_df) < min_group_size:
                continue
            metrics = _compute_metric_bundle(
                y_true=subgroup_df['y_true'].to_numpy(dtype=float),
                y_pred=subgroup_df['y_pred'].to_numpy(dtype=float),
                interval_low=None if 'pi_low' not in subgroup_df else subgroup_df['pi_low'].to_numpy(dtype=float),
                interval_high=None if 'pi_high' not in subgroup_df else subgroup_df['pi_high'].to_numpy(dtype=float),
                interval_alpha=interval_alpha,
            )
            metrics.update({
                'Model': model_name,
                'Timestamp': datetime.now().isoformat(),
                'Stratifier': stratifier,
                'Subgroup': str(subgroup),
            })
            rows.append(metrics)

    if rows:
        _append_sheet_rows(output_excel, 'StratifiedMetrics', pd.DataFrame(rows))


def _infer_target(model_name: str) -> str:
    if pd.isna(model_name):
        return 'Unknown'
    model_name = str(model_name)
    for suffix in _TARGET_SUFFIXES:
        if model_name.endswith(suffix):
            return suffix
    return model_name


def _infer_assessment(model_name: str) -> str:
    if pd.isna(model_name):
        return 'Unknown'
    model_name = str(model_name)
    lowered = model_name.lower()
    if 'sdmt' in lowered:
        return 'SDMT'
    if 'tmt' in lowered:
        return 'TMT'
    return 'Unknown'


def _clinical_status(row: pd.Series) -> str:
    if row.get('R2_Score', np.nan) < 0 or row.get('CCC_Lin', np.nan) < 0.4:
        return 'Needs major improvement'
    if row.get('CCC_Lin', np.nan) < 0.7 or row.get('ICC_A1', np.nan) < 0.7:
        return 'Exploratory / moderate'
    return 'Promising for discussion'


def build_clinical_discussion_workbook(output_excel: str, source_workbooks: Iterable[str]) -> None:
    summary_frames = []
    stratified_frames = []
    feature_audit_frames = []

    for workbook in source_workbooks:
        path = Path(workbook)
        if not path.exists():
            continue
        try:
            summary = pd.read_excel(path)
            summary['SourceWorkbook'] = path.name
            summary_frames.append(summary)
        except Exception:
            pass
        try:
            stratified = pd.read_excel(path, sheet_name='StratifiedMetrics')
            stratified['SourceWorkbook'] = path.name
            stratified_frames.append(stratified)
        except Exception:
            pass
        try:
            feature_audit = pd.read_excel(path, sheet_name='FeatureAudit')
            feature_audit['SourceWorkbook'] = path.name
            feature_audit_frames.append(feature_audit)
        except Exception:
            pass

    if not summary_frames:
        return

    summary_df = pd.concat(summary_frames, ignore_index=True)
    if 'Model' not in summary_df.columns:
        return

    summary_df = summary_df[summary_df['Model'].notna()].copy()
    if summary_df.empty:
        return

    if 'Timestamp' in summary_df.columns:
        summary_df['Timestamp'] = pd.to_datetime(summary_df['Timestamp'], errors='coerce')
        summary_df = summary_df.sort_values('Timestamp').drop_duplicates(subset=['Model'], keep='last')

    summary_df['Assessment'] = summary_df['Model'].map(_infer_assessment)
    summary_df['Target'] = summary_df['Model'].map(_infer_target)
    summary_df['Clinical_Status'] = summary_df.apply(_clinical_status, axis=1)

    best_rows = []
    for (assessment, target), group_df in summary_df.groupby(['Assessment', 'Target']):
        best = group_df.sort_values(['RMSE', 'MAE', 'CCC_Lin'], ascending=[True, True, False]).iloc[0]
        best_rows.append(best)
    best_df = pd.DataFrame(best_rows)

    stratified_best = pd.DataFrame()
    if stratified_frames and not best_df.empty:
        stratified_df = pd.concat(stratified_frames, ignore_index=True)
        if 'Model' not in stratified_df.columns:
            stratified_df = pd.DataFrame()
        else:
            stratified_df = stratified_df[stratified_df['Model'].notna()].copy()
        if 'Timestamp' in stratified_df.columns:
            stratified_df['Timestamp'] = pd.to_datetime(stratified_df['Timestamp'], errors='coerce')
            stratified_df = stratified_df.sort_values('Timestamp').drop_duplicates(
                subset=['Model', 'Stratifier', 'Subgroup'],
                keep='last'
            )
        stratified_best = stratified_df[stratified_df['Model'].isin(best_df['Model'])].copy()

    feature_audit_df = pd.concat(feature_audit_frames, ignore_index=True) if feature_audit_frames else pd.DataFrame()

    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='GlobalMetrics', index=False)
        best_df.to_excel(writer, sheet_name='BestByTarget', index=False)
        if not stratified_best.empty:
            stratified_best.to_excel(writer, sheet_name='StratifiedBest', index=False)
        if not feature_audit_df.empty:
            feature_audit_df.to_excel(writer, sheet_name='FeatureAuditCombined', index=False)
