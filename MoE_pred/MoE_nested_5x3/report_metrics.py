"""Explicit pooled, per-protein and equal-protein macro regression reports."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, weightedtau


def validate_ids(frame):
    if frame.empty or 'ID' not in frame:
        raise ValueError('An input must have records and an ID column')
    ids = frame['ID']
    if ids.isna().any() or ids.astype(str).str.strip().eq('').any() or ids.duplicated().any():
        raise ValueError('Record IDs must be nonempty and unique')


def protein_groups(metadata, protein_col=None):
    """Use an explicit biological grouping, or source + complete accession."""
    validate_ids(metadata)
    if protein_col:
        if protein_col not in metadata:
            raise ValueError('Missing protein column: ' + protein_col)
        groups = metadata[protein_col]
    else:
        groups = metadata['ID'].astype(str).str.extract(
            r'^((?:rcsb_[A-Za-z0-9]{4}|uniprot_[A-Za-z0-9]+))_', expand=False
        ).str.upper()
    if groups.isna().any() or groups.astype(str).str.strip().eq('').any():
        raise ValueError('Cannot determine protein groups; provide --protein-col')
    return groups.astype(str).str.strip().to_numpy()


def regression_metrics(y, prediction):
    y, prediction = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    if y.ndim != 1 or y.shape != prediction.shape or not len(y):
        raise ValueError('Targets/predictions must be matching nonempty 1-D arrays')
    if not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError('Targets/predictions must be finite; no silent row dropping')
    error = prediction - y
    sst = np.sum((y - y.mean()) ** 2)
    constant_target = np.ptp(y) == 0
    constant_prediction = np.ptp(prediction) == 0
    reason = ('too_few_records' if len(y) < 2 else
              'constant_target' if constant_target else
              'constant_prediction' if constant_prediction else 'ok')
    valid = reason == 'ok'
    return {
        'n_records': len(y),
        'mae': float(np.mean(np.abs(error))),
        'rmse': float(np.sqrt(np.mean(error ** 2))),
        'r2': float(1 - np.sum(error ** 2) / sst) if len(y) >= 2 and sst > 0 else np.nan,
        'pearson': float(pearsonr(y, prediction)[0]) if valid else np.nan,
        'spearman': float(spearmanr(y, prediction)[0]) if valid else np.nan,
        # Preserve the original weightedtau statistic; it is not ordinary Kendall tau.
        'kendall_weighted': float(weightedtau(y, prediction)[0]) if valid else np.nan,
        'correlation_status': reason,
    }


def build_reports(predictions, min_group_n=10, fisher_eps=1e-7):
    """Each split is evaluated separately. Macro correlations use eligible proteins."""
    if min_group_n < 3 or not 0 < fisher_eps < .01:
        raise ValueError('min_group_n must be >= 3; fisher_eps must be in (0, .01)')
    required = {'ID', 'protein_id', 'split', 'target', 'prediction'}
    if not required.issubset(predictions.columns) or predictions.empty:
        raise ValueError('Predictions require nonempty ID, protein_id, split, target, prediction')
    for name in ('protein_id', 'split'):
        if predictions[name].isna().any() or predictions[name].astype(str).str.strip().eq('').any():
            raise ValueError(name + ' must be nonempty')
    pooled_rows, protein_rows, macro_rows = [], [], []
    for split, frame in predictions.groupby('split', sort=False):
        validate_ids(frame)
        pooled = regression_metrics(frame.target, frame.prediction)
        pooled_rows.append(dict(split=split, aggregation='pooled',
                                n_proteins=frame.protein_id.nunique(), **pooled))
        local = []
        for protein, group in frame.groupby('protein_id', sort=True):
            row = dict(split=split, protein_id=protein, aggregation='per_protein',
                       **regression_metrics(group.target, group.prediction))
            eligible = len(group) >= min_group_n and np.isfinite(row['pearson']) and np.isfinite(row['spearman'])
            row['macro_correlation_eligible'] = bool(eligible)
            row['macro_exclusion_reason'] = ('ok' if eligible else
                'below_min_group_n' if len(group) < min_group_n else row['correlation_status'])
            clipped = float(np.clip(row['pearson'], -1 + fisher_eps, 1 - fisher_eps))
            row['pearson_fisher_clipped'] = bool(eligible and clipped != row['pearson'])
            row['pearson_fisher_z'] = float(np.arctanh(clipped)) if eligible else np.nan
            local.append(row)
        per = pd.DataFrame(local)
        protein_rows.extend(local)
        eligible = per['macro_correlation_eligible']
        # Correlation names state the averaging method; no n-3 or record weighting.
        specifications = [
            ('mae', 'mae', per.index == per.index, 'arithmetic_mean'),
            ('rmse', 'rmse', per.index == per.index, 'arithmetic_mean'),
            ('r2', 'r2', per.r2.notna(), 'arithmetic_mean'),
            ('pearson_fisher', 'pearson_fisher_z', eligible, 'tanh_mean_atanh'),
            ('spearman_mean', 'spearman', eligible, 'arithmetic_mean'),
            ('kendall_weighted_mean', 'kendall_weighted', eligible & per.kendall_weighted.notna(), 'arithmetic_mean'),
        ]
        for metric, column, mask, averaging in specifications:
            chosen = per.loc[mask]
            value = float(chosen[column].mean()) if len(chosen) else np.nan
            if metric == 'pearson_fisher':
                value = float(np.tanh(value))
            macro_rows.append({
                'split': split, 'aggregation': 'protein_macro', 'metric': metric,
                'value': value, 'averaging': averaging, 'weighting': 'equal_protein',
                'n_proteins_total': len(per), 'n_proteins_used': len(chosen),
                'n_proteins_excluded': len(per) - len(chosen),
                'n_records_total': len(frame), 'n_records_used': int(chosen.n_records.sum()),
                'n_fisher_clipped': int(chosen.pearson_fisher_clipped.sum()) if metric == 'pearson_fisher' else 0,
                'min_group_n': min_group_n, 'fisher_eps': fisher_eps,
                'status': 'ok' if len(chosen) else 'no_eligible_proteins',
            })
    return pd.DataFrame(pooled_rows), pd.DataFrame(protein_rows), pd.DataFrame(macro_rows)


def write_reports(predictions, output_dir, min_group_n=10, fisher_eps=1e-7, config=None):
    reports = build_reports(predictions, min_group_n, fisher_eps)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / 'predictions.csv', index=False)
    for name, table in zip(('pooled_metrics', 'per_protein_metrics', 'macro_metrics'), reports):
        table.to_csv(output_dir / (name + '.csv'), index=False)
    settings = dict(config or {})
    settings.update(min_group_n=min_group_n, fisher_eps=fisher_eps,
                    pearson_macro='tanh(mean(atanh(clip(r))))',
                    macro_weighting='equal_protein', spearman_macro='arithmetic_mean',
                    undefined_metrics='NaN; never replaced with zero',
                    correlation_support='at least min_group_n records; nonconstant target and prediction')
    (output_dir / 'report_config.json').write_text(
        json.dumps(settings, ensure_ascii=False, indent=2), encoding='utf-8')
    return reports


def align_predictions(pred_path, meta_path, prediction_col='prediction',
                      prediction_target_col=None, protein_col=None, split='test'):
    pred = pd.read_csv(pred_path, dtype={'ID': str})
    meta = pd.read_csv(meta_path, dtype={'ID': str})
    validate_ids(pred)
    validate_ids(meta)
    if set(pred.ID) != set(meta.ID):
        raise ValueError('Prediction and metadata ID sets must match exactly')
    pred = pred.set_index('ID').loc[meta.ID]
    if prediction_target_col:
        if not np.allclose(pred[prediction_target_col].to_numpy(dtype=float),
                           meta.target.to_numpy(dtype=float), rtol=1e-5, atol=1e-6):
            raise ValueError('Targets differ after ID alignment')
    return pd.DataFrame({
        'ID': meta.ID.to_numpy(), 'protein_id': protein_groups(meta, protein_col),
        'split': split, 'target': meta.target.to_numpy(dtype=float),
        'prediction': pred[prediction_col].to_numpy(dtype=float),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--prediction-col', default='prediction')
    parser.add_argument('--prediction-target-col')
    parser.add_argument('--protein-col')
    parser.add_argument('--split', default='test')
    parser.add_argument('--min-group-n', type=int, default=10)
    parser.add_argument('--fisher-eps', type=float, default=1e-7)
    parser.add_argument('-o', '--output-dir', required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        parser.error('Output directory must be empty or new')
    frame = align_predictions(args.predictions, args.metadata, args.prediction_col,
                              args.prediction_target_col, args.protein_col, args.split)
    pooled, _, macro = write_reports(frame, output, args.min_group_n, args.fisher_eps, vars(args))
    print(pooled.to_string(index=False))
    print(macro[['split', 'metric', 'value', 'n_proteins_used', 'n_proteins_total']].to_string(index=False))


if __name__ == '__main__':
    main()
