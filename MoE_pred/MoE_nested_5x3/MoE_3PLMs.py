"""Grouped 5 x 3 nested CV with TPE screening and repeated candidate review."""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

import moe_core as core
from report_metrics import build_reports, protein_groups, validate_ids, write_reports
from search_space import load_space, sample_space
from routing_reports import evaluate_with_routing, write_routing_reports, combine_outer_reports


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def append_row(path, row):
    path = Path(path)
    exists = path.exists()
    with path.open('a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def prediction_frame(ids, proteins, target, prediction, split):
    return pd.DataFrame({'ID': np.asarray(ids), 'protein_id': np.asarray(proteins),
                         'target': np.asarray(target), 'prediction': prediction, 'split': split})


def validation_metrics(target, prediction, proteins, scoring, min_group_n, spearman_weight=0.5):
    # The reporting implementation is shared with selection: no alternative macro definition.
    frame = prediction_frame(np.arange(len(target)).astype(str), proteins, target, prediction, 'validation')
    pooled, _, macro = build_reports(frame, min_group_n=min_group_n)
    if scoring.startswith('pooled_'):
        rho, r = float(pooled.spearman.iloc[0]), float(pooled.pearson.iloc[0])
    else:
        rho = float(macro.loc[macro.metric == 'spearman_mean', 'value'].iloc[0])
        r = float(macro.loc[macro.metric == 'pearson_fisher', 'value'].iloc[0])
    weight = 1.0 if scoring.endswith('_spearman') else 0.0 if scoring.endswith('_pearson') else spearman_weight
    score = rho if weight == 1 else r if weight == 0 else weight * rho + (1 - weight) * r
    return dict(spearman=rho, pearson=r, score=score)


def score_prediction(target, prediction, proteins, scoring, min_group_n, spearman_weight=0.5):
    return validation_metrics(target, prediction, proteins, scoring, min_group_n, spearman_weight)['score']


def load_data(paths, meta_path, group_col=None, protein_col=None):
    features, target, groups, ids = core.load_three_feats(paths, meta_path, group_col)
    metadata = pd.read_csv(meta_path, dtype={'ID': str, **({protein_col: str} if protein_col else {})})
    validate_ids(metadata)
    proteins = pd.Series(protein_groups(metadata, protein_col))
    # A protein must never be spread over separate CV groups (including explicit clusters).
    mapping = pd.DataFrame({'protein': proteins, 'group': groups})
    if mapping.groupby('protein')['group'].nunique().gt(1).any():
        raise ValueError('同一蛋白质被分配至不同 CV 分组，请检查 --group-col / --protein-col')
    return features, target, groups, proteins, ids


def subset(data, indices):
    features, target, groups, proteins, ids = data
    return ([x.iloc[indices].reset_index(drop=True) for x in features],
            *[x.iloc[indices].reset_index(drop=True) for x in (target, groups, proteins, ids)])


def validate_plans(data, splits, scoring, min_group_n):
    _, target, _, proteins, _ = data
    for _, val in splits:
        y = target.iloc[val].to_numpy()
        # Perfect predictions test whether the metric is defined on the targets at all.
        if not np.isfinite(score_prediction(y, y, proteins.iloc[val], scoring, min_group_n)):
            raise ValueError('验证折的选参指标无定义：检查标签变化、蛋白质数量及 min-group-n；不自动更换指标')


def suggest_params(trial, args):
    return dict(sample_space(trial, args.search_space_config), epochs=args.epochs, patience=args.patience)


def evaluate_candidate(data, splits, params, seeds, args, device, directory, phase, candidate, trial=None):
    """Only the current training subset is accepted; external/outer targets are absent."""
    features, target, _, proteins, ids = data
    seed_scores, best_epochs = [], []
    for seed in seeds:
        fold_scores = []
        seed_predictions = np.full(len(target), np.nan)
        for fold, (train_idx, val_idx) in enumerate(splits, 1):
            actual_seed = seed + fold * 10000
            core.seed_everything(actual_seed)
            fit, validation = core.scaled_datasets(features, target, train_idx, val_idx)
            model = core.make_model([x.shape[1] for x in features], params, device)

            def monitor(truth, prediction):
                return validation_metrics(truth, prediction, proteins.iloc[val_idx],
                                          args.scoring, args.min_group_n, args.spearman_weight)

            def callback(epoch, prediction, truth, validation):
                append_row(directory / 'training_history.csv', dict(phase=phase, candidate=candidate,
                    seed=seed, inner_fold=fold, epoch=epoch, val_spearman=validation['spearman'],
                    val_pearson=validation['pearson'], val_score=validation['score']))
                # Compare only the same first fold across trials. Later folds and review
                # are never pruned; startup trials and epoch warmup protect slow learners.
                if trial is not None and fold == 1:
                    value = validation['score']
                    if np.isfinite(value):
                        trial.report(value, epoch)
                        if epoch >= args.prune_warmup and trial.should_prune():
                            import optuna
                            raise optuna.TrialPruned(f'first-fold screening pruned at epoch {epoch}')

            try:
                best_epoch = core.train_model(model, fit, params, device, val_dataset=validation,
                                              callback=callback, validation_monitor=monitor)
            except core.UndefinedCorrelationError:
                # An undefined fold invalidates the whole candidate, with no MAE fallback.
                append_row(directory / 'invalid_candidates.csv', dict(phase=phase, candidate=candidate,
                    seed=seed, inner_fold=fold, reason='no_defined_validation_correlation'))
                return [float('nan')] * len(seeds), []
            pred, truth = core.evaluate_model(model, DataLoader(validation, batch_size=params['batch_size']), device)
            final_metrics = monitor(truth, pred)
            score = final_metrics['score']
            best_epochs.append(best_epoch)
            fold_scores.append(score)
            seed_predictions[val_idx] = pred
            append_row(directory / 'inner_folds.csv', dict(phase=phase, candidate=candidate, seed=seed,
                actual_seed=actual_seed, inner_fold=fold, best_epoch=best_epoch, score=score,
                val_spearman=final_metrics['spearman'], val_pearson=final_metrics['pearson'],
                params=json.dumps(params, sort_keys=True)))
            del model
        # All three folds must have defined scores; do not silently drop difficult folds.
        mean_score = float(np.mean(fold_scores))
        seed_scores.append(mean_score)
        if phase == 'review':
            run_dir = directory / f'review_{candidate:03d}' / f'seed_{seed}'
            write_reports(prediction_frame(ids, proteins, target, seed_predictions, 'inner_oof'),
                          run_dir, args.min_group_n,
                          config={'selection_score': 'unweighted mean of three fold scores', 'seed': seed})
    return seed_scores, best_epochs


def staged_search(data, args, device, directory, sampler_seed):
    import optuna
    directory.mkdir(parents=True, exist_ok=True)
    splits = core.grouped_splits(data[2], 3)
    validate_plans(data, splits, args.scoring, args.min_group_n)
    split_rows = []
    for fold, (fit, val) in enumerate(splits, 1):
        for role, indices in [('train', fit), ('validation', val)]:
            split_rows.append(pd.DataFrame({'inner_fold': fold, 'role': role,
                'ID': data[4].iloc[indices].to_numpy(), 'cv_group': data[2].iloc[indices].to_numpy(),
                'protein_id': data[3].iloc[indices].to_numpy()}))
    pd.concat(split_rows).to_csv(directory / 'splits.csv', index=False)
    study = optuna.create_study(direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=sampler_seed, n_startup_trials=args.startup_trials),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=args.startup_trials,
            n_warmup_steps=args.prune_warmup, interval_steps=5, n_min_trials=3))

    def objective(trial):
        params = suggest_params(trial, args)
        trial.set_user_attr('model_params', params)
        scores, _ = evaluate_candidate(data, splits, params, [args.screen_seed], args, device,
                                       directory, 'screen', trial.number, trial)
        if not np.isfinite(scores[0]):
            raise optuna.TrialPruned('undefined validation correlation')
        return scores[0]

    def save_trials(study, trial):
        study.trials_dataframe().to_csv(directory / 'screen_trials.csv', index=False)

    study.optimize(objective, n_trials=args.trials, callbacks=[save_trials])
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and np.isfinite(t.value)]
    completed.sort(key=lambda t: (-t.value, t.number))
    shortlisted, seen = [], set()
    for trial in completed:
        params = trial.user_attrs['model_params']
        key = json.dumps(params, sort_keys=True)
        if key not in seen:
            shortlisted.append(params)
            seen.add(key)
        if len(shortlisted) == args.shortlist:
            break
    if not shortlisted:
        raise ValueError('没有完成且指标有效的初筛候选，请检查训练记录')
    dump_json(directory / 'shortlist.json', shortlisted)
    # Screen scores never enter this ranking: only new seeds, full folds, no pruning.
    rows, best = [], None
    for base in shortlisted:
        for experts in (args.review_experts or [base['num_experts']]):
            params = dict(base, num_experts=experts)
            candidate = len(rows) + 1
            scores, epochs = evaluate_candidate(data, splits, params, args.review_seeds,
                args, device, directory, 'review', candidate)
            mean = float(np.mean(scores))
            std = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
            row = dict(candidate=candidate, mean_score=mean, std_score=std,
                       seed_scores=json.dumps(scores), params=json.dumps(params, sort_keys=True),
                       best_epochs=json.dumps(epochs))
            rows.append(row)
            pd.DataFrame(rows).to_csv(directory / 'review_candidates.csv', index=False)
            if np.isfinite(mean) and (best is None or (mean, -std) > (best['mean_score'], -best['std_score'])):
                best = dict(candidate=candidate, mean_score=mean, std_score=std, params=params,
                            refit_epochs=max(1, int(np.floor(np.median(epochs) + 0.5))), best_epochs=epochs)
            print(f'{directory.name}: review {candidate}, score={mean:.5f}, seed SD={std:.5f}', flush=True)
    if best is None:
        raise ValueError('全部复核候选指标无定义；不会退回初筛分数选参')
    dump_json(directory / 'selected.json', best)
    return best


def refit(data, selected, device, seed):
    features, target = data[:2]
    scalers = [MinMaxScaler().fit(x) for x in features]
    fit = core.FeatureDataset(*[pd.DataFrame(s.transform(x)) for s, x in zip(scalers, features)], target)
    core.seed_everything(seed)
    model = core.make_model([x.shape[1] for x in features], selected['params'], device)
    # Epoch count comes only from reviewed inner folds. There is no held-out early stopping.
    core.train_model(model, fit, selected['params'], device, fixed_epochs=selected['refit_epochs'])
    return model, scalers


def predict(data, model, scalers, batch_size, device, routing_dir=None,
            branch_names=('x1', 'x2', 'x3'), model_id='model', split='evaluation', plots=True):
    features, target = data[:2]
    dataset = core.FeatureDataset(*[pd.DataFrame(s.transform(x)) for s, x in zip(scalers, features)], target)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    if routing_dir is None:
        return core.evaluate_model(model, loader, device)[0]
    prediction, _, routing = evaluate_with_routing(model, loader, device)
    write_routing_reports(routing, data[4], data[3], routing_dir,
                          branch_names, model_id, split, plots)
    return prediction


def save_model(path, model, scalers, data, selected, seed):
    # A single portable, weights_only-loadable artifact, including fitted preprocessing.
    scaler_states = []
    for scaler in scalers:
        scaler_states.append({name: torch.from_numpy(np.asarray(getattr(scaler, name)).copy())
            for name in ('scale_', 'min_', 'data_min_', 'data_max_', 'data_range_')})
    torch.save(dict(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
        params=selected['params'], refit_epochs=selected['refit_epochs'], seed=seed,
        input_dims=[x.shape[1] for x in data[0]], feature_columns=[list(x.columns) for x in data[0]],
        scalers=scaler_states, scaler_feature_range=[0, 1], schema_version=2,
        routing_mode='full_softmax_topk_no_renormalization'), path)


def run(args):
    import optuna
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError('输出目录须为空或不存在，以免混合不同实验结果')
    train = load_data(args.train_paths, args.train_meta, args.group_col, args.protein_col)
    outer_splits = core.grouped_splits(train[2], 5)
    for fit, _ in outer_splits:
        local = subset(train, fit)
        validate_plans(local, core.grouped_splits(local[2], 3), args.scoring, args.min_group_n)
    validate_plans(train, core.grouped_splits(train[2], 3), args.scoring, args.min_group_n)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)
    output.mkdir(parents=True, exist_ok=True)
    versions = dict(python=platform.python_version(), torch=str(torch.__version__),
                    numpy=np.__version__, pandas=pd.__version__, optuna=optuna.__version__)
    hashes = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
              for name in ('MoE_3PLMs.py', 'moe_core.py', 'report_metrics.py', 'search_space.py',
                           'routing_reports.py')}
    config = dict(vars(args), outer_folds=5, inner_folds=3, versions=versions, source_sha256=hashes,
                  early_stopping='maximize configured validation correlation score every epoch',
                  selection='same correlation score as early stopping; mean folds then mean review seeds',
                  final_refit='median reviewed best epoch; no held-out early stopping')
    dump_json(output / 'config.json', config)
    upper = 6 * (3 * args.trials + 3 * args.shortlist * len(args.review_experts or [None]) * len(args.review_seeds) + 1)
    print(f'外5折/内3折；含全训练集最终搜索，最多约 {upper} 次训练（剪枝可减少）。设备={device}', flush=True)
    oof = prediction_frame(train[4], train[3], train[1], np.full(len(train[1]), np.nan), 'outer_oof')
    oof['outer_fold'] = 0
    split_rows, fold_pooled, fold_macro = [], [], []
    for fold, (fit, held) in enumerate(outer_splits, 1):
        for role, indices in [('train', fit), ('held_out', held)]:
            split_rows.append(pd.DataFrame({'outer_fold': fold, 'role': role,
                'ID': train[4].iloc[indices].to_numpy(), 'cv_group': train[2].iloc[indices].to_numpy(),
                'protein_id': train[3].iloc[indices].to_numpy()}))
    pd.concat(split_rows).to_csv(output / 'outer_splits.csv', index=False)
    for fold, (fit, held) in enumerate(outer_splits, 1):
        directory = output / f'outer_{fold}'
        print(f'\n===== 外层 {fold}/5 =====', flush=True)
        local = subset(train, fit)
        selected = staged_search(local, args, device, directory, args.seed + fold)
        model, scalers = refit(local, selected, device, args.seed)
        held_data = subset(train, held)
        prediction = predict(held_data, model, scalers, selected['params']['batch_size'], device,
            routing_dir=directory / 'evaluation' / 'routing', branch_names=args.branch_names,
            model_id=f'outer_{fold}', split='outer_held_out', plots=not args.no_routing_plots)
        frame = prediction_frame(held_data[4], held_data[3], held_data[1], prediction, f'outer_{fold}')
        pooled, _, macro = write_reports(frame, directory / 'evaluation', args.min_group_n)
        fold_pooled.append(pooled.assign(outer_fold=fold))
        fold_macro.append(macro.assign(outer_fold=fold))
        if args.save_outer_models:
            save_model(directory / 'model.pt', model, scalers, local, selected, args.seed)
        oof.loc[held, 'prediction'] = prediction
        oof.loc[held, 'outer_fold'] = fold
        oof.to_csv(output / 'oof_progress.csv', index=False)
        del model
    if not np.isfinite(oof.prediction).all() or oof.outer_fold.eq(0).any():
        raise RuntimeError('OOF 预测覆盖不完整')
    write_reports(oof, output / 'nested_oof', args.min_group_n)
    combine_outer_reports([output / f'outer_{fold}' / 'evaluation' / 'routing' for fold in range(1, 6)],
                          output / 'nested_oof' / 'routing')
    pooled = pd.concat(fold_pooled, ignore_index=True)
    macro = pd.concat(fold_macro, ignore_index=True)
    pooled.to_csv(output / 'outer_pooled_metrics.csv', index=False)
    macro.to_csv(output / 'outer_macro_metrics.csv', index=False)
    summary = []
    for metric in ['mae', 'rmse', 'r2', 'pearson', 'spearman', 'kendall_weighted']:
        values = pooled[metric]
        summary.append(dict(aggregation='pooled', metric=metric, mean=values.mean(),
                            std=values.std(ddof=1), valid_folds=values.notna().sum()))
    for metric, frame in macro.groupby('metric'):
        values = frame.value
        summary.append(dict(aggregation='macro', metric=metric, mean=values.mean(),
                            std=values.std(ddof=1), valid_folds=values.notna().sum()))
    pd.DataFrame(summary).to_csv(output / 'outer_summary.csv', index=False)
    print('\n===== 完整训练集：独立3折搜索与复核，确定最终参数 =====', flush=True)
    directory = output / 'final_selection'
    selected = staged_search(train, args, device, directory, args.seed + 100)
    model, scalers = refit(train, selected, device, args.seed)
    save_model(output / 'final_model.pt', model, scalers, train, selected, args.seed)
    # Load the external test only after every selection and final fit is finished.
    if args.test_meta:
        test = load_data(args.test_paths, args.test_meta, args.group_col, args.protein_col)
        if set(train[4]) & set(test[4]):
            raise ValueError('训练集与外部测试集存在重复 ID，拒绝生成独立测试报告')
        for fit_x, test_x in zip(train[0], test[0]):
            if list(fit_x.columns) != list(test_x.columns):
                raise ValueError('训练与测试特征列及顺序必须一致')
        overlap = dict(cv_groups=len(set(train[2]) & set(test[2])),
                       proteins=len(set(train[3]) & set(test[3])), sample_ids=0)
        dump_json(output / 'external_group_overlap.json', overlap)
        if overlap['cv_groups'] or overlap['proteins']:
            print(f'外部数据分组交集：{overlap}；外部测试不等同于未见蛋白质评估。', flush=True)
        prediction = predict(test, model, scalers, selected['params']['batch_size'], device,
            routing_dir=output / 'external_test' / 'routing', branch_names=args.branch_names,
            model_id='final_model', split='external_test', plots=not args.no_routing_plots)
        write_reports(prediction_frame(test[4], test[3], test[1], prediction, 'external_test'),
                      output / 'external_test', args.min_group_n)
    print(f'完成，结果位于 {output.resolve()}', flush=True)
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ('train', 'test'):
        for branch in (1, 2, 3):
            parser.add_argument(f'--{split}_x{branch}', required=split == 'train')
        parser.add_argument(f'--{split}_meta', required=split == 'train')
    parser.add_argument('-o', '--output-dir', required=True)
    parser.add_argument('--group-col', '--group_col', help='CV分组列；默认从ID提取蛋白质accession，可指定同源簇列')
    parser.add_argument('--protein-col', help='逐蛋白质报告分组列；默认从ID提取accession')
    parser.add_argument('--scoring', choices=['pooled_correlation', 'macro_correlation',
        'pooled_spearman', 'macro_spearman', 'pooled_pearson', 'macro_pearson'], default='pooled_correlation',
        help='早停、剪枝与选参统一使用此指标；correlation表示Spearman和Pearson加权平均')
    parser.add_argument('--spearman-weight', type=float, default=0.5,
                        help='correlation模式中Spearman权重，Pearson权重=1-此值')
    parser.add_argument('--search-space', default=str(Path(__file__).with_name('search_space.json')),
                        help='JSON搜索范围，hidden_dim_mlp直接填写独立维度')
    parser.add_argument('--min-group-n', type=int, default=10)
    parser.add_argument('--trials', type=int, default=12)
    parser.add_argument('--shortlist', type=int, default=3)
    parser.add_argument('--review-seeds', type=int, nargs='+', default=[101, 202, 303])
    parser.add_argument('--review-experts', type=int, nargs='+',
                        help='可选：复核时替换的专家数列表；不填则保留初筛候选专家数')
    parser.add_argument('--screen-seed', type=int, default=42)
    parser.add_argument('--seed', type=int, default=42, help='TPE及重训种子')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20, help='连续多少次验证相关得分无改善后停止')
    parser.add_argument('--prune-warmup', type=int, default=25)
    parser.add_argument('--startup-trials', type=int, default=5)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--save-outer-models', action='store_true')
    parser.add_argument('--branch-names', nargs=3, default=['x1', 'x2', 'x3'],
                        help='三个输入分支的标签，顺序对应x1/x2/x3；不是专家的生物学功能标签')
    parser.add_argument('--no-routing-plots', action='store_true', help='仅输出路由CSV，不生成PNG/PDF图')
    args = parser.parse_args(argv)
    if len(set(args.branch_names)) != 3 or 'all_branches' in args.branch_names:
        parser.error('branch-names须互不重复，且不能使用保留名称all_branches')
    try:
        args.search_space_config = load_space(args.search_space)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if not np.isfinite(args.spearman_weight) or not 0 <= args.spearman_weight <= 1:
        parser.error('spearman-weight须在[0,1]之间')
    args.train_paths = [getattr(args, f'train_x{i}') for i in (1, 2, 3)]
    args.test_paths = [getattr(args, f'test_x{i}') for i in (1, 2, 3)]
    if any([args.test_meta, *args.test_paths]) and not all([args.test_meta, *args.test_paths]):
        parser.error('外部测试需要同时提供 test_meta 和三个 test_x 参数')
    for name in ('trials', 'shortlist', 'epochs', 'patience', 'prune_warmup', 'startup_trials'):
        if getattr(args, name) < 1:
            parser.error(name + ' 必须为正整数')
    if args.min_group_n < 3 or args.shortlist > args.trials:
        parser.error('min-group-n须>=3，shortlist不能超过trials')
    if len(args.review_seeds) != len(set(args.review_seeds)) or args.screen_seed in args.review_seeds:
        parser.error('review-seeds须互不重复，且与screen-seed不同')
    if args.review_experts:
        top_k = args.search_space_config['top_k']
        max_top_k = max(top_k) if isinstance(top_k, list) else top_k['high']
        if len(args.review_experts) != len(set(args.review_experts)) or min(args.review_experts) < max_top_k:
            parser.error('review-experts不能重复，且每项须>=搜索范围中top_k的最大值')
    if any(x < 0 or x >= 2**32 - 100000 for x in [args.seed, args.screen_seed, *args.review_seeds]):
        parser.error('种子须在[0, 2**32-100000)之间')
    return args


if __name__ == '__main__':
    run(parse_args())
