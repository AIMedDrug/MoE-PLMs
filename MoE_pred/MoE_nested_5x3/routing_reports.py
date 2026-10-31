"""Inference-only diagnostics for the three calls to the shared sparse MoE.

Expert IDs are local to one fitted model. Never pool expert IDs across folds.
All entropy uses natural logarithms; unused experts remain in all reports.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import moe_core as core


def entropy(probabilities):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    logs = np.zeros_like(probabilities)
    np.log(probabilities, out=logs, where=probabilities > 0)
    return -(probabilities * logs).sum(axis=-1)


def evaluate_with_routing(model, loader, device):
    """Return predictions, targets and [sample, branch, expert] routing arrays."""
    if getattr(model.moe, 'routing_observer', None) is not None:
        raise ValueError('A routing observer is already installed')
    pending, dense, sparse, selected, top1 = [], [], [], [], []
    predictions, targets = [], []
    was_training = model.training

    def observe(logits, indices, weights):
        actual = torch.zeros_like(logits).scatter(1, indices, weights)
        mask = torch.zeros_like(logits, dtype=torch.bool).scatter(1, indices, True)
        pending.append((torch.softmax(logits, dim=1).cpu().numpy(),
                        actual.cpu().numpy(), mask.cpu().numpy(),
                        indices[:, 0].cpu().numpy()))

    model.moe.routing_observer = observe
    try:
        model.eval()
        with torch.no_grad():
            for x1, x2, x3, target in loader:
                pending.clear()
                prediction = model(x1.to(device), x2.to(device), x3.to(device))
                if len(pending) != 3 or any(len(p[0]) != len(target) for p in pending):
                    raise ValueError('Expected exactly three single-token MoE calls per batch')
                for destination, position in [(dense, 0), (sparse, 1), (selected, 2), (top1, 3)]:
                    destination.append(np.stack([p[position] for p in pending], axis=1))
                predictions.append(prediction.cpu().numpy().reshape(-1))
                targets.append(target.numpy().reshape(-1))
    finally:
        del model.moe.routing_observer
        model.train(was_training)
    if not predictions:
        raise ValueError('Cannot report an empty dataset')
    prediction = np.concatenate(predictions)
    if not np.isfinite(prediction).all():
        raise ValueError('Non-finite predictions')
    return prediction, np.concatenate(targets), {
        'dense': np.concatenate(dense), 'sparse': np.concatenate(sparse),
        'selected': np.concatenate(selected), 'top1': np.concatenate(top1),
        'top_k': model.moe.top_k,
    }


def distribution(values):
    values = np.asarray(values, dtype=float)
    if not values.size:
        return {key: np.nan for key in ('mean', 'std', 'min', 'q05', 'q25', 'median', 'q75', 'q95', 'max')}
    quantiles = np.quantile(values, [0, .05, .25, .5, .75, .95, 1])
    return dict(zip(('mean', 'std', 'min', 'q05', 'q25', 'median', 'q75', 'q95', 'max'),
                    [values.mean(), values.std(ddof=0), *quantiles]))


def build_routing_tables(routing, ids, proteins, branch_names, model_id, split):
    dense, sparse, selected = [np.asarray(routing[k]) for k in ('dense', 'sparse', 'selected')]
    top1 = np.asarray(routing['top1'])
    n, branches, experts = sparse.shape
    k = int(routing['top_k'])
    if n == 0 or branches != 3 or len(branch_names) != 3 or len(set(branch_names)) != 3:
        raise ValueError('Require nonempty data and three unique branch labels')
    if 'all_branches' in branch_names or len(ids) != n or len(proteins) != n:
        raise ValueError('Invalid branch labels or metadata lengths')
    if pd.Series(ids).isna().any() or pd.Series(ids).duplicated().any():
        raise ValueError('Sample IDs must be nonmissing and unique within one model/split')
    if dense.shape != sparse.shape or selected.shape != sparse.shape or top1.shape != (n, branches):
        raise ValueError('Inconsistent routing shapes')
    if not 1 <= k <= experts or not np.isin(selected, [False, True]).all():
        raise ValueError('Invalid top-k or selection mask')
    selected = selected.astype(bool)
    for weights in (dense, sparse):
        if not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError('Gate probabilities must be finite and nonnegative')
    if not np.allclose(dense.sum(-1), 1, atol=1e-6):
        raise ValueError('Full-softmax probabilities must sum to one')
    if not np.allclose(sparse, dense * selected, atol=1e-7, rtol=1e-6):
        raise ValueError('Selected weights must retain full-softmax probabilities without renormalization')
    if not np.all(selected.sum(-1) == k) or np.any(sparse[~selected] != 0):
        raise ValueError('Sparse weights disagree with actual Top-k selection')
    if not np.issubdtype(top1.dtype, np.integer) or np.any((top1 < 0) | (top1 >= experts)):
        raise ValueError('Invalid top-1 indices')
    if not np.take_along_axis(selected, top1[..., None], axis=-1).all():
        raise ValueError('Top-1 expert must be selected')

    identity = dict(model_id=str(model_id), split=str(split), n_experts=experts, top_k=k)
    retained_mass = sparse.sum(-1)
    if (retained_mass <= 0).any():
        raise ValueError('Selected probability mass must be positive')
    # Conditional normalization is ONLY for the entropy statistic, never inference.
    conditional = sparse / retained_mass[..., None]
    pre_entropy, post_entropy = entropy(dense), entropy(conditional)
    ids, proteins = np.asarray(ids), np.asarray(proteins)
    events = pd.DataFrame({
        **identity, 'ID': np.repeat(ids, branches * experts),
        'protein_id': np.repeat(proteins, branches * experts),
        'branch': np.tile(np.repeat(branch_names, experts), n),
        'expert': np.tile(np.arange(1, experts + 1), n * branches),
        'selected': selected.reshape(-1),
        'is_top1': (np.arange(experts) == top1[..., None]).reshape(-1),
        'pre_topk_probability': dense.reshape(-1),
        'gate_weight': sparse.reshape(-1),
    })
    branch_samples = pd.DataFrame({
        **identity, 'ID': np.repeat(ids, branches), 'protein_id': np.repeat(proteins, branches),
        'branch': np.tile(branch_names, n), 'top1_expert': top1.reshape(-1) + 1,
        'pre_topk_entropy_nats': pre_entropy.reshape(-1),
        'pre_topk_entropy_normalized': pre_entropy.reshape(-1) / np.log(experts) if experts > 1 else 0.,
        'routing_entropy_nats': post_entropy.reshape(-1),
        'routing_entropy_normalized': post_entropy.reshape(-1) / np.log(k) if k > 1 else 0.,
        'retained_probability_mass': retained_mass.reshape(-1),
        'selected_entropy_contribution_nats': entropy(sparse).reshape(-1),
    })
    mean_weights = sparse.mean(axis=1)
    primary = mean_weights.argmax(-1)  # deterministic smallest expert ID on exact ties
    samples = pd.DataFrame({
        **identity, 'ID': ids, 'protein_id': proteins, 'primary_expert': primary + 1,
        'primary_tie_count': (mean_weights == mean_weights.max(-1, keepdims=True)).sum(-1),
        'n_distinct_selected_experts': selected.any(axis=1).sum(-1),
        'mean_branch_routing_entropy_nats': post_entropy.mean(axis=1),
        'mean_branch_routing_entropy_normalized': post_entropy.mean(axis=1) / np.log(k) if k > 1 else 0.,
        'mean_branch_pre_topk_entropy_nats': pre_entropy.mean(axis=1),
        'aggregate_weight_entropy_nats': entropy(mean_weights / mean_weights.sum(-1, keepdims=True)),
        'mean_retained_probability_mass': retained_mass.mean(axis=1),
    })
    for expert in range(experts):
        samples[f'mean_gate_weight_expert_{expert + 1}'] = mean_weights[:, expert]
        samples[f'any_branch_selected_expert_{expert + 1}'] = selected[:, :, expert].any(axis=1)

    utilization, weights, histograms, entropy_rows = [], [], [], []
    bins = np.linspace(0, 1, 21)
    for branch_index, label in [*enumerate(branch_names), (None, 'all_branches')]:
        sel = selected if branch_index is None else selected[:, branch_index:branch_index + 1]
        post = sparse if branch_index is None else sparse[:, branch_index:branch_index + 1]
        pre = dense if branch_index is None else dense[:, branch_index:branch_index + 1]
        winner = top1 if branch_index is None else top1[:, branch_index:branch_index + 1]
        prefix = dict(identity, branch=label)
        decisions = sel.shape[0] * sel.shape[1]
        for expert in range(experts):
            count = int(sel[..., expert].sum())
            utilization.append(dict(prefix, expert=expert + 1, n_samples=n,
                n_routing_decisions=decisions, selection_count=count,
                utilization_frequency=count / decisions,
                routing_slot_share=count / (decisions * k),
                top1_count=int((winner == expert).sum()),
                top1_proportion=float((winner == expert).mean()),
                any_branch_sample_count=int(sel[..., expert].any(axis=1).sum()),
                any_branch_sample_proportion=float(sel[..., expert].any(axis=1).mean()),
                mean_gate_weight=float(post[..., expert].mean())))
            for kind, values in [
                ('post_topk_all', post[..., expert].reshape(-1)),
                ('post_topk_selected', post[..., expert][sel[..., expert]]),
                ('pre_topk_all', pre[..., expert].reshape(-1)),
            ]:
                weights.append(dict(prefix, expert=expert + 1, distribution=kind,
                                    n_values=values.size, **distribution(values)))
                counts, _ = np.histogram(values, bins=bins)
                for index, count_bin in enumerate(counts):
                    histograms.append(dict(prefix, expert=expert + 1, distribution=kind,
                        bin_left=bins[index], bin_right=bins[index + 1], count=int(count_bin),
                        proportion=count_bin / values.size if values.size else np.nan))
        for name, values in [
            ('pre_topk_entropy_nats', entropy(pre).reshape(-1)),
            ('pre_topk_entropy_normalized', entropy(pre).reshape(-1) / np.log(experts) if experts > 1 else np.zeros(decisions)),
            ('routing_entropy_nats', entropy(post / post.sum(-1, keepdims=True)).reshape(-1)),
            ('routing_entropy_normalized', entropy(post / post.sum(-1, keepdims=True)).reshape(-1) / np.log(k) if k > 1 else np.zeros(decisions)),
            ('retained_probability_mass', post.sum(-1).reshape(-1)),
            ('selected_entropy_contribution_nats', entropy(post).reshape(-1)),
        ]:
            entropy_rows.append(dict(prefix, metric=name, n_values=values.size, **distribution(values)))
    assignments = []
    for expert in range(experts):
        assignments.append(dict(identity, expert=expert + 1, n_samples=n,
            primary_sample_count=int((primary == expert).sum()),
            primary_sample_proportion=float((primary == expert).mean()),
            any_branch_sample_count=int(selected[..., expert].any(axis=1).sum()),
            any_branch_sample_proportion=float(selected[..., expert].any(axis=1).mean()),
            mean_sample_gate_weight=float(mean_weights[:, expert].mean())))
    return dict(gate_weights=events, sample_branch_routing=branch_samples, sample_routing=samples,
                expert_utilization=pd.DataFrame(utilization),
                sample_assignment=pd.DataFrame(assignments),
                gate_weight_summary=pd.DataFrame(weights),
                gate_weight_histogram=pd.DataFrame(histograms),
                routing_entropy_summary=pd.DataFrame(entropy_rows))


def plot_routing(tables, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    utilization = tables['expert_utilization']
    branches = utilization.branch.drop_duplicates().tolist()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, branch in zip(axes.flat, branches):
        frame = utilization[utilization.branch == branch]
        ax.bar(frame.expert - .18, frame.utilization_frequency, width=.36, label='Top-k selection / decisions')
        ax.bar(frame.expert + .18, frame.top1_proportion, width=.36, label='Top-1 / decisions')
        ax.set(title=branch, xlabel='Expert (model-local ID)', ylabel='Proportion', ylim=(0, 1))
        ax.set_xticks(frame.expert)
    axes.flat[0].legend(fontsize=8)
    _save_figure(fig, directory, 'expert_utilization', plt)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    assignment = tables['sample_assignment']
    for ax, column, title in zip(axes, ['primary_sample_proportion', 'any_branch_sample_proportion'],
            ['Primary expert: argmax of mean branch weight', 'Selected by at least one branch (overlapping)']):
        ax.bar(assignment.expert, assignment[column])
        ax.set(title=title, xlabel='Expert (model-local ID)', ylabel='Sample proportion', ylim=(0, 1))
        ax.set_xticks(assignment.expert)
    _save_figure(fig, directory, 'sample_assignment', plt)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    frame = tables['sample_branch_routing']
    for branch, part in frame.groupby('branch', sort=False):
        for ax, column in zip(axes, ['pre_topk_entropy_normalized', 'routing_entropy_normalized']):
            ax.hist(part[column], bins=np.linspace(0, 1.000001, 21),
                    weights=np.ones(len(part)) / len(part), histtype='step', label=branch)
            label = ('Full-softmax entropy / log(E)' if column.startswith('pre_')
                     else 'Conditional selected entropy / log(k)')
            ax.set(xlabel=label, ylabel='Proportion of branch decisions', xlim=(0, 1))
    axes[0].legend(fontsize=8)
    _save_figure(fig, directory, 'routing_entropy', plt)

    events = tables['gate_weights']
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, branch in zip(axes.flat, branches):
        part = events if branch == 'all_branches' else events[events.branch == branch]
        for expert, rows in part.groupby('expert'):
            ax.hist(rows.gate_weight, bins=np.linspace(0, 1.000001, 21),
                weights=np.ones(len(rows)) / len(rows), histtype='step', label=f'E{expert}')
        ax.set(title=branch, xlabel='Actual gate weight (unselected = 0)', ylabel='Proportion', xlim=(0, 1))
    axes.flat[0].legend(fontsize=8, ncol=2)
    _save_figure(fig, directory, 'gate_weight_distributions', plt)


def _save_figure(fig, directory, name, plt):
    fig.savefig(Path(directory) / f'{name}.png', dpi=200)
    fig.savefig(Path(directory) / f'{name}.pdf')
    plt.close(fig)


def write_routing_reports(routing, ids, proteins, directory, branch_names=('x1', 'x2', 'x3'),
                          model_id='model', split='evaluation', plots=True):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tables = build_routing_tables(routing, ids, proteins, branch_names, model_id, split)
    for name, frame in tables.items():
        frame.to_csv(directory / f'{name}.csv', index=False)
    metadata = dict(
        schema_version=2, model_id=model_id, split=split, branches=list(branch_names),
        routing_mode='full_softmax_topk_no_renormalization',
        n_samples=len(ids), n_experts=routing['sparse'].shape[-1], top_k=int(routing['top_k']),
        entropy_units='nats; natural logarithm; 0*log(0)=0',
        pre_topk_probability='softmax over all gate logits; selected entries are used unchanged',
        gate_weight='full-softmax probability when selected, zero otherwise; no renormalization',
        utilization_frequency='selection_count / n_routing_decisions; sums to k across experts',
        routing_slot_share='selection_count / (n_routing_decisions * k); sums to 1',
        top1_proportion='top1_count / n_routing_decisions; sums to 1; actual torch.topk first index',
        routing_entropy_nats='conditional entropy of w/sum(w), for diagnostics ONLY; weights used by model stay unchanged',
        routing_entropy_normalized='conditional selected entropy / log(k); 0 when k=1',
        retained_probability_mass='sum of actual selected weights; generally less than 1 when k<E',
        selected_entropy_contribution_nats='-sum(w*log(w)); selected contribution to full entropy, not entropy of a normalized distribution',
        aggregate_weight_entropy_nats='entropy of branch-mean actual weights normalized ONLY for this statistic',
        pre_topk_entropy_normalized='pre-topk entropy / log(E); 0 when E=1',
        primary_sample_proportion='argmax of mean actual weights across the three branches; sums to 1',
        primary_ties='smallest expert ID wins an exact mean-weight tie; primary_tie_count is recorded',
        any_branch_sample_proportion='fraction selected at least once across branches; overlapping, does not sum to 1',
        all_branches='3*N routing decisions, N unique samples; same shared experts within one fitted model',
        distribution_std='population standard deviation, ddof=0',
        histogram='20 equal bins on [0,1]; left closed/right open except final bin includes 1',
        scope='Expert IDs are local to one fitted model; branch states already mix all three PLM inputs',
        interpretation='Routing diagnostics alone do not establish structural/evolutionary/functional specialization',
    )
    (directory / 'routing_config.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    if plots:
        plot_routing(tables, directory)
    return tables


def combine_outer_reports(directories, destination):
    """Stack fold-specific results without matching arbitrary expert identities."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name in ('expert_utilization', 'sample_assignment', 'gate_weight_summary',
                 'gate_weight_histogram', 'routing_entropy_summary', 'sample_branch_routing',
                 'sample_routing'):
        frames = [pd.read_csv(Path(path) / f'{name}.csv', dtype={'ID': str, 'protein_id': str})
                  for path in directories]
        combined = pd.concat(frames, ignore_index=True)
        if name == 'sample_routing' and combined.ID.duplicated().any():
            raise ValueError('A sample appears in more than one outer held-out fold')
        combined.to_csv(destination / f'{name}_by_model.csv', index=False)
    # Entropy is invariant to renumbering experts; summarize normalized values
    # across OOF samples, retaining fold-level rows in the files above.
    branches = pd.concat([pd.read_csv(Path(p) / 'sample_branch_routing.csv') for p in directories])
    rows = []
    for label, part in [('all_branches', branches), *list(branches.groupby('branch', sort=False))]:
        for metric in ('routing_entropy_nats', 'routing_entropy_normalized',
                       'pre_topk_entropy_nats', 'pre_topk_entropy_normalized',
                       'retained_probability_mass', 'selected_entropy_contribution_nats'):
            rows.append(dict(branch=label, metric=metric, n_values=len(part), **distribution(part[metric])))
    pd.DataFrame(rows).to_csv(destination / 'oof_entropy_summary.csv', index=False)
    (destination / 'README.txt').write_text(
        'Expert IDs are model-local. Rows from different outer folds must not be pooled by expert ID.\n'
        'OOF entropy summaries are sample/branch-decision weighted, not unweighted fold means.\n'
        'Raw gate weights and plots are in each outer_*/evaluation/routing directory.\n'
        'If k/E differ across folds, prefer normalized entropy and inspect per-model summaries.\n',
        encoding='utf-8')
