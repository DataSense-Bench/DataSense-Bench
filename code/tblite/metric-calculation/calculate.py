"""Aggregate selection metrics from per-run, evaluation-averaged group scores.

Input JSON: {"base": 0.1, "random": 0.12, "runs": [
  {"id": "run1", "scores": [0.2, 0.18, 0.16, 0.14, 0.12]},
  {"id": "run2", "failed": true}
]}. Use unrounded scores in one consistent unit; missing evaluations are errors.
"""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev


def finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Scores must be finite numbers')
    return float(value)


def tied(x, y):
    # Preserve mathematical ties after floating-point averaging.
    return math.isclose(x, y, rel_tol=1e-14, abs_tol=1e-14)


def ranks(values):
    return [1 + sum(y < x and not tied(y, x) for y in values)
            + (sum(tied(y, x) for y in values) - 1) / 2 for x in values]


def correlation(x, y):
    a, b = mean(x), mean(y)
    denom = math.sqrt(sum((v-a)**2 for v in x) * sum((v-b)**2 for v in y))
    return sum((v-a)*(w-b) for v, w in zip(x, y)) / denom if denom else None


def calculate(data):
    base, random = finite(data['base']), finite(data['random'])
    rows = []
    if not data['runs']:
        raise ValueError('At least one run is required')
    for run in data['runs']:
        if run.get('failed') is True:
            top, rho, best = base, -1.0, 0
        else:
            scores = [finite(x) for x in run['scores']]
            if len(scores) != 5:
                raise ValueError('Each valid run needs five group scores in predicted order')
            top, rho, best = scores[0], correlation([5,4,3,2,1], ranks(scores)), int(tied(scores[0], max(scores)))
        rows.append({'id': run.get('id'), 'top_group': top, 'gain': top-random,
                     'spearman': rho, 'best_group_accuracy': best})
    summary = {}
    for key in ['top_group', 'gain', 'spearman', 'best_group_accuracy']:
        values = [r[key] for r in rows if r[key] is not None]
        summary[key] = {'mean': mean(values) if values else None,
                        'std': stdev(values) if len(values) > 1 else None, 'n': len(values)}
    return {'runs': rows, 'summary': summary, 'std_convention': 'sample (ddof=1)'}


def calculate_csv(path, references):
    """Scores use fractions/rewards in [0,1]; group order is never re-sorted."""
    grouped = {}
    with path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            task, agent = row['task'], row['agent']
            if task != 'tblite':
                continue
            run, group = int(row['run']), int(row['group'])
            score = finite(float(row['score']))
            if task not in ('tblite', 'bfcl') or run < 1 or group not in range(1, 6) or not 0 <= score <= 1:
                raise ValueError('Invalid task, run, group or score')
            expected = 100 if task == 'tblite' else 800
            if int(row['n_evaluations']) != 3 or int(row['cases_per_evaluation']) != expected:
                raise ValueError('Checkpoint must cover the full three-evaluation protocol')
            groups = grouped.setdefault((task, agent), {}).setdefault(run, {})
            if group in groups:
                raise ValueError(f'Duplicate checkpoint: {task}/{agent}/run{run}/g{group}')
            groups[group] = score
    if not grouped:
        raise ValueError('Score CSV is empty')
    tasks = {}
    for (task, agent), runs in sorted(grouped.items()):
        if any(set(groups) != set(range(1, 6)) for groups in runs.values()):
            raise ValueError(f'Incomplete five-group selection: {task}/{agent}')
        ref = references[task]
        scale = 100 if ref['unit'] == 'accuracy_percent' else 1
        if ref['unit'] not in ('accuracy_percent', 'accuracy_fraction', 'mean_reward'):
            raise ValueError(f'Unknown reference unit: {ref["unit"]}')
        base, random = finite(ref['base']['mean']) / scale, finite(ref['random']['mean']) / scale
        records = [{'id': f'run{run}', 'scores': [groups[g] for g in range(1, 6)]}
                   for run, groups in sorted(runs.items())]
        result = calculate({'base': base, 'random': random, 'runs': records})
        result['groups'] = {
            f'g{g}': {'mean': mean(values), 'std': stdev(values) if len(values) > 1 else None,
                     'max': max(values), 'n': len(values)}
            for g in range(1, 6) for values in [[groups[g] for groups in runs.values()]]}
        task_result = tasks.setdefault(task, {'unit': 'mean_reward' if task == 'tblite' else 'accuracy_fraction',
                                             'base': base, 'random': random, 'agents': {}})
        task_result['agents'][agent] = result
    return {'tasks': tasks, 'std_convention': 'sample (ddof=1)'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--references', type=Path,
                   default=Path(__file__).resolve().parents[3] / 'results/baselines/baseline-aggregates-user-20260926.json',
                   help='Base/random reference JSON used with score CSV input')
    a = p.parse_args()
    data = (calculate_csv(a.input, json.loads(a.references.read_text())) if a.input.suffix == '.csv'
            else calculate(json.loads(a.input.read_text())))
    result = json.dumps(data, indent=2, allow_nan=False) + '\n'
    if a.output:
        a.output.write_text(result)
    else:
        print(result, end='')


if __name__ == '__main__':
    main()
