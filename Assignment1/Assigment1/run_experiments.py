"""Reproduce all results and publication figures. No network or data download."""
import argparse
import json
import platform
import sys
from pathlib import Path
from time import perf_counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn
from Part_1.task1_dataset import generate_gaussian_dataset
from Part_1.perceptron import Perceptron
from Part_2.train_mlp_numpy import train

ROOT = Path(__file__).resolve().parent
CONDITIONS = [('Separated', 4.0, 1.0), ('Close means', 1.0, 1.0),
              ('High variance', 4.0, 9.0), ('Close + high variance', 1.0, 9.0)]
BATCH_SIZES = [1, 16, 32, 128, 800]
SEEDS = [42, 43, 44]
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 120})


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def boundary(ax, predictor, data, title):
    x, y, xt, yt = data
    if y.ndim == 2:
        y, yt = y.argmax(axis=1), yt.argmax(axis=1)
    all_x = np.vstack((x, xt))
    lo, hi = all_x.min(axis=0) - 0.5, all_x.max(axis=0) + 0.5
    xx, yy = np.meshgrid(np.linspace(lo[0], hi[0], 180), np.linspace(lo[1], hi[1], 180))
    pred = predictor(np.column_stack((xx.ravel(), yy.ravel())))
    if pred.ndim == 2:
        pred = pred.argmax(axis=1)
    ax.contourf(xx, yy, pred.reshape(xx.shape), levels=[-2, 0, 2] if y.min() < 0 else [-.5, .5, 1.5],
                colors=['#dfebfa', "#6d3622"], alpha=0.8)
    for label, color in zip(np.unique(y), ['#2864a0', '#d55b38']):
        ax.scatter(*x[y == label].T, s=10, c=color, alpha=0.55, label=f'{label:+d} train')
        ax.scatter(*xt[yt == label].T, s=22, facecolors='none', edgecolors=color, label=f'{label:+d} test')
    ax.set(title=title, xlabel='x1', ylabel='x2')


def run_part1(out):
    cases = []
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    errfig, errax = plt.subplots(figsize=(9, 3.4))
    for case_index, (name, distance, variance) in enumerate(CONDITIONS):
        runs = []
        for seed in range(42, 52):
            data = generate_gaussian_dataset(seed, distance, variance)
            x, y, xt, yt = data
            model = Perceptron(2, max_epochs=100, learning_rate=.01).train(x, y)
            row = {'seed': seed, 'train_accuracy': float(100 * np.mean(model.forward(x) == y)),
                   'test_accuracy': float(100 * np.mean(model.forward(xt) == yt)),
                   'epochs_checked': len(model.errors_per_epoch),
                   'converged': bool(model.errors_per_epoch[-1] == 0),
                   'weights': model.weights.tolist(), 'errors': model.errors_per_epoch}
            runs.append(row)
            if seed == 42:
                np.savez(out / f'part1_case{case_index + 1}_data.npz',
                         X_train=x, y_train=y, X_test=xt, y_test=yt)
                boundary(axes.flat[case_index], model.forward, data,
                         f"{name} | test {row['test_accuracy']:.1f}%")
                errax.plot(np.arange(1, len(row['errors']) + 1), row['errors'], label=name)
        cases.append({'name': name, 'mean_distance_per_coordinate': distance,
                      'variance': variance, 'runs': runs})
    axes.flat[0].legend(fontsize=8, loc='upper left')
    save_figure(fig, out / 'figures/part1_boundaries.png')
    errax.set(xlabel='Epoch (before update)', ylabel='Misclassified training samples')
    errax.legend(fontsize=8, ncol=2)
    errax.grid(alpha=.2)
    save_figure(errfig, out / 'figures/part1_errors.png')
    save_json(out / 'part1_results.json', cases)
    return cases


def plot_mlp(results, out):
    for size, stem in [(800, 'part2_full_batch'), (1, 'part3_sgd')]:
        run = next(r for r in results if r['seed'] == 42 and r['batch_size'] == size)
        h = run['history']
        epochs = [r['epoch'] for r in h]
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        for split, color in [('train', '#2864a0'), ('test', '#d55b38')]:
            axes[0].plot(epochs, [r[f'{split}_accuracy'] for r in h], label=split, color=color)
            axes[1].plot(epochs, [r[f'{split}_loss'] for r in h], label=split, color=color)
        axes[0].set(xlabel='Epoch', ylabel='Accuracy (%)', ylim=(40, 102))
        axes[1].set(xlabel='Epoch', ylabel='Mean cross entropy')
        for ax in axes:
            ax.grid(alpha=.2)
            ax.legend()
        save_figure(fig, out / f'figures/{stem}.png')

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for size in BATCH_SIZES:
        runs = [r for r in results if r['batch_size'] == size]
        vals = np.array([[h['test_accuracy'] for h in r['history']] for r in runs])
        epochs = [h['epoch'] for h in runs[0]['history']]
        mean, std = vals.mean(axis=0), vals.std(axis=0, ddof=1)
        line, = axes[0].plot(epochs, mean, label=f'B={size}')
        axes[0].fill_between(epochs, mean - std, mean + std, color=line.get_color(), alpha=.1)
        seed42 = runs[0]['history']
        axes[1].plot([r['updates'] for r in seed42], [r['test_accuracy'] for r in seed42], label=f'B={size}')
    axes[0].set(xlabel='Epoch', ylabel='Test accuracy (%)', title='Mean +/- sample SD (3 seeds)')
    axes[1].set(xlabel='Parameter updates (log scale)', ylabel='Test accuracy (%)',
                title='Seed 42; unequal update budgets', xscale='symlog', xlim=(1, 1.3e6))
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    save_figure(fig, out / 'figures/batch_comparison.png')

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    grouped = [[r for r in results if r['batch_size'] == size] for size in BATCH_SIZES]
    for key, ax, label in [('test_accuracy', axes[0], 'Final test accuracy (%)'),
                           ('elapsed_seconds', axes[1], 'Runtime (seconds)')]:
        values = [[r['history'][-1][key] for r in group] for group in grouped]
        ax.bar(range(5), [np.mean(v) for v in values],
               yerr=[np.std(v, ddof=1) for v in values], capsize=4, color='#2864a0')
        ax.set(xticks=range(5), xticklabels=[str(s) for s in BATCH_SIZES], xlabel='Batch size', ylabel=label)
    axes[0].set_ylim(0, 105)
    save_figure(fig, out / 'figures/batch_summary.png')


def run_all(output_dir=None):
    out = ROOT / 'results' if output_dir is None else Path(output_dir)
    (out / 'figures').mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    run_part1(out)
    print('Part I complete: 4 conditions x 10 seeds', flush=True)
    results = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for seed in SEEDS:
        for size in BATCH_SIZES:
            print(f'Training MLP: seed={seed}, batch_size={size}, epochs=1500', flush=True)
            model, history, data = train(batch_size=size, seed=seed, verbose=False)
            row = {'seed': seed, 'batch_size': size, 'history': history,
                   'weights': [{k: v.tolist() for k, v in layer.params.items()}
                               for layer in model.linear_layers]}
            predictions = model.forward(data[2]).argmax(axis=1)
            truth = data[3].argmax(axis=1)
            row['test_confusion_matrix'] = [[int(np.sum((truth == i) & (predictions == j)))
                                            for j in range(2)] for i in range(2)]
            results.append(row)
            save_json(out / 'mlp_results.json', results)
            if seed == 42 and size in (1, 800):
                boundary(axes[0 if size == 800 else 1], model.forward, data,
                         f"B={size} | test {history[-1]['test_accuracy']:.1f}%")
            if size == 800:
                np.savez(out / f'moons_seed{seed}.npz',
                         X_train=data[0], y_train=data[1], X_test=data[2], y_test=data[3])
            print(f"Finished: test={history[-1]['test_accuracy']:.2f}%, "
                  f"time={history[-1]['elapsed_seconds']:.2f}s", flush=True)
    save_figure(fig, out / 'figures/mlp_boundaries.png')
    plot_mlp(results, out)
    metadata = {'student_id': '12411103', 'python': sys.version, 'platform': platform.platform(),
                'numpy': np.__version__, 'scikit_learn': sklearn.__version__,
                'matplotlib': matplotlib.__version__, 'mlp_seeds': SEEDS,
                'perceptron_seeds': list(range(42, 52)), 'batch_sizes': BATCH_SIZES,
                'mlp_config': {'hidden': [20], 'learning_rate': .01, 'epochs': 1500,
                               'eval_freq': 10, 'noise': 0.0, 'weight_std': .1},
                'total_seconds': perf_counter() - started}
    save_json(out / 'metadata.json', metadata)
    print(f'All results written to {out}', flush=True)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', type=Path, default=ROOT / 'results')
    run_all(parser.parse_args().output_dir)
