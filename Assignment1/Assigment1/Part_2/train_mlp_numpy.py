"""Train a NumPy MLP with full-batch, stochastic or mini-batch descent."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
try:
    from .mlp_numpy import MLP
    from .modules import CrossEntropy
except ImportError:
    from mlp_numpy import MLP
    from modules import CrossEntropy

DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500
EVAL_FREQ_DEFAULT = 10


def make_dataset(seed=42, noise=0.0):
    """1,000 points; random stratified 800/200 split; no standardization.
    The assignment does not specify noise; our default adds none.
    """
    x, labels = make_moons(n_samples=1000, noise=noise, random_state=seed)
    x_train, x_test, y_train, y_test = train_test_split(
        x, labels, test_size=0.2, stratify=labels, random_state=seed
    )
    return x_train, np.eye(2)[y_train], x_test, np.eye(2)[y_test]


def accuracy(predictions, targets):
    """Classification accuracy in percent."""
    return float(100 * np.mean(predictions.argmax(axis=1) == targets.argmax(axis=1)))


def train(dnn_hidden_units=DNN_HIDDEN_UNITS_DEFAULT,
          learning_rate=LEARNING_RATE_DEFAULT, max_steps=MAX_EPOCHS_DEFAULT,
          eval_freq=EVAL_FREQ_DEFAULT, batch_size=None, seed=42, noise=0.0,
          data=None, verbose=True):
    """One step means one EPOCH (one full pass through the training set).
    batch_size=None: full batch; 1: SGD; intermediate values: mini-batch.
    Return (model, history, data). Reset initialization/shuffling for each run.
    Test data never influence updates or checkpoint selection.
    """
    if max_steps < 1 or eval_freq < 1 or learning_rate <= 0:
        raise ValueError('epochs, eval_freq and learning_rate must be positive')
    if batch_size is not None and batch_size < 1:
        raise ValueError('batch_size must be positive or None')
    hidden = ([int(v) for v in dnn_hidden_units.split(',') if v.strip()]
              if isinstance(dnn_hidden_units, str) else list(dnn_hidden_units))
    data = make_dataset(seed, noise) if data is None else data
    x_train, y_train, x_test, y_test = data
    n_train = len(x_train)
    batch_size = n_train if batch_size is None else min(batch_size, n_train)
    model = MLP(x_train.shape[1], hidden, y_train.shape[1], seed)
    loss = CrossEntropy()
    rng = np.random.default_rng(seed + 10000)
    history = []
    updates = 0
    started = perf_counter()

    def evaluate(epoch):
        train_probs = model.forward(x_train)
        test_probs = model.forward(x_test)
        row = {
            'epoch': epoch, 'updates': updates,
            'elapsed_seconds': perf_counter() - started,
            'train_loss': loss.forward(train_probs, y_train),
            'test_loss': loss.forward(test_probs, y_test),
            'train_accuracy': accuracy(train_probs, y_train),
            'test_accuracy': accuracy(test_probs, y_test),
        }
        history.append(row)
        if verbose:
            print(f"Epoch {epoch:4d} | loss {row['train_loss']:.5f} | "
                  f"train {row['train_accuracy']:.2f}% | test {row['test_accuracy']:.2f}%",
                  flush=True)

    evaluate(0)
    for epoch in range(1, max_steps + 1):
        order = rng.permutation(n_train)
        for start in range(0, n_train, batch_size):
            indices = order[start:start + batch_size]
            probabilities = model.forward(x_train[indices])
            model.backward(loss.backward(probabilities, y_train[indices]))
            model.step(learning_rate)
            updates += 1
        if epoch % eval_freq == 0 or epoch == max_steps:
            evaluate(epoch)
    return model, history, data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dnn_hidden_units', default=DNN_HIDDEN_UNITS_DEFAULT)
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT)
    parser.add_argument('--max_steps', type=int, default=MAX_EPOCHS_DEFAULT,
                        help='Complete epochs, not individual updates')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT)
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Omit for full batch; use 1 for SGD')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--noise', type=float, default=0.0)
    parser.add_argument('--output', type=Path, help='Optional JSON history path')
    args = parser.parse_args()
    _, history, _ = train(args.dnn_hidden_units, args.learning_rate,
                          args.max_steps, args.eval_freq, args.batch_size,
                          args.seed, args.noise)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(history, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
