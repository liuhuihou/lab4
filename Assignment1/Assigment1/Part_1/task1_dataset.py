"""Generate the two-class Gaussian dataset required by Assignment 1, Part I."""

from pathlib import Path

import numpy as np


def generate_gaussian_dataset(seed=42, mean_distance=4.0, variance=1.0):
    """Return the required 160 training and 40 test samples.

    Each class is sampled from a two-dimensional Gaussian distribution.  The
    labels are -1 and +1 so that the returned arrays can be passed directly to
    the perceptron used in the following tasks.

    Args:
        seed: Random seed used to make the experiment reproducible.
        mean_distance: Difference between class means in EACH coordinate.
        variance: Shared variance in each coordinate (covariance = variance * I).

    Returns:
        X_train: Array with shape (160, 2).
        y_train: Array with shape (160,).
        X_test: Array with shape (40, 2).
        y_test: Array with shape (40,).
    """
    rng = np.random.default_rng(seed)

    if mean_distance < 0 or variance <= 0:
        raise ValueError('mean_distance must be nonnegative and variance positive')
    negative_mean = np.full(2, -mean_distance / 2)
    positive_mean = np.full(2, mean_distance / 2)
    negative_covariance = np.eye(2) * variance
    positive_covariance = np.eye(2) * variance

    negative_samples = rng.multivariate_normal(
        negative_mean, negative_covariance, size=100
    )
    positive_samples = rng.multivariate_normal(
        positive_mean, positive_covariance, size=100
    )

    # Shuffle each class independently before taking 80 training and 20 test
    # samples. This guarantees that both splits contain the required number
    # of examples from each class.
    negative_samples = negative_samples[rng.permutation(100)]
    positive_samples = positive_samples[rng.permutation(100)]

    X_train = np.vstack((negative_samples[:80], positive_samples[:80]))
    y_train = np.concatenate(
        (-np.ones(80, dtype=int), np.ones(80, dtype=int))
    )

    X_test = np.vstack((negative_samples[80:], positive_samples[80:]))
    y_test = np.concatenate(
        (-np.ones(20, dtype=int), np.ones(20, dtype=int))
    )

    # Mix the two classes after the stratified split.
    train_order = rng.permutation(X_train.shape[0])
    test_order = rng.permutation(X_test.shape[0])

    return (
        X_train[train_order],
        y_train[train_order],
        X_test[test_order],
        y_test[test_order],
    )


def plot_dataset(X_train, y_train, X_test, y_test, output_path):
    """Plot the training and test samples and save the figure."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 6))
    colors = {-1: "royalblue", 1: "tomato"}

    for label in (-1, 1):
        axis.scatter(
            X_train[y_train == label, 0],
            X_train[y_train == label, 1],
            color=colors[label],
            alpha=0.75,
            label=f"Class {label:+d} - train",
        )
        axis.scatter(
            X_test[y_test == label, 0],
            X_test[y_test == label, 1],
            facecolors="none",
            edgecolors=colors[label],
            linewidths=1.5,
            label=f"Class {label:+d} - test",
        )

    axis.set_xlabel("x1")
    axis.set_ylabel("x2")
    axis.set_title("Samples from two Gaussian distributions")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main():
    X_train, y_train, X_test, y_test = generate_gaussian_dataset()

    # These checks make accidental changes to the required class counts clear.
    assert X_train.shape == (160, 2)
    assert X_test.shape == (40, 2)
    assert np.sum(y_train == -1) == np.sum(y_train == 1) == 80
    assert np.sum(y_test == -1) == np.sum(y_test == 1) == 20

    output_path = Path(__file__).with_name("task1_dataset.png")
    plot_dataset(X_train, y_train, X_test, y_test, output_path)

    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    print("Training samples per class: 80")
    print("Test samples per class: 20")
    print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    main()
