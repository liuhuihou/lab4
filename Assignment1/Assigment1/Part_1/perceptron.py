import numpy as np


class Perceptron(object):

    def __init__(self, n_inputs, max_epochs=100, learning_rate=0.01):
        """
        Initializes the perceptron object.
        - n_inputs: Number of inputs.
        - max_epochs: Maximum number of training cycles.
        - learning_rate: Magnitude of weight changes at each training cycle.
        - weights: Initialize weights (including bias).
        """
        if n_inputs <= 0:
            raise ValueError("n_inputs must be positive")
        if max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        self.n_inputs = n_inputs
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.weights = np.zeros(n_inputs + 1, dtype=float)
        self.errors_per_epoch = []

    def forward(self, input_vec):
        """
        Predicts label from input.
        Args:
            input_vec (np.ndarray): One sample with shape (n_inputs,) or a
                batch with shape (n_samples, n_inputs).
        Returns:
            int or np.ndarray: Predicted label(s), either 1 or -1.
        """
        inputs = np.asarray(input_vec, dtype=float)

        if inputs.ndim == 1:
            if inputs.shape[0] != self.n_inputs:
                raise ValueError(
                    f"Expected {self.n_inputs} features, got {inputs.shape[0]}"
                )
            score = inputs @ self.weights[:-1] + self.weights[-1]
            return 1 if score >= 0 else -1

        if inputs.ndim == 2:
            if inputs.shape[1] != self.n_inputs:
                raise ValueError(
                    f"Expected {self.n_inputs} features, got {inputs.shape[1]}"
                )
            scores = inputs @ self.weights[:-1] + self.weights[-1]
            return np.where(scores >= 0, 1, -1)

        raise ValueError("input_vec must be a one- or two-dimensional array")

    def train(self, training_inputs, labels):
        """
        Trains the perceptron using batch gradient descent.

        One update is performed per epoch using the mean gradient over all
        currently misclassified samples, as specified in the tutorial.

        Args:
            training_inputs (np.ndarray): Training data with shape
                (n_samples, n_inputs).
            labels (np.ndarray): Labels with shape (n_samples,), containing
                only -1 and 1.

        Returns:
            Perceptron: The trained instance.
        """
        inputs = np.asarray(training_inputs, dtype=float)
        targets = np.asarray(labels)

        if inputs.ndim != 2 or inputs.shape[1] != self.n_inputs or len(inputs) == 0:
            raise ValueError(
                f"training_inputs must have shape (n_samples, {self.n_inputs})"
            )
        if targets.ndim != 1 or targets.shape[0] != inputs.shape[0]:
            raise ValueError("labels must contain one value for every sample")
        if not np.all(np.isin(targets, (-1, 1))):
            raise ValueError("labels must contain only -1 and 1")

        self.errors_per_epoch = []

        for _ in range(self.max_epochs):
            predictions = self.forward(inputs)
            misclassified = predictions * targets < 0
            n_errors = int(np.sum(misclassified))
            self.errors_per_epoch.append(n_errors)

            # A linearly separable dataset has converged when no sample is
            # misclassified, so no further updates are necessary.
            if n_errors == 0:
                break

            wrong_inputs = inputs[misclassified]
            wrong_labels = targets[misclassified]

            # Append a constant feature so the bias is updated together with
            # the ordinary weights. The tutorial defines the gradient as the
            # negative mean of x_i * y_i over misclassified samples.
            augmented_inputs = np.column_stack(
                (wrong_inputs, np.ones(n_errors, dtype=float))
            )
            gradient = -np.mean(
                augmented_inputs * wrong_labels[:, np.newaxis], axis=0
            )
            self.weights -= self.learning_rate * gradient

        return self
