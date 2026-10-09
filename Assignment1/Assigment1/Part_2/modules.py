"""NumPy layers using row-major batches; no automatic differentiation."""
import numpy as np


class Linear:
    def __init__(self, in_features, out_features, rng=None):
        """Small N(0, 0.1**2) weights and zero biases."""
        rng = np.random.default_rng() if rng is None else rng
        self.params = {
            'weight': rng.normal(0.0, 0.1, (in_features, out_features)),
            'bias': np.zeros(out_features),
        }
        self.grads = {k: np.zeros_like(v) for k, v in self.params.items()}

    def forward(self, x):
        self.x = x
        return x @ self.params['weight'] + self.params['bias']

    def backward(self, dout):
        self.grads['weight'] = self.x.T @ dout
        self.grads['bias'] = dout.sum(axis=0)
        return dout @ self.params['weight'].T


class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return np.maximum(x, 0)

    def backward(self, dout):
        return dout * self.mask


class SoftMax:
    def forward(self, x):
        shifted = x - x.max(axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        self.probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
        return self.probabilities

    def backward(self, dout):
        """Jacobian-vector product for an upstream probability gradient.
        Training uses fused CrossEntropy.backward and skips this operation.
        """
        p = self.probabilities
        return p * (dout - (dout * p).sum(axis=1, keepdims=True))


class CrossEntropy:
    def forward(self, x, y):
        """Mean cross entropy of probabilities x and one-hot targets y."""
        return float(-np.sum(y * np.log(np.maximum(x, np.finfo(float).tiny))) / len(x))

    def backward(self, x, y):
        """Fused softmax + mean cross entropy gradient w.r.t. LOGITS.
        Pass directly to MLP.backward; do not apply SoftMax.backward again.
        The batch normalization occurs here exactly once.
        """
        return (x - y) / len(x)
