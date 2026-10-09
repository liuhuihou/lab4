"""Arbitrary-depth NumPy MLP with ReLU hidden layers and softmax output."""
import numpy as np
try:
    from .modules import Linear, ReLU, SoftMax
except ImportError:
    from modules import Linear, ReLU, SoftMax


class MLP:
    def __init__(self, n_inputs, n_hidden, n_classes, seed=42):
        dimensions = [n_inputs, *n_hidden, n_classes]
        if any(not isinstance(d, (int, np.integer)) or d <= 0 for d in dimensions):
            raise ValueError('All layer dimensions must be positive integers')
        rng = np.random.default_rng(seed)
        self.layers = []
        self.linear_layers = []
        for i, (incoming, outgoing) in enumerate(zip(dimensions[:-1], dimensions[1:])):
            layer = Linear(incoming, outgoing, rng)
            self.layers.append(layer)
            self.linear_layers.append(layer)
            if i < len(dimensions) - 2:
                self.layers.append(ReLU())
        self.softmax = SoftMax()

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return self.softmax.forward(x)

    def backward(self, dout):
        """Backpropagate a gradient w.r.t. logits from fused cross entropy."""
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def step(self, learning_rate):
        for layer in self.linear_layers:
            for key in layer.params:
                layer.params[key] -= learning_rate * layer.grads[key]
