"""Run with python -m unittest discover -s tests -v from the submission root."""
import unittest
import numpy as np
from Part_1.perceptron import Perceptron
from Part_1.task1_dataset import generate_gaussian_dataset
from Part_2.modules import SoftMax, CrossEntropy, ReLU
from Part_2.mlp_numpy import MLP
from Part_2.train_mlp_numpy import make_dataset, train


class ModelTests(unittest.TestCase):
    def test_gaussian_split_and_reproducibility(self):
        data = generate_gaussian_dataset()
        self.assertEqual(data[0].shape, (160, 2))
        self.assertEqual(data[2].shape, (40, 2))
        for y, count in [(data[1], 80), (data[3], 20)]:
            np.testing.assert_array_equal(np.unique(y, return_counts=True)[1], [count, count])
        for a, b in zip(data, generate_gaussian_dataset()):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(len(np.unique(np.vstack((data[0], data[2])), axis=0)), 200)

    def test_perceptron_one_batch_update(self):
        x = np.array([[1., 0.], [3., 2.], [9., 9.]])
        y = np.array([-1, -1, 1])
        model = Perceptron(2, max_epochs=1, learning_rate=0.1).train(x, y)
        np.testing.assert_allclose(model.weights, [-0.2, -0.1, -0.1])
        self.assertEqual(model.forward(x[0]), model.forward(x)[0])

    def test_perceptron_convergence_and_epoch_limit(self):
        x, y, _, _ = generate_gaussian_dataset()
        model = Perceptron(2).train(x, y)
        self.assertTrue(np.all(model.forward(x) == y))
        model = Perceptron(2, max_epochs=7).train(np.zeros((2, 2)), [-1, 1])
        self.assertEqual(len(model.errors_per_epoch), 7)

    def test_moons_split(self):
        x, y, xt, yt = make_dataset()
        self.assertEqual(x.shape, (800, 2))
        self.assertEqual(xt.shape, (200, 2))
        np.testing.assert_array_equal(y.sum(axis=0), [400, 400])
        np.testing.assert_array_equal(yt.sum(axis=0), [100, 100])
        self.assertEqual(len(np.unique(np.vstack((x, xt)), axis=0)), 1000)

    def test_softmax_stability_and_jacobian(self):
        sm = SoftMax()
        x = np.array([[1000., 1001., 999.], [-1000., -1001., -999.]])
        p = sm.forward(x)
        np.testing.assert_allclose(p.sum(axis=1), 1)
        self.assertTrue(np.all(np.isfinite(p)))
        g = np.array([[0.1, -0.3, 0.4], [-0.2, 0.7, 0.6]])
        analytic = sm.backward(g)
        numeric = np.zeros_like(x)
        for index in np.ndindex(x.shape):
            plus, minus = x.copy(), x.copy()
            plus[index] += 1e-5
            minus[index] -= 1e-5
            numeric[index] = np.sum((sm.forward(plus) - sm.forward(minus)) * g) / 2e-5
        np.testing.assert_allclose(analytic, numeric, atol=1e-8)

    def test_relu(self):
        relu = ReLU()
        np.testing.assert_array_equal(relu.forward(np.array([[-2., 0., 3.]])), [[0., 0., 3.]])
        np.testing.assert_array_equal(relu.backward(np.ones((1, 3))), [[0., 0., 1.]])

    def test_full_network_numerical_gradients(self):
        # Check every parameter of a two-hidden-layer network, including biases.
        model = MLP(2, [4, 3], 2, seed=9)
        x = np.random.default_rng(7).normal(size=(5, 2))
        y = np.eye(2)[[0, 1, 1, 0, 1]]
        loss = CrossEntropy()
        p = model.forward(x)
        model.backward(loss.backward(p, y))
        max_error = 0.0
        for layer in model.linear_layers:
            for key, parameter in layer.params.items():
                analytic = layer.grads[key].copy()
                numeric = np.zeros_like(parameter)
                for index in np.ndindex(parameter.shape):
                    old = parameter[index]
                    parameter[index] = old + 1e-6
                    plus = loss.forward(model.forward(x), y)
                    parameter[index] = old - 1e-6
                    minus = loss.forward(model.forward(x), y)
                    parameter[index] = old
                    numeric[index] = (plus - minus) / 2e-6
                max_error = max(max_error, float(np.max(np.abs(analytic - numeric))))
                np.testing.assert_allclose(analytic, numeric, atol=2e-7, rtol=2e-4)
        print(f'\nMaximum absolute network gradient error: {max_error:.3e}')

    def test_batch_modes_remainder_and_no_test_leakage(self):
        data = make_dataset()
        for size in (1, 128, 800):
            model, history, _ = train(max_steps=2, batch_size=size, verbose=False, data=data)
            self.assertEqual(history[-1]['updates'], 2 * int(np.ceil(800 / size)))
            self.assertLess(history[-1]['train_loss'], history[0]['train_loss'])
        changed_test = (*data[:2], data[2] + 100, data[3][:, ::-1])
        a, _, _ = train(max_steps=2, verbose=False, data=data)
        b, _, _ = train(max_steps=2, verbose=False, data=changed_test)
        for la, lb in zip(a.linear_layers, b.linear_layers):
            for key in la.params:
                np.testing.assert_array_equal(la.params[key], lb.params[key])


if __name__ == '__main__':
    unittest.main()
