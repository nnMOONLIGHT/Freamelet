# -*- coding: utf-8 -*-
"""
A plain multilayer perceptron, implemented from first principles.

No PyTorch, no TensorFlow, no scikit-learn estimator - just numpy arrays,
matrix multiplication, and manually derived gradients. This is the
"from-scratch" part of the assignment: every line of forward and
backward computation is written out below.
"""
import numpy as np


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    return (x > 0).astype(x.dtype)


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


class MLP:
    """input -> Dense(h1, relu) -> Dense(h2, relu) -> Dense(out, softmax)"""

    def __init__(self, n_in, n_hidden1, n_hidden2, n_out, seed=42):
        rng = np.random.default_rng(seed)
        # He initialisation, sized for ReLU layers
        self.W1 = rng.normal(0, np.sqrt(2.0 / n_in), (n_in, n_hidden1))
        self.b1 = np.zeros(n_hidden1)
        self.W2 = rng.normal(0, np.sqrt(2.0 / n_hidden1), (n_hidden1, n_hidden2))
        self.b2 = np.zeros(n_hidden2)
        self.W3 = rng.normal(0, np.sqrt(2.0 / n_hidden2), (n_hidden2, n_out))
        self.b3 = np.zeros(n_out)

        # Adam optimizer moment buffers
        self._params = ["W1", "b1", "W2", "b2", "W3", "b3"]
        self.m = {p: np.zeros_like(getattr(self, p)) for p in self._params}
        self.v = {p: np.zeros_like(getattr(self, p)) for p in self._params}
        self.t = 0

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = relu(z2)
        z3 = a2 @ self.W3 + self.b3
        out = softmax(z3)
        cache = (X, z1, a1, z2, a2, z3, out)
        return out, cache

    def predict_proba(self, X):
        out, _ = self.forward(X)
        return out

    def backward(self, cache, y_onehot):
        X, z1, a1, z2, a2, z3, out = cache
        n = X.shape[0]

        # dL/dz3 for softmax + cross-entropy simplifies to (out - y)
        dz3 = (out - y_onehot) / n
        dW3 = a2.T @ dz3
        db3 = dz3.sum(axis=0)

        da2 = dz3 @ self.W3.T
        dz2 = da2 * relu_grad(z2)
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * relu_grad(z1)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)

        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}

    def adam_step(self, grads, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=1e-4):
        self.t += 1
        for p in self._params:
            g = grads[p] + weight_decay * getattr(self, p)
            self.m[p] = beta1 * self.m[p] + (1 - beta1) * g
            self.v[p] = beta2 * self.v[p] + (1 - beta2) * (g ** 2)
            m_hat = self.m[p] / (1 - beta1 ** self.t)
            v_hat = self.v[p] / (1 - beta2 ** self.t)
            update = lr * m_hat / (np.sqrt(v_hat) + eps)
            setattr(self, p, getattr(self, p) - update)

    def to_json(self):
        return {p: getattr(self, p).tolist() for p in self._params}

    @classmethod
    def from_json(cls, data, n_in, n_hidden1, n_hidden2, n_out):
        m = cls(n_in, n_hidden1, n_hidden2, n_out)
        for p in m._params:
            setattr(m, p, np.array(data[p]))
        return m


def cross_entropy(probs, y_onehot):
    eps = 1e-9
    return -np.mean(np.sum(y_onehot * np.log(probs + eps), axis=1))


def one_hot(y, n_classes):
    out = np.zeros((y.shape[0], n_classes))
    out[np.arange(y.shape[0]), y] = 1
    return out
