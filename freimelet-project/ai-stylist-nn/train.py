# -*- coding: utf-8 -*-
"""
End-to-end training entry point.

    python3 train.py

Generates the synthetic image set, extracts features with the exact
same code path used in the browser demo, trains the MLP with manual
backprop + Adam, prints train/val accuracy, and writes:

  - weights.json         (raw network weights, python-side)
  - web_model.json        (weights + normalisation stats + style
                            metadata, ready to be inlined into the
                            static site's JavaScript)
"""
import json

import numpy as np

from dataset import generate_dataset
from features import extract_features
from model import MLP, cross_entropy, one_hot
from styles import STYLES, NUM_CLASSES, NUM_FEATURES, FEATURE_NAMES


def build_feature_matrix(images):
    feats = np.zeros((len(images), NUM_FEATURES))
    for i, img in enumerate(images):
        feats[i] = extract_features(img)
    return feats


def main():
    print("Generating synthetic training images...")
    images, labels = generate_dataset(samples_per_class=450, size=96, seed=7)
    print(f"  {len(images)} images across {NUM_CLASSES} classes")

    print("Extracting hand-written visual features...")
    X = build_feature_matrix(images)
    y = labels

    # normalise features to zero mean / unit variance (stats saved for inference)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0) + 1e-8
    Xn = (X - mu) / sigma

    # shuffle + split
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(Xn))
    Xn, y = Xn[idx], y[idx]
    n_val = int(0.15 * len(Xn))
    X_val, y_val = Xn[:n_val], y[:n_val]
    X_train, y_train = Xn[n_val:], y[n_val:]

    net = MLP(NUM_FEATURES, 32, 16, NUM_CLASSES, seed=42)
    y_train_oh = one_hot(y_train, NUM_CLASSES)

    epochs = 400
    batch_size = 64
    n = X_train.shape[0]
    print("Training...")
    for epoch in range(epochs):
        perm = rng.permutation(n)
        X_train, y_train_oh, y_train = X_train[perm], y_train_oh[perm], y_train[perm]
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            xb = X_train[start:start + batch_size]
            yb = y_train_oh[start:start + batch_size]
            out, cache = net.forward(xb)
            loss = cross_entropy(out, yb)
            epoch_loss += loss * xb.shape[0]
            grads = net.backward(cache, yb)
            net.adam_step(grads, lr=0.008)
        if (epoch + 1) % 50 == 0 or epoch == 0:
            train_pred = net.predict_proba(X_train).argmax(axis=1)
            train_acc = (train_pred == y_train).mean()
            val_pred = net.predict_proba(X_val).argmax(axis=1)
            val_acc = (val_pred == y_val).mean()
            print(f"  epoch {epoch + 1:4d}  loss={epoch_loss / n:.4f}  "
                  f"train_acc={train_acc:.3f}  val_acc={val_acc:.3f}")

    val_pred = net.predict_proba(X_val).argmax(axis=1)
    val_acc = (val_pred == y_val).mean()
    print(f"Final validation accuracy: {val_acc:.3f}")

    # ---- export ----
    weights = net.to_json()
    with open("weights.json", "w", encoding="utf-8") as f:
        json.dump(weights, f)
    print("Saved weights.json")

    web_model = {
        "meta": {
            "architecture": [NUM_FEATURES, 32, 16, NUM_CLASSES],
            "feature_names": FEATURE_NAMES,
            "val_accuracy": round(float(val_acc), 4),
        },
        "norm": {"mu": mu.tolist(), "sigma": sigma.tolist()},
        "weights": weights,
        "styles": STYLES,
    }
    with open("web_model.json", "w", encoding="utf-8") as f:
        json.dump(web_model, f, ensure_ascii=False)
    print("Saved web_model.json (ready to inline into the site)")


if __name__ == "__main__":
    main()
