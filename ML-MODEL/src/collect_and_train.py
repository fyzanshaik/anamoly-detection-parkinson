import os
import time
import numpy as np

FEATURE_DIM   = 10
HIDDEN_DIM    = 5
CLASS_H1      = 8
CLASS_H2      = 4
NUM_CLASSES   = 4
N_NORMAL      = 600
N_FAULT_EACH  = 400
LR_AE         = 0.01
LR_CLS        = 0.01
EPOCHS_AE     = 500
EPOCHS_CLS    = 400
BATCH_SIZE    = 32

SEED_BASELINE  = np.array([-9.15, 0.55, 2.35, 0.06, 0.07, 0.08, 9.47, 0.05, 9.55, 9.47], dtype=np.float32)
FEATURE_STD    = np.array([ 0.10, 0.10, 0.10, 0.04, 0.04, 0.04, 0.12, 0.04, 0.20, 0.12], dtype=np.float32)
MOTOR_DRIFT    = np.array([ 0.00, 0.00, 0.00, 0.30, 0.30, 0.30, 2.50, 0.80, 4.00, 2.00], dtype=np.float32)
FAULT_PATTERNS = np.array([
    [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
    [0.040, 0.020, 0.030, 0.050, 0.040, 0.045, 0.160, 0.070, 0.240, 0.140],
    [0.020, 0.030, 0.020, 0.030, 0.025, 0.050, 0.080, 0.055, 0.280, 0.090],
    [0.030, 0.040, 0.030, 0.080, 0.070, 0.075, 0.100, 0.110, 0.360, 0.110],
], dtype=np.float32)

def relu(x):      return np.maximum(0, x)
def relu_d(x):    return (x > 0).astype(np.float32)
def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def generate_samples(fault_type, n, speed_frac=0.39, intensity=1.0):
    rng = np.random.default_rng(42 + fault_type)
    base  = SEED_BASELINE + MOTOR_DRIFT * speed_frac
    fp    = FAULT_PATTERNS[fault_type] * intensity
    noise = rng.normal(0, FEATURE_STD * 0.3, size=(n, FEATURE_DIM)).astype(np.float32)
    return (base + fp + noise).astype(np.float32)

def normalize(X, mean=None, std=None):
    if mean is None: mean = X.mean(axis=0)
    if std  is None: std  = X.std(axis=0) + 1e-6
    return (X - mean) / std, mean, std

def batches(X, y=None, size=BATCH_SIZE):
    idx = np.random.permutation(len(X))
    for i in range(0, len(X), size):
        b = idx[i:i+size]
        yield (X[b], y[b]) if y is not None else (X[b],)

def train_autoencoder():
    rng = np.random.default_rng(0)
    X_raw = generate_samples(0, N_NORMAL, speed_frac=0.39, intensity=0.0)
    X, mean, std = normalize(X_raw)

    W1 = rng.normal(0, 0.1, (FEATURE_DIM, HIDDEN_DIM)).astype(np.float32)
    b1 = np.zeros(HIDDEN_DIM, dtype=np.float32)
    W2 = rng.normal(0, 0.1, (HIDDEN_DIM, FEATURE_DIM)).astype(np.float32)
    b2 = np.zeros(FEATURE_DIM, dtype=np.float32)

    for epoch in range(EPOCHS_AE):
        for (xb,) in batches(X):
            h  = relu(xb @ W1 + b1)
            o  = h @ W2 + b2
            d  = o - xb
            dW2 = h.T @ d / len(xb)
            db2 = d.mean(axis=0)
            dh  = d @ W2.T * relu_d(xb @ W1 + b1)
            dW1 = xb.T @ dh / len(xb)
            db1 = dh.mean(axis=0)
            W1 -= LR_AE * dW1;  b1 -= LR_AE * db1
            W2 -= LR_AE * dW2;  b2 -= LR_AE * db2

    h      = relu(X @ W1 + b1)
    o      = h @ W2 + b2
    errors = np.mean((X - o) ** 2, axis=1)
    thr    = float(np.percentile(errors, 95))
    print(f"[autoencoder] mse={errors.mean():.5f}  threshold={thr:.5f}")
    return W1, b1, W2, b2, thr, mean, std

def train_classifier():
    rng = np.random.default_rng(1)
    parts, labels = [], []
    for ft in range(NUM_CLASSES):
        s = generate_samples(ft, N_FAULT_EACH,
                             speed_frac=0.39 if ft==0 else 1.0,
                             intensity=0.0 if ft==0 else 1.0)
        parts.append(s)
        labels.append(np.full(N_FAULT_EACH, ft, dtype=np.int32))
    X_raw = np.vstack(parts)
    y     = np.concatenate(labels)
    X, mean, std = normalize(X_raw)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    W1 = rng.normal(0, 0.1, (FEATURE_DIM, CLASS_H1)).astype(np.float32)
    b1 = np.zeros(CLASS_H1, dtype=np.float32)
    W2 = rng.normal(0, 0.1, (CLASS_H1, CLASS_H2)).astype(np.float32)
    b2 = np.zeros(CLASS_H2, dtype=np.float32)
    W3 = rng.normal(0, 0.1, (CLASS_H2, NUM_CLASSES)).astype(np.float32)
    b3 = np.zeros(NUM_CLASSES, dtype=np.float32)

    for epoch in range(EPOCHS_CLS):
        for xb, yb in batches(X, y):
            h1 = relu(xb @ W1 + b1)
            h2 = relu(h1 @ W2 + b2)
            p  = softmax(h2 @ W3 + b3)
            oh = np.zeros_like(p); oh[np.arange(len(yb)), yb] = 1
            d3 = (p - oh) / len(xb)
            dW3 = h2.T @ d3;          db3 = d3.sum(axis=0)
            d2  = d3 @ W3.T * relu_d(h1 @ W2 + b2)
            dW2 = h1.T @ d2;          db2 = d2.sum(axis=0)
            d1  = d2 @ W2.T * relu_d(xb @ W1 + b1)
            dW1 = xb.T @ d1;          db1 = d1.sum(axis=0)
            W1 -= LR_CLS*dW1; b1 -= LR_CLS*db1
            W2 -= LR_CLS*dW2; b2 -= LR_CLS*db2
            W3 -= LR_CLS*dW3; b3 -= LR_CLS*db3

    h1   = relu(X @ W1 + b1)
    h2   = relu(h1 @ W2 + b2)
    pred = (softmax(h2 @ W3 + b3)).argmax(axis=1)
    acc  = (pred == y).mean()
    print(f"[classifier]  accuracy={acc:.4f}")
    return W1, b1, W2, b2, W3, b3, mean, std

def arr1d(arr, name):
    vals = ", ".join(f"{v:.6f}f" for v in arr.ravel())
    return f"const float {name}[{arr.size}] = {{{vals}}};\n"

def arr2d(arr, name):
    r, c  = arr.shape
    rows  = ["  {" + ", ".join(f"{v:.6f}f" for v in row) + "}" for row in arr]
    return f"const float {name}[{r}][{c}] = {{\n" + ",\n".join(rows) + "\n};\n"

def export_autoencoder(W1, b1, W2, b2, thr, mean, std, path):
    with open(path, "w") as f:
        f.write("#pragma once\n")
        f.write(f"// Autoencoder {FEATURE_DIM}→{HIDDEN_DIM}→{FEATURE_DIM}  threshold={thr:.6f}\n\n")
        f.write(arr2d(W1,   "encoder_weights"))
        f.write(arr1d(b1,   "encoder_bias"))
        f.write(arr2d(W2.T, "decoder_weights"))
        f.write(arr1d(b2,   "decoder_bias"))
        f.write(arr1d(mean, "ae_norm_mean"))
        f.write(arr1d(std,  "ae_norm_std"))
        f.write(f"\nconst float AE_THRESHOLD = {thr:.6f}f;\n")
    print(f"[export] {path}")

def export_classifier(W1, b1, W2, b2, W3, b3, mean, std, path):
    with open(path, "w") as f:
        f.write("#pragma once\n")
        f.write(f"// Classifier {FEATURE_DIM}→{CLASS_H1}→{CLASS_H2}→{NUM_CLASSES}\n\n")
        f.write(arr2d(W1, "cls_w1")); f.write(arr1d(b1, "cls_b1"))
        f.write(arr2d(W2, "cls_w2")); f.write(arr1d(b2, "cls_b2"))
        f.write(arr2d(W3, "cls_w3")); f.write(arr1d(b3, "cls_b3"))
        f.write(arr1d(mean, "cls_norm_mean"))
        f.write(arr1d(std,  "cls_norm_std"))
    print(f"[export] {path}")

def main():
    base = os.path.join(os.path.dirname(__file__), "..", "..")
    print("[train] Autoencoder...")
    W1, b1, W2, b2, thr, ae_mean, ae_std = train_autoencoder()
    export_autoencoder(W1, b1, W2, b2, thr, ae_mean, ae_std,
                       os.path.join(base, "src", "model_weights.h"))
    print("[train] Classifier...")
    cW1,cb1,cW2,cb2,cW3,cb3,cm,cs = train_classifier()
    export_classifier(cW1,cb1,cW2,cb2,cW3,cb3,cm,cs,
                      os.path.join(base, "src", "classifier_weights.h"))

if __name__ == "__main__":
    main()
