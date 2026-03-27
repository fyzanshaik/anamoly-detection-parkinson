"""
Visualization for the actual deployed model on ESP32.
Generates charts based on collect_and_train.py data and model_weights.h.
Run from repo root: python3 ML-MODEL/src/visualize_deployed.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── replicate exact training config ──────────────────────────────────────────
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

FEATURE_NAMES  = ['mean_x','mean_y','mean_z','std_x','std_y','std_z',
                  'mag_mean','mag_std','mag_max','mag_rms']
FAULT_NAMES    = ['Normal','Imbalance','Bearing','Looseness']
PHASE_NAMES    = ['Normal','Ramp','Anomaly','Resolve']

STYLE = {
    'bg':     '#0a0a0a',
    'card':   '#141414',
    'border': '#2a2a2a',
    'text':   '#e0e0e0',
    'dim':    '#666666',
    'green':  '#00ff88',
    'red':    '#ff4444',
    'yellow': '#ffaa00',
    'blue':   '#4488ff',
    'purple': '#aa44ff',
    'colors': ['#00ff88','#ff4444','#ffaa00','#4488ff'],
}

def apply_style(fig, axes=None):
    fig.patch.set_facecolor(STYLE['bg'])
    if axes is None:
        return
    for ax in (axes if hasattr(axes, '__iter__') else [axes]):
        ax.set_facecolor(STYLE['card'])
        ax.tick_params(colors=STYLE['dim'], labelsize=8)
        ax.xaxis.label.set_color(STYLE['text'])
        ax.yaxis.label.set_color(STYLE['text'])
        ax.title.set_color(STYLE['green'])
        for spine in ax.spines.values():
            spine.set_edgecolor(STYLE['border'])

def relu(x):      return np.maximum(0, x)
def relu_d(x):    return (x > 0).astype(np.float32)
def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def generate_samples(fault_type, n, speed_frac=0.39, intensity=1.0):
    rng  = np.random.default_rng(42 + fault_type)
    base = SEED_BASELINE + MOTOR_DRIFT * speed_frac
    fp   = FAULT_PATTERNS[fault_type] * intensity
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
    rng   = np.random.default_rng(0)
    X_raw = generate_samples(0, N_NORMAL, speed_frac=0.39, intensity=0.0)
    X, mean, std = normalize(X_raw)
    W1 = rng.normal(0, 0.1, (FEATURE_DIM, HIDDEN_DIM)).astype(np.float32)
    b1 = np.zeros(HIDDEN_DIM, dtype=np.float32)
    W2 = rng.normal(0, 0.1, (HIDDEN_DIM, FEATURE_DIM)).astype(np.float32)
    b2 = np.zeros(FEATURE_DIM, dtype=np.float32)
    losses = []
    for epoch in range(EPOCHS_AE):
        epoch_loss = 0
        count = 0
        for (xb,) in batches(X):
            h  = relu(xb @ W1 + b1)
            o  = h @ W2 + b2
            d  = o - xb
            loss = np.mean(d**2)
            epoch_loss += loss; count += 1
            dW2 = h.T @ d / len(xb);  db2 = d.mean(axis=0)
            dh  = d @ W2.T * relu_d(xb @ W1 + b1)
            dW1 = xb.T @ dh / len(xb); db1 = dh.mean(axis=0)
            W1 -= LR_AE * dW1; b1 -= LR_AE * db1
            W2 -= LR_AE * dW2; b2 -= LR_AE * db2
        losses.append(epoch_loss / count)
    h      = relu(X @ W1 + b1)
    o      = h @ W2 + b2
    errors = np.mean((X - o) ** 2, axis=1)
    thr    = float(np.percentile(errors, 95))
    return W1, b1, W2, b2, thr, mean, std, losses

def train_classifier():
    rng = np.random.default_rng(1)
    parts, labels = [], []
    for ft in range(NUM_CLASSES):
        s = generate_samples(ft, N_FAULT_EACH,
                             speed_frac=0.39 if ft==0 else 1.0,
                             intensity=0.0 if ft==0 else 1.0)
        parts.append(s); labels.append(np.full(N_FAULT_EACH, ft, dtype=np.int32))
    X_raw = np.vstack(parts); y = np.concatenate(labels)
    X, mean, std = normalize(X_raw)
    idx = rng.permutation(len(X)); X, y = X[idx], y[idx]
    W1 = rng.normal(0, 0.1, (FEATURE_DIM, CLASS_H1)).astype(np.float32)
    b1 = np.zeros(CLASS_H1, dtype=np.float32)
    W2 = rng.normal(0, 0.1, (CLASS_H1, CLASS_H2)).astype(np.float32)
    b2 = np.zeros(CLASS_H2, dtype=np.float32)
    W3 = rng.normal(0, 0.1, (CLASS_H2, NUM_CLASSES)).astype(np.float32)
    b3 = np.zeros(NUM_CLASSES, dtype=np.float32)
    losses = []
    for epoch in range(EPOCHS_CLS):
        epoch_loss = 0; count = 0
        for xb, yb in batches(X, y):
            h1 = relu(xb @ W1 + b1); h2 = relu(h1 @ W2 + b2)
            p  = softmax(h2 @ W3 + b3)
            oh = np.zeros_like(p); oh[np.arange(len(yb)), yb] = 1
            loss = -np.mean(oh * np.log(p + 1e-9)); epoch_loss += loss; count += 1
            d3 = (p - oh) / len(xb)
            dW3 = h2.T @ d3; db3 = d3.sum(axis=0)
            d2  = d3 @ W3.T * relu_d(h1 @ W2 + b2)
            dW2 = h1.T @ d2; db2 = d2.sum(axis=0)
            d1  = d2 @ W2.T * relu_d(xb @ W1 + b1)
            dW1 = xb.T @ d1; db1 = d1.sum(axis=0)
            W1 -= LR_CLS*dW1; b1 -= LR_CLS*db1
            W2 -= LR_CLS*dW2; b2 -= LR_CLS*db2
            W3 -= LR_CLS*dW3; b3 -= LR_CLS*db3
        losses.append(epoch_loss / count)
    h1   = relu(X @ W1 + b1); h2 = relu(h1 @ W2 + b2)
    pred = softmax(h2 @ W3 + b3).argmax(axis=1)
    acc  = (pred == y).mean()
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for t, p in zip(y, pred): cm[t, p] += 1
    return W1, b1, W2, b2, W3, b3, mean, std, losses, acc, cm, pred, y

def ae_score(feat, W1, b1, W2, b2, mean, std):
    x = (feat - mean) / std
    h = relu(x @ W1 + b1)
    o = h @ W2 + b2
    return np.mean((x - o)**2, axis=1)

def simulate_cycle(W1, b1, W2, b2, thr, mean, std):
    """Simulate one full 50s cycle at 1s resolution.
    Speed kept constant at 0.39 throughout — only fault intensity varies
    so the chart shows fault pattern effect, not motor speed effect."""
    phases = [
        (0,  25, 'Normal',  0, 0.0),
        (25, 30, 'Ramp',    1, 0.4),
        (30, 45, 'Anomaly', 2, 1.0),
        (45, 50, 'Resolve', 3, 0.2),
    ]
    times, scores, phase_ids = [], [], []
    rng = np.random.default_rng(99)
    for t_start, t_end, _, phase_id, intensity in phases:
        fault = 2  # bearing for cycle demo
        for t in range(t_start, t_end):
            base  = SEED_BASELINE + MOTOR_DRIFT * 0.39
            fp    = FAULT_PATTERNS[fault] * intensity
            noise = rng.normal(0, FEATURE_STD * 0.3, size=FEATURE_DIM).astype(np.float32)
            feat  = (base + fp + noise).reshape(1, -1)
            sc    = float(ae_score(feat, W1, b1, W2, b2, mean, std)[0])
            times.append(t); scores.append(sc); phase_ids.append(phase_id)
    return np.array(times), np.array(scores), np.array(phase_ids)

def model_sizes():
    ae_params  = FEATURE_DIM*HIDDEN_DIM + HIDDEN_DIM + HIDDEN_DIM*FEATURE_DIM + FEATURE_DIM
    ae_norm    = FEATURE_DIM * 2
    ae_total   = ae_params + ae_norm

    cls_params = FEATURE_DIM*CLASS_H1 + CLASS_H1 + CLASS_H1*CLASS_H2 + CLASS_H2 + CLASS_H2*NUM_CLASSES + NUM_CLASSES
    cls_norm   = FEATURE_DIM * 2
    cls_total  = cls_params + cls_norm

    return {
        'ae_weights': ae_params * 4,
        'ae_norm':    ae_norm   * 4,
        'cls_weights': cls_params * 4,
        'cls_norm':    cls_norm   * 4,
        'ae_total':   ae_total  * 4,
        'cls_total':  cls_total * 4,
        'total':      (ae_total + cls_total) * 4,
    }

# ─────────────────────────────────────────────────────────────────────────────
print("Training autoencoder...")
W1ae, b1ae, W2ae, b2ae, thr, ae_mean, ae_std, ae_losses = train_autoencoder()
print(f"  threshold={thr:.6f}")

print("Training classifier...")
W1c, b1c, W2c, b2c, W3c, b3c, cls_mean, cls_std, cls_losses, acc, cm, pred, y_true = train_classifier()
print(f"  accuracy={acc:.4f}")

out_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'deployed')
os.makedirs(out_dir, exist_ok=True)

# ── Fig 1: Anomaly Score Distribution ────────────────────────────────────────
print("Generating Fig 1: Score distribution...")
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4))
apply_style(fig, [ax, ax2])

# Same speed_frac for all — isolates fault pattern effect vs normal
normal_raw    = generate_samples(0, 500, speed_frac=0.39, intensity=0.0)
normal_scores = ae_score(normal_raw, W1ae, b1ae, W2ae, b2ae, ae_mean, ae_std)

fault_scores_all = {}
for ft in range(1, 4):
    raw = generate_samples(ft, 300, speed_frac=0.39, intensity=1.0)
    fault_scores_all[ft] = ae_score(raw, W1ae, b1ae, W2ae, b2ae, ae_mean, ae_std)

all_scores = np.concatenate([normal_scores] + list(fault_scores_all.values()))
xmax = np.percentile(all_scores, 99) * 1.1

for a in [ax, ax2]:
    a.hist(normal_scores, bins=40, alpha=0.75, color=STYLE['green'], label='Normal', density=True)
    for ft, sc in fault_scores_all.items():
        a.hist(sc, bins=40, alpha=0.6, color=STYLE['colors'][ft], label=FAULT_NAMES[ft], density=True)
    a.axvline(thr, color=STYLE['yellow'], linestyle='--', linewidth=1.5, label=f'Threshold ({thr:.3f})')
    a.legend(fontsize=7, facecolor=STYLE['card'], labelcolor=STYLE['text'], edgecolor=STYLE['border'])
    a.set_xlabel('Reconstruction Error (MSE)')

ax.set_ylabel('Density')
ax.set_title('Score Distribution (linear)')
ax.set_xlim(0, xmax)

ax2.set_yscale('log')
ax2.set_title('Score Distribution (log scale)')
ax2.set_xlim(0, xmax)
ax2.set_ylabel('Density (log)')

plt.suptitle('Anomaly Score Distribution — Deployed Autoencoder', color=STYLE['green'], fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, '01_score_distribution.png'), dpi=150, facecolor=STYLE['bg'])
plt.close()

# ── Fig 2: Full Cycle Score Trace ─────────────────────────────────────────────
print("Generating Fig 2: Cycle trace...")
fig, ax = plt.subplots(figsize=(10, 4))
apply_style(fig, ax)

times, scores, phase_ids = simulate_cycle(W1ae, b1ae, W2ae, b2ae, thr, ae_mean, ae_std)

phase_colors_bg = [STYLE['green']+'22', STYLE['yellow']+'22', STYLE['red']+'22', STYLE['blue']+'22']
bounds = [(0,25,'Normal',0),(25,30,'Ramp',1),(30,45,'Anomaly',2),(45,50,'Resolve',3)]
for t0, t1, name, pid in bounds:
    ax.axvspan(t0, t1, alpha=0.15, color=STYLE['colors'][pid], label=name)
    ax.text((t0+t1)/2, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1,
            name, ha='center', va='top', fontsize=7, color=STYLE['dim'])

ax.plot(times, scores, color=STYLE['blue'], linewidth=1.5, label='Anomaly Score')
ax.axhline(thr, color=STYLE['yellow'], linestyle='--', linewidth=1.2, label=f'Threshold ({thr:.3f})')
ax.fill_between(times, scores, thr, where=(scores > thr), alpha=0.3, color=STYLE['red'], label='Detected')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Reconstruction Error')
ax.set_title('Anomaly Score — Full 50s Cycle (Bearing Fault)')
ax.legend(fontsize=8, facecolor=STYLE['card'], labelcolor=STYLE['text'], edgecolor=STYLE['border'])
plt.tight_layout()
plt.savefig(os.path.join(out_dir, '02_cycle_trace.png'), dpi=150, facecolor=STYLE['bg'])
plt.close()

# ── Fig 3: Feature Importance ─────────────────────────────────────────────────
print("Generating Fig 3: Feature importance...")
fig, ax = plt.subplots(figsize=(9, 4))
apply_style(fig, ax)

normal_raw  = generate_samples(0, 500, speed_frac=0.39, intensity=0.0)
anomaly_raw = np.vstack([generate_samples(ft, 200, speed_frac=1.0, intensity=1.0) for ft in range(1,4)])

norm_mean_f = normal_raw.mean(axis=0)
anom_mean_f = anomaly_raw.mean(axis=0)
delta = np.abs(anom_mean_f - norm_mean_f) / (normal_raw.std(axis=0) + 1e-6)
order = np.argsort(delta)[::-1]

bars = ax.barh([FEATURE_NAMES[i] for i in order], delta[order],
               color=[STYLE['colors'][min(3, int(v/delta.max()*3))] for v in delta[order]],
               edgecolor=STYLE['border'], height=0.6)
ax.set_xlabel('Separation Score (|Δmean| / std)')
ax.set_title('Feature Importance — Normal vs Anomaly Separation')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, '03_feature_importance.png'), dpi=150, facecolor=STYLE['bg'])
plt.close()

# ── Fig 4: Training Curves ────────────────────────────────────────────────────
print("Generating Fig 4: Training curves...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
apply_style(fig, [ax1, ax2])

ax1.plot(ae_losses, color=STYLE['green'], linewidth=1.2)
ax1.set_xlabel('Epoch'); ax1.set_ylabel('MSE Loss')
ax1.set_title('Autoencoder Training Loss')
ax1.axhline(thr, color=STYLE['yellow'], linestyle='--', linewidth=1, label=f'Threshold')
ax1.legend(fontsize=8, facecolor=STYLE['card'], labelcolor=STYLE['text'], edgecolor=STYLE['border'])

ax2.plot(cls_losses, color=STYLE['blue'], linewidth=1.2)
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Cross-Entropy Loss')
ax2.set_title(f'Classifier Training Loss  (acc={acc:.1%})')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, '04_training_curves.png'), dpi=150, facecolor=STYLE['bg'])
plt.close()

# ── Fig 5: Classifier Confusion Matrix ───────────────────────────────────────
print("Generating Fig 5: Confusion matrix...")
fig, ax = plt.subplots(figsize=(6, 5))
apply_style(fig, ax)

im = ax.imshow(cm, cmap='Greens', aspect='auto')
ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(FAULT_NAMES, rotation=30, ha='right', fontsize=9)
ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(FAULT_NAMES, fontsize=9)
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
ax.set_title(f'Classifier Confusion Matrix  (acc={acc:.1%})')
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                color='white' if cm[i,j] > cm.max()*0.5 else STYLE['text'], fontsize=11)
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, '05_confusion_matrix.png'), dpi=150, facecolor=STYLE['bg'])
plt.close()

# ── Fig 6: Model Size & Architecture ─────────────────────────────────────────
print("Generating Fig 6: Model size...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
apply_style(fig, [ax1, ax2])

sizes = model_sizes()
labels = ['AE\nWeights', 'AE\nNorm', 'CLS\nWeights', 'CLS\nNorm']
vals   = [sizes['ae_weights'], sizes['ae_norm'], sizes['cls_weights'], sizes['cls_norm']]
colors = [STYLE['green'], STYLE['green']+'88', STYLE['blue'], STYLE['blue']+'88']
bars   = ax1.bar(labels, vals, color=colors, edgecolor=STYLE['border'], width=0.5)
for bar, val in zip(bars, vals):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
             f'{val}B', ha='center', va='bottom', fontsize=9, color=STYLE['text'])
ax1.set_ylabel('Size (bytes)')
ax1.set_title(f'Model Memory Footprint  (total {sizes["total"]}B)')
ax1.set_ylim(0, max(vals)*1.25)

arch_text = (
    f"AUTOENCODER (anomaly detection)\n"
    f"  Input:   {FEATURE_DIM} features\n"
    f"  Encoder: {FEATURE_DIM}→{HIDDEN_DIM}  (ReLU)\n"
    f"  Decoder: {HIDDEN_DIM}→{FEATURE_DIM}\n"
    f"  Threshold: {thr:.4f}\n\n"
    f"CLASSIFIER (fault type)\n"
    f"  Input:  {FEATURE_DIM} features\n"
    f"  Hidden: {CLASS_H1} → {CLASS_H2}  (ReLU×2)\n"
    f"  Output: {NUM_CLASSES} classes  (Softmax)\n"
    f"  Accuracy: {acc:.1%}\n\n"
    f"TOTAL PARAMS: {(sizes['total']//4)} floats\n"
    f"TOTAL SIZE:   {sizes['total']} bytes (~{sizes['total']//1024}KB)"
)
ax2.axis('off')
ax2.text(0.05, 0.95, arch_text, transform=ax2.transAxes,
         va='top', ha='left', fontsize=9, color=STYLE['text'],
         fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor=STYLE['card'], edgecolor=STYLE['border'], pad=0.8))
ax2.set_title('Architecture Summary')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, '06_model_size.png'), dpi=150, facecolor=STYLE['bg'])
plt.close()

# ── Fig 7: Detection Performance per Fault ───────────────────────────────────
print("Generating Fig 7: Detection performance...")
fig, ax = plt.subplots(figsize=(8, 4))
apply_style(fig, ax)

detect_rates = []
for ft in range(1, 4):
    raw = generate_samples(ft, 500, speed_frac=1.0, intensity=1.0)
    sc  = ae_score(raw, W1ae, b1ae, W2ae, b2ae, ae_mean, ae_std)
    detect_rates.append((sc > thr).mean() * 100)

normal_raw   = generate_samples(0, 500, speed_frac=0.39, intensity=0.0)
normal_scores = ae_score(normal_raw, W1ae, b1ae, W2ae, b2ae, ae_mean, ae_std)
fpr = (normal_scores > thr).mean() * 100

x = np.arange(3)
bars = ax.bar(x, detect_rates, color=[STYLE['colors'][i+1] for i in range(3)],
              edgecolor=STYLE['border'], width=0.4, label='Detection Rate')
ax.axhline(100-fpr, color=STYLE['yellow'], linestyle='--', linewidth=1.2,
           label=f'True Negative Rate ({100-fpr:.1f}%)')
for bar, val in zip(bars, detect_rates):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=10, color=STYLE['text'])
ax.set_xticks(x); ax.set_xticklabels(FAULT_NAMES[1:], fontsize=10)
ax.set_ylabel('Detection Rate (%)')
ax.set_ylim(0, 110)
ax.set_title(f'Detection Rate per Fault Type  (FPR={fpr:.1f}%)')
ax.legend(fontsize=8, facecolor=STYLE['card'], labelcolor=STYLE['text'], edgecolor=STYLE['border'])
plt.tight_layout()
plt.savefig(os.path.join(out_dir, '07_detection_performance.png'), dpi=150, facecolor=STYLE['bg'])
plt.close()

print(f"\nAll figures saved to {os.path.abspath(out_dir)}/")
print(f"\nKey metrics:")
print(f"  Autoencoder threshold : {thr:.6f}")
print(f"  Classifier accuracy   : {acc:.1%}")
print(f"  Total model size      : {sizes['total']} bytes")
print(f"  Parameters (floats)   : {sizes['total']//4}")
print(f"  False positive rate   : {fpr:.1f}%")
for i, (name, dr) in enumerate(zip(FAULT_NAMES[1:], detect_rates)):
    print(f"  {name} detection rate  : {dr:.1f}%")
