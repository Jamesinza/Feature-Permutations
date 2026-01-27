"""
Synthetic permutation recovery test for LearnableFeaturePermute.

- Generates latent signals, permutes feature columns by a fixed ground-truth permutation.
- Target depends on canonical (unpermuted) feature order (so model must invert permutation).
- Model = LearnableFeaturePermute -> simple aggregator -> MLP regression head.
- After training we extract a discrete permutation via Hungarian (scipy) or greedy argmax,
  and report accuracy + Kendall-tau.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import os
from math import comb
np.set_printoptions(precision=3, suppress=True)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False
    print("scipy not available: will use greedy argmax fallback for discrete assignment.")

# -----------------------
# Helpers: Sinkhorn layer
# -----------------------
class LearnableFeaturePermute(tf.keras.layers.Layer):
    def __init__(self, num_features, num_iters=30, initial_temperature=1.0, name=None):
        super().__init__(name=name)
        self.num_features = num_features
        self.num_iters = num_iters
        # Temperature as trainable scalar could be helpful; keep fixed here for stability.
        self.temperature = tf.Variable(initial_temperature, trainable=False, dtype=tf.float32)
        # logits shape (F, F)
        init = tf.random.normal((num_features, num_features), stddev=0.01)
        self.logits = tf.Variable(init, trainable=True, dtype=tf.float32, name=f"{self.name}_logits")

    def _sinkhorn(self, log_alpha):
        # log_alpha: (F, F)
        logP = log_alpha
        for _ in range(self.num_iters):
            logP = logP - tf.reduce_logsumexp(logP, axis=1, keepdims=True)  # row norm
            logP = logP - tf.reduce_logsumexp(logP, axis=0, keepdims=True)  # col norm
        return tf.exp(logP)

    def call(self, x, training=False):
        # x: (B, T, F)
        logits = self.logits
        # temperature inverse scaling
        P = self._sinkhorn(logits / self.temperature)
        # apply permutation (mixing) to feature dimension
        b, t, f = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        flat = tf.reshape(x, [-1, f])   # (B*T, F)
        out = tf.matmul(flat, P)        # (B*T, F)
        out = tf.reshape(out, [b, t, f])
        return out

    def get_perm_matrix(self):
        # returns numpy P computed at current logits/temperature (detached)
        P = self._sinkhorn(self.logits / self.temperature)
        return P.numpy()

# -----------------------
# Synthetic dataset
# -----------------------
def make_sinusoid_basis(T, F, freqs=None, phases=None):
    """Create F latent sinusoids over T timesteps."""
    t = np.arange(T) / float(T)
    if freqs is None:
        freqs = np.linspace(1, 4, F)
    if phases is None:
        phases = np.linspace(0, np.pi, F)
    S = np.stack([np.sin(2 * np.pi * f * t + p) for f, p in zip(freqs, phases)], axis=-1)  # (T, F)
    return S.astype(np.float32)

def generate_dataset(N_samples=2000, WL=16, F=8, noise_std=0.01):
    """
    Create dataset where canonical features are generated from sinusoidal bases.
    We then permute columns with a fixed ground-truth permutation.
    Target y = linear combination of canonical (unpermuted) features at last timestep.
    """
    # base continuous signals (longer than WL so we can sample windows)
    T_total = N_samples + WL + 50
    base = make_sinusoid_basis(T_total, F)
    # produce sliding windows from base with small random amplitude/per-sample scaling
    X = []
    y = []
    for i in range(N_samples):
        start = i
        window = base[start:start+WL].copy()
        # add tiny amplitude modulation to avoid trivial perfect periodicity
        window = window * (1.0 + 0.1 * np.random.randn(1, F))
        X.append(window)
    X = np.stack(X, axis=0)  # (N, WL, F)

    # define ground-truth permutation (random)
    rng = np.random.default_rng(SEED)
    perm_true = rng.permutation(F)
    # define target weights in canonical order (so correct model needs canonical order)
    weights = np.linspace(1.0, 0.2, F)  # larger weight for low-index features
    # build targets: use canonical features at last timestep (not permuted)
    # So we must invert permutation to access them.
    X_unpermuted = X.copy()
    # permute input columns
    X_perm = X_unpermuted[..., perm_true]
    # produce y from canonical last-step features (before permutation)
    last_step = X_unpermuted[:, -1, :]  # (N, F)
    y = (last_step * weights[None, :]).sum(axis=1)
    # add noise
    y = y + noise_std * np.random.randn(len(y))
    return X_perm.astype(np.float32), y.astype(np.float32), perm_true, weights

# -----------------------
# Metrics: discrete extraction + kendall-tau
# -----------------------
def extract_discrete_perm(P):
    """
    P: (F, F) soft doubly-stochastic
    Return: predicted permutation array pred_perm of length F, mapping row index -> assigned column index.
    Prefer Hungarian if available; otherwise greedy argmax with conflict resolution.
    """
    P = np.asarray(P)
    F = P.shape[0]
    if SCIPY_AVAILABLE:
        # cost = -P so maximizing assignment equals minimizing negative
        row_ind, col_ind = linear_sum_assignment(-P)
        # linear_sum_assignment returns arrays of indices; rows are 0..F-1
        pred_perm = np.zeros(F, dtype=int)
        pred_perm[row_ind] = col_ind
        return pred_perm
    else:
        # greedy: pick largest remaining entry row-by-row
        P_copy = P.copy()
        pred = -np.ones(F, dtype=int)
        assigned_cols = set()
        for r in range(F):
            # pick argmax over columns not assigned yet
            c = int(np.argmax(P_copy[r]))
            # if already assigned, zero it and retry
            if c in assigned_cols:
                # mask assigned columns and retry
                masked = P_copy[r].copy()
                for a in assigned_cols:
                    masked[a] = -1e9
                c = int(np.argmax(masked))
            pred[r] = c
            assigned_cols.add(c)
        return pred

def kendall_tau_from_perms(true_perm, pred_perm):
    """
    Compute normalized Kendall tau distance between two permutations.
    Return tau in [-1,1] (1 = perfect agreement). We return coefficient.
    """
    n = len(true_perm)
    # convert permutation so that true_perm[i] gives column assigned to row i.
    # We want comparison between orderings; create inverse mapping to indices
    # Turn both into rank arrays over same domain: get positions of each element in the canonical order.
    # Simpler: reorder pred_perm indices by true_perm's order.
    # Build arrays a and b representing sequences of element ids
    # Let elements be 0..n-1 representing canonical feature ids.
    # true_order: position mapping of elements in true_perm
    pos_true = np.empty(n, dtype=int)
    for i, col in enumerate(true_perm):
        pos_true[col] = i
    # pred_perm gives column assigned to row r
    # we build arrays of length n where element k's position in predicted ordering is pos_pred[k]
    pos_pred = np.empty(n, dtype=int)
    for i, col in enumerate(pred_perm):
        pos_pred[col] = i
    # Now Kendall tau on pos_true vs pos_pred arrays: compare relative ordering of same items
    # compute number of concordant pairs
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i+1, n):
            a = pos_true[i] - pos_true[j]
            b = pos_pred[i] - pos_pred[j]
            if a * b > 0:
                concordant += 1
            elif a * b < 0:
                discordant += 1
            else:
                pass
    total_pairs = concordant + discordant
    if total_pairs == 0:
        return 0.0
    tau = (concordant - discordant) / total_pairs
    return tau

# -----------------------
# Build model
# -----------------------
def build_model(WL, F, hidden=64):
    inp = tf.keras.Input(shape=(WL, F))
    perm = LearnableFeaturePermute(F, num_iters=20, initial_temperature=1.0, name="perm")
    x = perm(inp)        # permuted features (soft)
    # simple aggregator: take last timestep then an MLP
    x = tf.keras.layers.Lambda(lambda z: z[:, -1, :])(x)
    x = tf.keras.layers.Dense(hidden, activation='gelu')(x)
    x = tf.keras.layers.Dense(hidden//2, activation='gelu')(x)
    out = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inp, outputs=out)
    return model, perm

# -----------------------
# Training & evaluation
# -----------------------
def run_recovery_test(N=2000, WL=16, F=8, epochs=200, batch=64, lr=1e-3, verbose=1):
    X, y, perm_true, weights = generate_dataset(N_samples=N, WL=WL, F=F, noise_std=0.01)
    # split
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model, perm_layer = build_model(WL=WL, F=F, hidden=64)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='mse', metrics=['mae'])
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=batch, verbose=verbose)

    # extract soft P
    P_soft = perm_layer.get_perm_matrix()  # (F,F)
    pred_perm = extract_discrete_perm(P_soft)
    # compute metrics
    # True perm tells how original columns were permuted to create input:
    # X_perm = X_unpermuted[..., perm_true]
    # We want predicted perm to map row index to column index; ideally pred_perm should be inverse of perm_true:
    # If perm_true maps canonical_col -> permuted_col, then inverse_perm maps permuted_col -> canonical_col.
    inv_true = np.empty_like(perm_true)
    for i, c in enumerate(perm_true):
        inv_true[c] = i
    # pred_perm gives column index for each row (row = canonical row after perm layer)
    # We compare pred_perm to inv_true
    acc = (pred_perm == inv_true).mean()
    tau = kendall_tau_from_perms(inv_true, pred_perm)

    print("\n--- RESULTS ---")
    print("Ground-truth permutation (perm_true):", perm_true)
    print("Inverse true (target mapping for permutation layer):", inv_true)
    print("Predicted discrete perm (row -> col):", pred_perm)
    print(f"Recovery accuracy (fraction exact matches): {acc:.3f}")
    print(f"Kendall-tau between inverse-true and predicted: {tau:.3f}")
    # show final validation loss
    val_loss = history.history['val_loss'][-1]
    print(f"Final val_loss: {val_loss:.6g}")

    # Plot learned P heatmap and training curves
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.imshow(P_soft, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title("Learned soft permutation matrix P (rows -> cols)")
    plt.xlabel("columns (permuted input positions)")
    plt.ylabel("rows (post-permute canonical positions)")

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.yscale('log')
    plt.legend()
    plt.title("Training loss")
    plt.tight_layout()
    plt.show()

    return {
        'perm_true': perm_true,
        'inv_true': inv_true,
        'pred_perm': pred_perm,
        'soft_P': P_soft,
        'accuracy': acc,
        'kendall_tau': tau,
        'history': history.history
    }

if __name__ == "__main__":
    # small quick run; adjust epochs to taste
    result = run_recovery_test(N=3000, WL=16, F=8, epochs=200, batch=128, lr=2e-3, verbose=1)
