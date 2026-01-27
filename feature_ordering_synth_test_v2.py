"""
Four-branch recovery test using the original TimeSeriesModel architecture.
Run in Python with TensorFlow and matplotlib. SciPy is optional (Hungarian assignment).
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from typing import List
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False
    print("scipy not available — using greedy discrete assignment fallback.")

# ----------------------------
# Recreate layers from experiment
# ----------------------------
@tf.keras.utils.register_keras_serializable(package="Custom")
class LearnableFeaturePermute(tf.keras.layers.Layer):
    def __init__(self, num_features, num_iters=20, temperature=1.0, name=None):
        super().__init__(name=name)
        self.num_features = int(num_features)
        self.num_iters = int(num_iters)
        self.temperature = tf.Variable(float(temperature), trainable=False, dtype=tf.float32)
        # logits (F, F)
        init = tf.random.normal([self.num_features, self.num_features], stddev=0.01)
        self.logits = self.add_weight(shape=(self.num_features, self.num_features),
                                      initializer=tf.constant_initializer(init.numpy()),
                                      trainable=True, name=(name or "perm") + "_logits")

    def _sinkhorn(self, log_alpha):
        logP = log_alpha
        for _ in range(self.num_iters):
            logP = logP - tf.reduce_logsumexp(logP, axis=1, keepdims=True)
            logP = logP - tf.reduce_logsumexp(logP, axis=0, keepdims=True)
        return tf.exp(logP)

    def call(self, x):
        # x: (B, T, F)
        P = self._sinkhorn(self.logits / self.temperature)
        flat = tf.reshape(x, [-1, tf.shape(x)[-1]])  # (B*T, F)
        out = tf.matmul(flat, P)                     # (B*T, F)
        out_shape = tf.concat([tf.shape(x)[:-1], tf.constant([self.num_features], dtype=tf.int32)], axis=0)
        return tf.reshape(out, out_shape)

    def soft_P(self):
        return self._sinkhorn(self.logits / self.temperature)

@tf.keras.utils.register_keras_serializable(package="Custom")
class LearnableQueryPooling(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.query = self.add_weight(shape=(1,1,dim), initializer="glorot_uniform",
                                     trainable=True, name="learnable_query")
        self.proj = tf.keras.layers.Dense(dim)

    def call(self, x):
        x = self.proj(x)
        batch = tf.shape(x)[0]
        q = tf.tile(self.query, [batch, 1, 1])
        scores = tf.matmul(q, x, transpose_b=True)
        scores = tf.nn.softmax(scores, axis=-1)
        context = tf.matmul(scores, x)
        return tf.squeeze(context, axis=1)

@tf.keras.utils.register_keras_serializable(package="Custom")
class MultiHeadReadout(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads=2):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        key_dim = dim // num_heads if dim >= num_heads else 1
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)

    def build(self, input_shape):
        self.query = self.add_weight(
            shape=(1, 1, self.dim),
            initializer="glorot_uniform",
            trainable=True,
            name="readout_query"
        )
        super().build(input_shape)

    def call(self, x):
        batch_size = tf.shape(x)[0]
        q = tf.tile(self.query, [batch_size, 1, 1])
        out = self.mha(query=q, value=x, key=x)
        return tf.squeeze(out, axis=1)

@tf.keras.utils.register_keras_serializable(package="Custom")
class AdvancedGatedReadout(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads=2):
        super().__init__()
        self.dim = dim
        self.lq = LearnableQueryPooling(dim)
        self.mhr = MultiHeadReadout(dim, num_heads)
        self.gate = tf.keras.layers.Dense(5, activation="softmax")

    def call(self, x):
        last    = x[:, -1, :]
        avg     = tf.reduce_mean(x, axis=1)
        mx      = tf.reduce_max(x, axis=1)
        lq_out  = self.lq(x)
        mha_out = self.mhr(x)
        primitives = tf.stack([last, avg, mx, lq_out, mha_out], axis=1)  # (B,5,D)
        gate_w = self.gate(tf.concat([last, avg, mx, lq_out, mha_out], axis=-1))
        gate_w = tf.expand_dims(gate_w, axis=-1)
        out = tf.reduce_sum(gate_w * primitives, axis=1)
        return out

class TimeSeriesModel(tf.keras.Model):
    def __init__(self, num_features, dims, norm_layer):
        super().__init__()
        self.permute1 = LearnableFeaturePermute(num_features, name="perm1")
        self.permute2 = LearnableFeaturePermute(num_features, name="perm2")
        self.permute3 = LearnableFeaturePermute(num_features, name="perm3")
        self.permute4 = LearnableFeaturePermute(num_features, name="perm4")

        self.agr = AdvancedGatedReadout(dims)

        self.norm = norm_layer
        self.gru1 = tf.keras.layers.GRU(dims, return_sequences=True, seed=SEED)
        self.lstm = tf.keras.layers.LSTM(dims, return_sequences=True, seed=SEED)
        self.conv1 = tf.keras.layers.Conv1D(dims, 3, padding='same')
        self.act1 = tf.keras.layers.Activation('gelu')
        self.t_dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(dims, activation='gelu'))

        self.conc = tf.keras.layers.Concatenate()
        self.dense = tf.keras.layers.Dense(4, activation='softmax')
        self.add = tf.keras.layers.Add()
        self.head = tf.keras.layers.Dense(1)

        self.drop1 = tf.keras.layers.Dropout(0.1)
        self.drop2 = tf.keras.layers.Dropout(0.1)
        self.drop3 = tf.keras.layers.Dropout(0.1)
        self.drop4 = tf.keras.layers.Dropout(0.1)

    def call(self, x, training=False):
        x = self.norm(x)
        x1 = self.gru1(self.permute1(x))
        x1 = self.drop1(x1, training=training)

        x2 = self.lstm(self.permute2(x))
        x2 = self.drop2(x2, training=training)

        x3 = self.conv1(self.permute3(x))
        x3 = self.act1(x3)
        x3 = self.drop3(x3, training=training)

        x4 = self.t_dense(self.permute4(x))
        x4 = self.drop4(x4, training=training)

        comb = [x1, x2, x3, x4]
        fusion = self.conc(comb)
        weights = self.dense(fusion)

        w1 = weights[:, :, 0:1]
        w2 = weights[:, :, 1:2]
        w3 = weights[:, :, 2:3]
        w4 = weights[:, :, 3:4]

        x = self.add([x1 * w1, x2 * w2, x3 * w3, x4 * w4])
        x = self.agr(x)
        return self.head(x)

    @property
    def perm_vars(self):
        return (
            self.permute1.trainable_variables +
            self.permute2.trainable_variables +
            self.permute3.trainable_variables +
            self.permute4.trainable_variables
        )

    @property
    def model_vars(self):
        return (
            self.agr.trainable_variables +
            self.gru1.trainable_variables +
            self.lstm.trainable_variables +
            self.conv1.trainable_variables +
            self.t_dense.trainable_variables +
            self.dense.trainable_variables +
            self.head.trainable_variables
        )

# ----------------------------
# Synthetic data (matching multi-branch identifiability)
# ----------------------------
def make_sinusoid_basis(T, F):
    t = np.arange(T) / float(T)
    freqs = np.linspace(1.0, 4.0, F)
    phases = np.linspace(0.0, np.pi/2.0, F)
    S = np.stack([np.sin(2 * np.pi * f * t + p) for f, p in zip(freqs, phases)], axis=-1)
    return S.astype(np.float32)

def generate_multi_branch_data(N_samples=3000, WL=16, F=8, noise_std=0.01):
    T_total = N_samples + WL + 50
    base = make_sinusoid_basis(T_total, F)
    X = []
    for i in range(N_samples):
        w = base[i:i+WL].copy()
        w = w * (1.0 + 0.05 * np.random.randn(1, F))
        X.append(w)
    X = np.stack(X, axis=0)  # (N, WL, F)

    rng = np.random.default_rng(SEED)
    # global scramble Q: canonical_col -> observed_col
    Q = rng.permutation(F)
    # branch canonical mappings C_j: permutation arrays of canonical columns
    branch_C = [rng.permutation(F) for _ in range(4)]
    # branch weights (per-canonical position)
    branch_weights = [np.linspace(1.0, 0.2, F) * (0.5 + 0.5 * rng.random()) for _ in range(4)]

    # Build target: each branch contributes from canonical last-step with its C_j mapping
    last = X[:, -1, :]  # (N, F) canonical
    y = np.zeros(len(last), dtype=np.float32)
    for j in range(4):
        contrib = (last[:, branch_C[j]] * branch_weights[j][None, :]).sum(axis=1)
        y += contrib
    y += noise_std * np.random.randn(len(y)).astype(np.float32)

    # Observed inputs scrambled by Q
    X_obs = X[..., Q]  # (N, WL, F) observed col order

    # compute ground-truth mapping for each branch as row->col mapping (observed_row -> canonical_col)
    # We want P_true such that P_true[observed_row] = canonical_col index (same format as Hungarian output).
    # If Q maps canonical_col -> observed_col, then for canonical position r, observed_col = Q[C_j[r]]
    # So row r (observed index) corresponds to some canonical col — invert mapping as earlier.
    P_trues = []
    for j in range(4):
        # For canonical row idx r (0..F-1), observed_col = Q[ C_j[r] ]
        # We want array true_perm where true_perm[observed_row] = canonical_row
        observed_cols_for_rows = Q[branch_C[j]]  # observed_col per canonical row index
        # Build inverse mapping: for each observed_col, which canonical row does it belong to?
        inv = np.empty(F, dtype=int)
        for canonical_row, observed_col in enumerate(observed_cols_for_rows):
            inv[observed_col] = canonical_row
        P_trues.append(inv)  # inv maps observed_col -> canonical_row (row->col mapping format)
    return X_obs.astype(np.float32), y.astype(np.float32), Q, branch_C, P_trues, branch_weights

# ----------------------------
# Utilities: discrete extraction + kendall tau
# ----------------------------
def extract_discrete_from_soft(P: np.ndarray) -> np.ndarray:
    # P shape (F, F), rows = observed cols, cols = canonical rows
    if SCIPY_AVAILABLE:
        r, c = linear_sum_assignment(-P)  # maximize P
        pred = np.zeros(P.shape[0], dtype=int)
        pred[r] = c
        return pred
    else:
        # greedy row-wise with conflict resolution
        Pcopy = P.copy()
        F = P.shape[0]
        pred = -np.ones(F, dtype=int)
        assigned = set()
        for r in range(F):
            masked = Pcopy[r].copy()
            for a in assigned:
                masked[a] = -1e9
            c = int(np.argmax(masked))
            pred[r] = c
            assigned.add(c)
        return pred

def kendall_tau_perm(true_perm: np.ndarray, pred_perm: np.ndarray) -> float:
    # Both are arrays: observed_col -> canonical_row
    F = len(true_perm)
    pos_true = np.empty(F, dtype=int)
    pos_pred = np.empty(F, dtype=int)
    for obs_col, canon_row in enumerate(true_perm):
        pos_true[obs_col] = canon_row
    for obs_col, canon_row in enumerate(pred_perm):
        pos_pred[obs_col] = canon_row
    concordant = 0
    discordant = 0
    for i in range(F):
        for j in range(i+1, F):
            a = pos_true[i] - pos_true[j]
            b = pos_pred[i] - pos_pred[j]
            if a * b > 0:
                concordant += 1
            elif a * b < 0:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return 0.0
    return float((concordant - discordant) / total)

# ----------------------------
# Train/evaluate using TimeSeriesModel with separate optimizers
# ----------------------------
def run_test(N=3000, WL=16, F=8, epochs=200, batch_size=128,
             lr_model=1e-3, lr_perm=1e-3, lambda_entropy=0.08, gamma_repel=0.03,
             hidden_dim=16, verbose=True):
    X, y, Q, branch_C, P_trues, branch_weights = generate_multi_branch_data(N_samples=N, WL=WL, F=F)
    split = int(0.8 * N)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # normalization layer adapted on training set
    norm = tf.keras.layers.Normalization()
    # flatten over windows/time axis for adapt
    norm.adapt(X_train)

    model = TimeSeriesModel(num_features=F, dims=hidden_dim, norm_layer=norm)
    # Build model to initialize weights
    _ = model(tf.zeros([1, WL, F], dtype=tf.float32))

    # Separate var lists
    perm_vars = model.perm_vars
    model_vars = model.model_vars

    opt_model = tf.keras.optimizers.AdamW(learning_rate=lr_model)
    opt_perm = tf.keras.optimizers.AdamW(learning_rate=lr_perm)

    mse = tf.keras.losses.MeanSquaredError()
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(2000, seed=SEED).batch(batch_size)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)

    history = {'train_loss': [], 'val_loss': [], 'ent': [], 'repulsion': []}

    for ep in range(1, epochs+1):
        ep_loss = 0.0
        steps = 0
        for xb, yb in train_ds:
            with tf.GradientTape(persistent=True) as tape:
                preds = model(xb, training=True)[:, 0]
                loss_mse = mse(yb, preds)
                # compute soft P's and stats
                Ps = [p.soft_P() for p in [model.permute1, model.permute2, model.permute3, model.permute4]]
                P_stack = tf.stack(Ps, axis=0)  # (4, F, F)
                P_clamped = tf.clip_by_value(P_stack, 1e-9, 1.0)
                row_ent = -tf.reduce_mean(tf.reduce_sum(P_clamped * tf.math.log(P_clamped + 1e-12), axis=-1), axis=1)  # (4,)
                mean_ent = tf.reduce_mean(row_ent)
                # pairwise L1 repulsion sum
                rep_sum = tf.constant(0.0, dtype=tf.float32)
                for i in range(4):
                    for j in range(i+1, 4):
                        rep_sum += tf.reduce_mean(tf.abs(P_stack[i] - P_stack[j]))
                        
                total_loss = loss_mse + lambda_entropy * mean_ent - gamma_repel * rep_sum

            # grads
            grads_model = tape.gradient(total_loss, model_vars)
            grads_perm = tape.gradient(total_loss, perm_vars)

            # replace None with zeros
            grads_model = [(g if g is not None else tf.zeros_like(v)) for g, v in zip(grads_model, model_vars)]
            grads_perm = [(g if g is not None else tf.zeros_like(v)) for g, v in zip(grads_perm, perm_vars)]

            # optional clipping
            grads_model = [tf.clip_by_norm(g, 5.0) for g in grads_model]
            grads_perm = [tf.clip_by_norm(g, 2.0) for g in grads_perm]

            opt_model.apply_gradients(zip(grads_model, model_vars))
            opt_perm.apply_gradients(zip(grads_perm, perm_vars))

            del tape
            ep_loss += float(total_loss.numpy())
            steps += 1

        train_loss = ep_loss / max(1, steps)

        # validation
        val_loss = 0.0
        vsteps = 0
        for xv, yv in val_ds:
            vp = model(xv, training=False)[:, 0]
            val_loss += float(mse(yv, vp).numpy())
            vsteps += 1
        val_loss = val_loss / max(1, vsteps)

        # logging stats
        Ps_np = [p.soft_P().numpy() for p in [model.permute1, model.permute2, model.permute3, model.permute4]]
        row_ent_np = -np.mean(np.sum(np.clip(Ps_np, 1e-12, 1.0) * np.log(np.clip(Ps_np, 1e-12, 1.0)), axis=-1), axis=1)
        rep_np = 0.0
        for i in range(4):
            for j in range(i+1, 4):
                rep_np += float(np.mean(np.abs(Ps_np[i] - Ps_np[j])))

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['ent'].append(float(np.mean(row_ent_np)))
        history['repulsion'].append(float(rep_np))

        if verbose: # and (ep <= 5 or ep % (max(1, epochs // 10)) == 0):
            print(f"Epoch {ep:03d} | train {train_loss:.6f} val {val_loss:.6f} ent {np.mean(row_ent_np):.5f} rep {rep_np:.5f}")

    # final discrete extraction and metrics
    P_final = [p.soft_P().numpy() for p in [model.permute1, model.permute2, model.permute3, model.permute4]]
    preds = [extract_discrete_from_soft(P_final[i]) for i in range(4)]
    accuracies = []
    taus = []
    for j in range(4):
        true = P_trues[j]   # observed_col -> canonical_row
        pred = preds[j]
        acc = float((pred == true).mean())
        tau = kendall_tau_perm(true, pred)
        accuracies.append(acc)
        taus.append(tau)
        print(f"\nBranch {j}: true_perm (observed->canon): {true}")
        print(f"          pred_perm:                       {pred}")
        print(f"          acc {acc:.3f} | kendall-tau {tau:.3f}")

    # plots
    plt.figure(figsize=(12,5))
    plt.subplot(1,3,1)
    plt.plot(history['train_loss'], label='train'); plt.plot(history['val_loss'], label='val')
    plt.yscale('log'); plt.legend(); plt.title('Loss')
    plt.subplot(1,3,2)
    plt.plot(history['ent']); plt.title('mean row entropy')
    plt.subplot(1,3,3)
    plt.plot(history['repulsion']); plt.title('repulsion (sum pairwise L1)')
    plt.tight_layout(); plt.show()

    # show final P matrices
    fig, axs = plt.subplots(1, 4, figsize=(12,3))
    for i in range(4):
        axs[i].imshow(P_final[i], aspect='auto', cmap='viridis')
        axs[i].set_title(f"P_final b{i}")
    plt.show()

    return {
        'P_trues': P_trues,
        'P_pred': preds,
        'accuracies': accuracies,
        'taus': taus,
        'history': history,
        'P_final': P_final,
        'Q': Q,
        'branch_C': branch_C,
        'branch_weights': branch_weights
    }

if __name__ == "__main__":
    out = run_test(N=10000, WL=8, F=8, epochs=200, batch_size=256,
                   lr_model=1e-3, lr_perm=1e-2, lambda_entropy=0.08, gamma_repel=0.03,
                   hidden_dim=4, verbose=True)
