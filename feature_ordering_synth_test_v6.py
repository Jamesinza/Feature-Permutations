"""
Ready-to-run experiment implementing Gumbel-Sinkhorn + straight-through permutation learning.
Tuned defaults for a short experiment. Run with Python (TensorFlow 2.x), matplotlib and optionally SciPy.

Usage:
    python gumbel_sinkhorn_permutation_experiment.py

Outputs: prints epoch diagnostics and shows simple plots at the end. Paste results if you want follow-ups.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from typing import List
import time

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
# Utilities / diagnostics
# ----------------------------

def diagnostic_perm_stats(perms, perm_vars):
    stats = {}
    F = perms[0].num_features
    uniform_entropy = float(np.log(F))
    P_np = [p.soft_P().numpy() for p in perms]  # list (F,F)
    ent_np = [np.mean(-np.sum(np.clip(P, 1e-12, 1.0) * np.log(np.clip(P, 1e-12, 1.0)), axis=-1)) for P in P_np]
    stats['ent_per_branch'] = ent_np
    stats['ent_mean'] = float(np.mean(ent_np))
    stats['uniform_entropy'] = uniform_entropy
    logits_np = [v.numpy() for v in perm_vars]
    stats['logits_norm'] = [float(np.linalg.norm(l)) for l in logits_np]
    stats['logits_mean_abs'] = [float(np.mean(np.abs(l))) for l in logits_np]
    kl_rows = []
    for P in P_np:
        q = np.ones_like(P) / F
        kl = np.mean(np.sum(P * (np.log(np.clip(P,1e-12,1.0)) - np.log(q)), axis=-1))
        kl_rows.append(float(kl))
    stats['kl_uniform'] = kl_rows
    return stats

# ----------------------------
# Permutation layer: Gumbel-Sinkhorn + straight-through optional
# ----------------------------
@tf.keras.utils.register_keras_serializable(package="Custom")
class LearnableFeaturePermute(tf.keras.layers.Layer):
    def __init__(self, num_features, num_iters=20, temperature=1.0, name=None, use_gumbel=True, straight_through=True):
        super().__init__(name=name)
        self.num_features = int(num_features)
        self.num_iters = int(num_iters)
        self.temperature = tf.Variable(float(temperature), trainable=False, dtype=tf.float32)
        self.use_gumbel = use_gumbel
        self.straight_through = straight_through

        # initialize logits near zero for near-uniform start
        self.logits = self.add_weight(shape=(self.num_features, self.num_features),
                                      initializer=tf.zeros_initializer(),
                                      trainable=True, name=(name or "perm") + "_logits")

    def _sinkhorn(self, log_alpha):
        logP = log_alpha
        for _ in range(self.num_iters):
            logP = logP - tf.reduce_logsumexp(logP, axis=1, keepdims=True)
            logP = logP - tf.reduce_logsumexp(logP, axis=0, keepdims=True)
        return tf.exp(logP)

    def _sample_gumbel(self, shape, eps=1e-20):
        u = tf.random.uniform(shape, minval=0.0, maxval=1.0)
        return -tf.math.log(-tf.math.log(u + eps) + eps)

    def soft_P(self, add_gumbel=False):
        logits = self.logits / self.temperature
        if self.use_gumbel and add_gumbel:
            g = self._sample_gumbel(tf.shape(logits))
            logits = logits + g
        return self._sinkhorn(logits)

    def call(self, x, training=False, hard=False):
        # x: (B, T, F)
        if self.use_gumbel and training:
            P_soft = self.soft_P(add_gumbel=True)
        else:
            P_soft = self.soft_P(add_gumbel=False)

        if self.straight_through and hard:
            # discrete forward: project to permutation with Hungarian on numpy
            P_np = P_soft.numpy()
            try:
                if SCIPY_AVAILABLE:
                    r, c = linear_sum_assignment(-P_np)
                    P_hard = np.zeros_like(P_np)
                    P_hard[r, c] = 1.0
                else:
                    # greedy fallback
                    F = P_np.shape[0]
                    P_hard = np.zeros_like(P_np)
                    assigned = set()
                    for r in range(F):
                        masked = P_np[r].copy()
                        for a in assigned:
                            masked[a] = -1e9
                        c = int(np.argmax(masked))
                        P_hard[r, c] = 1.0
                        assigned.add(c)
            except Exception:
                # safe fallback
                P_hard = np.eye(self.num_features, dtype=np.float32)
            P_hard_tf = tf.convert_to_tensor(P_hard, dtype=tf.float32)
            P = tf.stop_gradient(P_hard_tf - P_soft) + P_soft
        else:
            P = P_soft

        flat = tf.reshape(x, [-1, tf.shape(x)[-1]])  # (B*T, F)
        out = tf.matmul(flat, P)                     # (B*T, F)
        out_shape = tf.concat([tf.shape(x)[:-1], tf.constant([self.num_features], dtype=tf.int32)], axis=0)
        return tf.reshape(out, out_shape)

    def set_temperature(self, t):
        self.temperature.assign(float(t))

# ----------------------------
# Other model components (unchanged structure, slightly simplified build)
# ----------------------------
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
    def __init__(self, num_features, dims, norm_layer, perm_kwargs=None):
        super().__init__()
        perm_kwargs = perm_kwargs or {}
        self.permute1 = LearnableFeaturePermute(num_features, name="perm1", **perm_kwargs)
        self.permute2 = LearnableFeaturePermute(num_features, name="perm2", **perm_kwargs)
        self.permute3 = LearnableFeaturePermute(num_features, name="perm3", **perm_kwargs)
        self.permute4 = LearnableFeaturePermute(num_features, name="perm4", **perm_kwargs)

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

    def call(self, x, training=False, hard_perm=False):
        x = self.norm(x)
        x1 = self.gru1(self.permute1(x, training=training, hard=hard_perm))
        x1 = self.drop1(x1, training=training)

        x2 = self.lstm(self.permute2(x, training=training, hard=hard_perm))
        x2 = self.drop2(x2, training=training)

        x3 = self.conv1(self.permute3(x, training=training, hard=hard_perm))
        x3 = self.act1(x3)
        x3 = self.drop3(x3, training=training)

        x4 = self.t_dense(self.permute4(x, training=training, hard=hard_perm))
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
    Q = rng.permutation(F)
    branch_C = [rng.permutation(F) for _ in range(4)]
    branch_weights = [np.linspace(1.0, 0.2, F) * (0.5 + 0.5 * rng.random()) for _ in range(4)]

    last = X[:, -1, :]  # (N, F) canonical
    y = np.zeros(len(last), dtype=np.float32)
    for j in range(4):
        contrib = (last[:, branch_C[j]] * branch_weights[j][None, :]).sum(axis=1)
        y += contrib
    y += noise_std * np.random.randn(len(y)).astype(np.float32)

    X_obs = X[..., Q]  # (N, WL, F) observed col order

    P_trues = []
    for j in range(4):
        observed_cols_for_rows = Q[branch_C[j]]
        inv = np.empty(F, dtype=int)
        for canonical_row, observed_col in enumerate(observed_cols_for_rows):
            inv[observed_col] = canonical_row
        P_trues.append(inv)
    return X_obs.astype(np.float32), y.astype(np.float32), Q, branch_C, P_trues, branch_weights

# ----------------------------
# Utilities: discrete extraction + kendall tau
# ----------------------------

def extract_discrete_from_soft(P: np.ndarray) -> np.ndarray:
    if SCIPY_AVAILABLE:
        r, c = linear_sum_assignment(-P)  # maximize P
        pred = np.zeros(P.shape[0], dtype=int)
        pred[r] = c
        return pred
    else:
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
# Training / experiment runner
# ----------------------------

def run_test(N=2000, WL=8, F=8, epochs=200, batch_size=256,
             lr_model=1e-3, lr_perm=1e-4, lambda_entropy=0.6, gamma_repel=0.03,
             hidden_dim=8, verbose=True):

    X, y, Q, branch_C, P_trues, branch_weights = generate_multi_branch_data(N_samples=N, WL=WL, F=F)
    split = int(0.8 * N)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    norm = tf.keras.layers.Normalization()
    norm.adapt(X_train)

    perm_kwargs = dict(use_gumbel=True, straight_through=True, num_iters=20, temperature=1.0)
    model = TimeSeriesModel(num_features=F, dims=hidden_dim, norm_layer=norm, perm_kwargs=perm_kwargs)
    _ = model(tf.zeros([1, WL, F], dtype=tf.float32))

    perm_vars = model.perm_vars
    model_vars = model.model_vars

    opt_model = tf.keras.optimizers.AdamW(learning_rate=lr_model)
    opt_perm = tf.keras.optimizers.AdamW(learning_rate=lr_perm)

    mse = tf.keras.losses.MeanSquaredError()
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(2000, seed=SEED).batch(batch_size)
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)

    history = {'train_loss': [], 'val_loss': [], 'mean_row_max': [], 'ent': [], 'repulsion': []}

    ENT_MODEL_SCALE = 0.05
    start_temp = 1.0
    end_temp = 0.02

    @tf.function()
    def train_step(xb, yb):
        with tf.GradientTape(persistent=True) as tape:
            preds = model(xb, training=True, hard_perm=False)[:, 0]
            loss_mse = mse(yb, preds)

            Ps = [
                model.permute1.soft_P(),
                model.permute2.soft_P(),
                model.permute3.soft_P(),
                model.permute4.soft_P()
            ]
            P_stack = tf.stack(Ps, axis=0)
            P_clamped = tf.clip_by_value(P_stack, 1e-9, 1.0)

            # row sharpness
            row_max = tf.reduce_mean(tf.reduce_max(P_clamped, axis=-1), axis=1)
            mean_row_max = tf.reduce_mean(row_max)

            # entropy (diagnostic but stays TF)
            row_ent = -tf.reduce_mean(
                tf.reduce_sum(P_clamped * tf.math.log(P_clamped + 1e-12), axis=-1),
                axis=1
            )
            mean_ent = tf.reduce_mean(row_ent)

            # repulsion
            rep_sum = tf.constant(0.0, tf.float32)
            for i in tf.range(4):
                for j in tf.range(i + 1, 4):
                    rep_sum += tf.reduce_mean(tf.abs(P_stack[i] - P_stack[j]))
            Ff = tf.cast(tf.shape(P_stack)[1], tf.float32)
            rep_norm = rep_sum / (Ff * Ff + 1e-12)

            total_loss = loss_mse + (ENT_MODEL_SCALE * lambda_entropy) * (1.0 - mean_row_max)
            perm_loss  = (lambda_entropy * (1.0 - mean_row_max)) - (gamma_repel * rep_norm)

        grads_model = tape.gradient(total_loss, model_vars)
        grads_perm  = tape.gradient(perm_loss, perm_vars)
        del tape

        grads_model = [tf.clip_by_norm(g, 5.0) if g is not None else tf.zeros_like(v)
                       for g, v in zip(grads_model, model_vars)]
        grads_perm = [tf.clip_by_norm(g, 2.0) if g is not None else tf.zeros_like(v)
                      for g, v in zip(grads_perm, perm_vars)]

        opt_model.apply_gradients(zip(grads_model, model_vars))
        opt_perm.apply_gradients(zip(grads_perm, perm_vars))

        return total_loss, mean_row_max, mean_ent, rep_norm

    @tf.function()
    def val_step(xv, yv):
        preds = model(xv, training=False, hard_perm=False)[:, 0]
        return mse(yv, preds)
        
        

    def temp_for_epoch(ep, max_ep):
        return start_temp * ((end_temp / start_temp) ** (ep / float(max_ep)))

    patience = 50
    min_delta = 1e-6
    wait = 0
    best_val = float('inf')
    best_weights = None
    best_epoch = 0
    restore_best_at_end = True    

    for ep in range(1, epochs + 1):
        t0 = time.time()
        ep_loss = 0.0
        steps = 0

        # anneal temperature at epoch start
        t = temp_for_epoch(ep, epochs)
        for p in [model.permute1, model.permute2, model.permute3, model.permute4]:
            p.set_temperature(t)

        # training
        for xb, yb in train_ds:
            loss, row_max, ent, rep = train_step(xb, yb)

            ep_loss += float(loss)
            steps += 1
            
        train_loss = ep_loss / max(1, steps)

        # validation
        val_loss = 0.0
        for xv, yv in val_ds:
            val_loss += float(val_step(xv, yv))

        # logging stats (compute perms for diagnostics)
        Ps_np = [p.soft_P().numpy() for p in [model.permute1, model.permute2, model.permute3, model.permute4]]
        row_ent_np = -np.mean(np.sum(np.clip(Ps_np, 1e-12, 1.0) * np.log(np.clip(Ps_np, 1e-12, 1.0)), axis=-1), axis=1)
        rep_np = 0.0
        for i in range(4):
            for j in range(i + 1, 4):
                rep_np += float(np.mean(np.abs(Ps_np[i] - Ps_np[j])))
        row_max_np = np.mean([np.mean(np.max(Ps_np[i], axis=-1)) for i in range(4)])

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['mean_row_max'].append(float(row_max_np))
        history['ent'].append(float(np.mean(row_ent_np)))
        history['repulsion'].append(float(rep_np))

        if verbose:
            elapsed = time.time() - t0
            print(f"Epoch {ep:03d} | train {train_loss:.6f} val {val_loss:.6f} mean_row_max {row_max_np:.4f} ent {np.mean(row_ent_np):.4f} rep {rep_np:.4f} t={elapsed:.2f}s")

        if val_loss + min_delta < best_val:
            best_val = val_loss
            best_weights = model.get_weights()            # captures perms + model weights
            best_epoch = ep
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            print(f"Early stopping at epoch {ep} (best epoch {best_epoch}, val {best_val:.6f})")
            break

    if restore_best_at_end and best_weights is not None:
        model.set_weights(best_weights)
        print(f"Restored best weights from epoch {best_epoch} with val {best_val:.6f}")        

    # final discrete extraction and metrics
    P_final = [p.soft_P().numpy() for p in [model.permute1, model.permute2, model.permute3, model.permute4]]
    preds = [extract_discrete_from_soft(P_final[i]) for i in range(4)]
    accuracies = []
    taus = []
    for j in range(4):
        true = P_trues[j]
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
    plt.plot(history['mean_row_max']); plt.title('mean row max (sharpness)')
    plt.subplot(1,3,3)
    plt.plot(history['repulsion']); plt.title('repulsion (sum pairwise L1)')
    plt.tight_layout(); plt.show()

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
    out = run_test(N=10000, WL=8, F=8, epochs=10000, batch_size=1024,
                   lr_model=1e-3, lr_perm=5e-3, lambda_entropy=0.6, gamma_repel=0.03,
                   hidden_dim=8, verbose=True)

    print('\nExperiment finished — returned dictionary `out`.')
