"""
Here we implement a more complex epoch-driven update system.

Cycle:
baseline phase → propose permutation → adaptation phase → evaluate → (accept/reject) → temperature controller update

We run model training only in adaptation phases for full epochs.
Perm proposals are applied only during perm-phase.
Temperature controller training / committing happens only after adaptation and only if the perm change is accepted.
All decisions are based on validation loss averaged over a small evaluation window to reduce variance.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import random
import math
import matplotlib.pyplot as plt
import pickle, os

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# =========================================================== #
#                         Modules                             #
# =========================================================== #
@tf.keras.utils.register_keras_serializable(package="Custom")
class LearnableFeaturePermute(tf.keras.layers.Layer):
    def __init__(self, num_features, num_iters=10, temperature=1.0):
        super().__init__()
        self.num_features = num_features
        self.num_iters = num_iters

        self.temperature = tf.Variable(
            temperature, trainable=False, dtype=tf.float32
        )
        self.logits = self.add_weight(
            shape=(num_features, num_features),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            trainable=True
        )

    def _sinkhorn(self, log_alpha):
        # log_alpha: shape (F, F)
        logP = log_alpha
        for _ in range(self.num_iters):
            # normalize rows
            logP = logP - tf.reduce_logsumexp(logP, axis=1, keepdims=True)
            # normalize cols
            logP = logP - tf.reduce_logsumexp(logP, axis=0, keepdims=True)
        return tf.exp(logP)
        

    def call(self, x):
        # Sinkhorn permutation matrix P (F, F)
        P = self._sinkhorn(self.logits / self.temperature)
        # Flatten batch/time dims into one leading dim, keep features in last dim
        flat = tf.reshape(x, [-1, tf.shape(x)[-1]])  # shape (-1, F)
        out = tf.matmul(flat, P)                      # (-1, F)
    
        # Build output shape reliably: concat tensors (not Python lists)
        out_shape = tf.concat([tf.shape(x)[:-1], tf.constant([self.num_features], dtype=tf.int32)], axis=0)
        return tf.reshape(out, out_shape)


class MetaController(tf.keras.Model):
    def __init__(self, a_max=0.2, dim=1):
        super().__init__()
        self.a_max = a_max

        self.net = tf.keras.Sequential([
            tf.keras.layers.Dense(dim, activation="gelu"),
            tf.keras.layers.Dense(1, activation="gelu"),
        ])
        self.norm = tf.keras.layers.LayerNormalization()

    def call(self, state):
        # state shape: (batch, features)
        norm = self.norm(state)
        a = self.net(norm)
        return self.a_max * a


@tf.keras.utils.register_keras_serializable(package="Custom")
class LearnableQueryPooling(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        # store as (1,1,D) to simplify tiling
        self.query = self.add_weight(shape=(1,1,dim), initializer="glorot_uniform",
                                     trainable=True, name="learnable_query")
        self.proj = tf.keras.layers.Dense(dim)

    def call(self, x):
        # x: (B, T, D)
        x = self.proj(x)                       # (B, T, D)
        batch = tf.shape(x)[0]
        q = tf.tile(self.query, [batch, 1, 1]) # (B, 1, D)
        scores = tf.matmul(q, x, transpose_b=True)  # (B, 1, T)
        scores = tf.nn.softmax(scores, axis=-1)     # (B, 1, T)
        context = tf.matmul(scores, x)              # (B, 1, D)
        return tf.squeeze(context, axis=1)          # (B, D)


@tf.keras.utils.register_keras_serializable(package="Custom")
class MultiHeadReadout(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads=2):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        key_dim = dim // num_heads if dim >= num_heads else 1
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)

    def build(self, input_shape):
        # input_shape = (batch, time, dim)
        self.query = self.add_weight(
            shape=(1, 1, self.dim),           # <-- FIXED: properly rank-3
            initializer="glorot_uniform",
            trainable=True,
            name="readout_query"
        )
        super().build(input_shape)

    def call(self, x):
        # x: (B, T, D)
        batch_size = tf.shape(x)[0]
        # Repeat query for each batch element
        q = tf.tile(self.query, [batch_size, 1, 1])  # (B, 1, D)
        # Multi-Head Attention readout
        out = self.mha(query=q, value=x, key=x)      # (B, 1, D)
        return tf.squeeze(out, axis=1)               # (B, D)   
        

@tf.keras.utils.register_keras_serializable(package="Custom")
class AdvancedGatedReadout(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads=2):
        super().__init__()
        self.dim = dim
        self.lq = LearnableQueryPooling(dim)
        self.mhr = MultiHeadReadout(dim, num_heads)
        # gate produces 5 mixture weights for the five primitives
        self.gate = tf.keras.layers.Dense(5, activation="softmax")

    def call(self, x):
        # x: (B, T, D)
        last    = x[:, -1, :]               # (B,D)
        avg     = tf.reduce_mean(x, axis=1) # (B,D)
        mx      = tf.reduce_max(x, axis=1)  # (B,D)        
        lq_out  = self.lq(x)                # (B,D)
        mha_out = self.mhr(x)               # (B,D)

        # Stack primitives into (B, 5, D)
        primitives = tf.stack([last, avg, mx, lq_out, mha_out], axis=1)  # (B,5,D)

        # Gate: produce mixture weights (B,5)
        gate_w = self.gate(tf.concat([last, avg, mx, lq_out, mha_out], axis=-1))  # (B,5)
        gate_w = tf.expand_dims(gate_w, axis=-1)  # (B,5,1)

        out = tf.reduce_sum(gate_w * primitives, axis=1)  # (B,D)
        return out


@tf.keras.utils.register_keras_serializable(package="Custom")
class TimeSeriesModel(tf.keras.Model):
    def __init__(self, num_features, dims, norm):
        super().__init__()
        self.permute1 = LearnableFeaturePermute(num_features)
        self.permute2 = LearnableFeaturePermute(num_features)
        self.permute3 = LearnableFeaturePermute(num_features)
        self.permute4 = LearnableFeaturePermute(num_features)
        
        self.agr = AdvancedGatedReadout(dims)

        self.norm = norm
        self.gru1 = tf.keras.layers.GRU(dims, return_sequences=True, seed=SEED)
        self.gru2 = tf.keras.layers.GRU(dims, return_sequences=True, seed=SEED+1)
        self.lstm = tf.keras.layers.LSTM(dims, return_sequences=True, seed=SEED)
        # self.conv1 = tf.keras.layers.Conv1D(dims, 3, padding='same')
        # self.conv2 = tf.keras.layers.Conv1D(dims, 3, padding='same')
        # self.act1 = tf.keras.layers.Activation('gelu')
        # self.act2 = tf.keras.layers.Activation('gelu')
        self.t_dense = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(dims, activation='gelu'))
        
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
        # x1= self.act1(x1)
        x1 = self.drop1(x1)
        
        x2 = self.lstm(self.permute2(x))
        x2 = self.drop2(x2)
        
        x3 = self.gru2(self.permute3(x))
        # x3= self.act2(x3)
        x3 = self.drop3(x3)
        
        x4 = self.t_dense(self.permute4(x))
        x4 = self.drop4(x4)
        
        comb = [x1,x2,x3,x4]
        fusion = self.conc(comb)
        weights = self.dense(fusion)

        w1 = weights[:, :, 0:1]
        w2 = weights[:, :, 1:2]
        w3 = weights[:, :, 2:3]
        w4 = weights[:, :, 3:4]
        
        x = self.add([x1*w1, x2*w2, x3*w3, x4*w4])
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
            self.gru2.trainable_variables +
            self.t_dense.trainable_variables +
            self.dense.trainable_variables +
            self.head.trainable_variables
        )


# =========================================================== #
#                         Functions                           #
# =========================================================== #
def create_features(df, WL):
    df['mid'] = (df['High'] + df['Low']) / 2.0
    df['new_high'] = (df['High'] + df[['Open','Close']].max(axis=1)) / 2
    df['new_low'] = (df[['Open','Close']].min(axis=1) + df['Low']) / 2
    df['new_mid'] = (df['new_high'] + df['new_low']) / 2.0
    df['open_pct'] = df['Open'].pct_change()
    df['high_pct'] = df['High'].pct_change()
    df['low_pct'] = df['Low'].pct_change()
    df['close_pct'] = df['Close'].pct_change()

    df['candle_size'] = df['High'] - df['Low']
    df['candle_dir'] = np.where(df['candle_size'] != 0,
                                (df['Close'] - df['Open']) / df['candle_size'], 0)
    df['upper_wick']  = np.where(df['candle_size'] != 0,
                                 (df['High'] - df[['Open','Close']].max(axis=1)), 0)
    df['lower_wick']  = np.where(df['candle_size'] != 0,
                                 (df[['Open','Close']].min(axis=1) - df['Low']), 0)
    df['close_pos']   = np.where(df['candle_size'] != 0,
                                 (df['Close'] - df['Low']) / df['candle_size'], 0)
    df['open_pos']    = np.where(df['candle_size'] != 0,
                                 (df['Open'] - df['Low']) / df['candle_size'], 0)
    df['range_change'] = np.where(df['candle_size'] != 0,
                                  df['candle_size'] / df['candle_size'].shift(1), 0)
    df['vol_rank'] = df['candle_size'].rolling(WL).rank(pct=True)
    
    df['body_size']    = abs(df['Close'] - df['Open'])
    df['body_rank'] = df['body_size'].rolling(WL).rank(pct=True)
        
    df['dir_continuity'] = np.sign(df['Close'] - df['Open']) * np.sign(df['Close'].shift(1) - df['Open'].shift(1))
    df['candle_mid']  = (df['High'] + df['Low']) / 2.0
    df['candle_body'] = abs(df['Close'] - df['Open'])
    df['close_vs_highest'] = df['Close'] / df['High'].rolling(WL).max()
    df['close_vs_lowest'] = df['Close'] / df['Low'].rolling(WL).min()
    df['momentum'] = (df['Close'] - df['Open']) / df['Open']
    df['volatility'] = df['candle_size'] / df['Open']
    df['energy'] = df['body_size'] * np.sign(df['Close'] - df['Open'])
    df['pressure'] = df['energy'].rolling(WL).sum() / WL
    df['wick_symmetry'] = abs(df['upper_wick'] - df['lower_wick'])
    df['body_entropy'] = -(
        (df['close_pos'] * np.log(df['close_pos'] + 1e-9)) +
        ((1 - df['close_pos']) * np.log(1 - df['close_pos'] + 1e-9))
    )
    df.dropna(inplace=True)
    df.reset_index(inplace=True, drop=True)
    return df

def create_windows(X, y, WL):
    X_windows = np.empty([len(X) - WL, WL, X.shape[-1]], dtype=np.float32)
    y_windows = np.empty([len(X) - WL], dtype=np.float32)
    for i in range(len(X) - WL):
        X_windows[i] = X[i:i+WL]
        y_windows[i] = y.flatten()[i+WL]
    return X_windows, y_windows

def load_csv_file(PATH, NUM_SAMPLES):
    raw_df = pd.read_csv(PATH)
    df = raw_df.tail(NUM_SAMPLES).copy()
    df = df[df['High'] != df['Low']]
    df = df[['Open','High','Low','Close']]
    df.dropna(inplace=True)
    df.reset_index(inplace=True, drop=True)
    return df
    
def load_data(PATH, WL, NUM_SAMPLES):
    data = load_csv_file(PATH, NUM_SAMPLES)
    df = create_features(data.copy(), WL)
    features = [f for f in df.columns]
    X = df[features].values
    y = df['Close'].values
    X_windows, y_windows = create_windows(X, y, WL)
    return X_windows, y_windows, features

def compute_batch_size(dataset_length: int) -> int:
    base_unit = 24_576
    base_batch = 32
    return int(base_batch * math.ceil(max(1, dataset_length) / base_unit))

def get_lr(optimizer):
    lr = optimizer.learning_rate
    if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
        return lr(optimizer.iterations).numpy()
    return tf.keras.backend.get_value(lr)

def safe_entropy(P, eps):
    P = tf.clip_by_value(P, eps, 1.0)
    return -tf.reduce_mean(tf.reduce_sum(P * tf.math.log(P), axis=-1))

@tf.function
def train_step(x, y, apply_model_updates=True, apply_perm_updates=True, eps=1e-8):
    """
    Vectorized train step:
      - computes task loss
      - builds stacked permutation matrices P_stack (K, F, F)
      - computes per-permutation entropy (K,), repulsion (K,), advantage (K,)
      - forms perm-specific losses (K,) and applies per-perm gradient updates (if allowed)
      - updates model weights from task_loss (if allowed)
    Returns diagnostics for aggregation at epoch end.
    """
    perm_names = ["p1", "p2", "p3", "p4"]
    perm_layers = [model.permute1, model.permute2, model.permute3, model.permute4]

    with tf.GradientTape(persistent=True) as tape:
        # Forward pass
        y_pred = model(x, training=True)
        task_loss = loss_fn(y, y_pred)

        # ---------------------------
        # Build stacked permutation matrices P_stack (K, F, F)
        # ---------------------------
        temps = [tf.clip_by_value(temperatures[n], temp_min, temp_max) for n in perm_names]
        logits_list = [p.logits for p in perm_layers]

        # Compute Sinkhorn for each permutation and stack
        # Using Python list comprehension is OK inside @tf.function here;
        # each perm._sinkhorn is TF operations and will be part of the graph.
        P_list = [perm._sinkhorn(logits / temp) for perm, logits, temp in zip(perm_layers, logits_list, temps)]
        P_stack = tf.stack(P_list, axis=0)                            # (K, F, F)
        P_stack = tf.clip_by_value(P_stack, SINKHORN_EPS, 1.0)

        # ---------------------------
        # Entropy (vectorized)
        # ---------------------------
        P_clamped = tf.clip_by_value(P_stack, eps, 1.0)
        # per-permutation row-wise entropy then mean across rows -> shape (K,)
        row_entropy_vec = -tf.reduce_mean(tf.reduce_sum(P_clamped * tf.math.log(P_clamped), axis=-1), axis=-1)

        # ---------------------------
        # Repulsion (vectorized pairwise L1 mean)
        # ---------------------------
        # Broadcast to (K, K, F, F)
        P_i = tf.expand_dims(P_stack, 1)                # (K,1,F,F)
        P_j = tf.expand_dims(tf.stop_gradient(P_stack), 0)  # (1,K,F,F) stop_gradient mirrors old code
        pairwise_abs = tf.reduce_mean(tf.abs(P_i - P_j), axis=[-2, -1])  # (K, K)

        # zero-out diagonal (self-distances)
        K = tf.shape(pairwise_abs)[0]
        eye = tf.eye(K, dtype=pairwise_abs.dtype)
        pairwise_abs_no_diag = pairwise_abs * (1.0 - eye)

        # repulsion per perm is sum of distances to other perms -> (K,)
        repulsion_vec = tf.reduce_sum(pairwise_abs_no_diag, axis=1)

        # ---------------------------
        # Advantage (vectorized)
        # ---------------------------
        mean_ent = tf.reduce_mean(row_entropy_vec)
        advantage_vec = mean_ent - row_entropy_vec

        # ---------------------------
        # Perm losses (vectorized)
        # ---------------------------
        entropy_w_vec = tf.stack([entropy_weights[n] for n in perm_names])
        perm_loss_vec = (
            task_loss
            + entropy_w_vec * row_entropy_vec
            - BETA_ADV * advantage_vec
            + GAMMA_REPEL * repulsion_vec
        )

    # ---------------------------
    # MODEL update (task loss)
    # ---------------------------
    if apply_model_updates:
        grads_model = tape.gradient(task_loss, model.model_vars)
        safe_grads_model = []
        for g, v in zip(grads_model, model.model_vars):
            if g is None:
                safe_grads_model.append(tf.zeros_like(v))
            else:
                safe_grads_model.append(tf.clip_by_norm(g, MODEL_GRAD_CLIP))
        optimizers["model"].apply_gradients(zip(safe_grads_model, model.model_vars))
    else:
        # compute grads for diagnostics but do not apply
        _ = tape.gradient(task_loss, model.model_vars)

    # ---------------------------
    # PERM updates (selective, preserve freeze logic)
    # ---------------------------
    perm_grad_norm = {
        "p1": tf.constant(0.0, dtype=tf.float32),
        "p2": tf.constant(0.0, dtype=tf.float32),
        "p3": tf.constant(0.0, dtype=tf.float32),
        "p4": tf.constant(0.0, dtype=tf.float32),
    }

    for i, name in enumerate(perm_names):
        if (not apply_perm_updates) or frozen_vars[name]:
            continue

        perm = perm_layers[i]
        single_perm_loss = perm_loss_vec[i]

        grads = tape.gradient(single_perm_loss, perm.trainable_variables)
        safe_grads = []
        total_norm = tf.constant(0.0, dtype=tf.float32)

        for g, v in zip(grads, perm.trainable_variables):
            if g is None:
                safe_grads.append(tf.zeros_like(v))
            else:
                clipped = tf.clip_by_norm(g, PERM_GRAD_CLIP)
                safe_grads.append(clipped)
                total_norm += tf.reduce_sum(tf.square(clipped))

        perm_grad_norm[name] = tf.sqrt(total_norm)
        optimizers["perm"][name].apply_gradients(zip(safe_grads, perm.trainable_variables))

    # release tape
    del tape

    # ---------------------------
    # Diagnostics: pairwise distances named dict
    # ---------------------------
    pairwise_dist = {}
    for i in range(len(perm_names)):
        for j in range(i + 1, len(perm_names)):
            key = f"{perm_names[i]}_{perm_names[j]}"
            pairwise_dist[key] = pairwise_abs_no_diag[i, j]

    # Row entropy as a dict keyed by perm name (keeps old API)
    row_entropy = {perm_names[i]: row_entropy_vec[i] for i in range(len(perm_names))}

    return {
        "loss": task_loss,
        "row_entropy": row_entropy,
        "perm_grad_norm": perm_grad_norm,
        "pairwise_dist": pairwise_dist,
    }


def epoch_controller_update(epoch,
                            epoch_mean_loss,
                            epoch_mean_entropies,   # dict {'p1':float,...}
                            steps_in_epoch,
                            apply_perm_updates=True,
                            eps=1e-8):
    """
    Epoch-driven controller update with strict staggering:
      - If it's an entropy-update epoch (stagger interval reached), SKIP temperature updates this epoch.
      - Otherwise run temperature updates (fast).
      - Entropy-weight updates only run on entropy-update epochs (slow).
      - Freeze counters are evaluated every epoch (but they will base decisions on entropies computed
        under the model/temperature regime that actually produced them).
    """

    # Convert inputs to tensors
    loss_tensor = tf.convert_to_tensor(epoch_mean_loss, dtype=tf.float32)
    loss_delta = tf.math.log(loss_tensor + eps) - tf.math.log(prev_loss + eps)

    do_entropy_update = False
    if apply_perm_updates:
        # Determine whether this epoch is an entropy-update epoch
        epoch_int = int(epoch) if not isinstance(epoch, tf.Tensor) else int(epoch.numpy())
        last_entropy_epoch_int = int(last_entropy_update_epoch.numpy())
        do_entropy_update = (epoch_int - last_entropy_epoch_int) >= ENTROPY_UPDATE_GAP
        
        # -----------------------
        # CASE A: Entropy-update epoch -> SKIP temperature updates this epoch
        # -----------------------
        if do_entropy_update:
            # We intentionally DO NOT change temperatures here.
            # Entropy update will run using entropies computed under the same temperatures
            # that were used for training during the epoch (clean signal).
            # Compute mean_entropy scalar for target computations
            mean_entropy_scalar = sum([epoch_mean_entropies[n] for n in epoch_mean_entropies]) / len(epoch_mean_entropies)

            # Entropy-weight updates (gentle multiplicative nudges)
            for name in ["p1", "p2", "p3", "p4"]:
                if frozen_vars[name]:
                    continue
                ent = epoch_mean_entropies[name]
                target = mean_entropy_scalar * 0.7
                # gentle update factor; ENTROPY_UPDATE_FACTOR is a global hyperparam
                if ent > target:
                    factor = ENTROPY_UPDATE_FACTOR
                else:
                    factor = 1.0 / ENTROPY_UPDATE_FACTOR
                new_w = tf.clip_by_value(entropy_weights[name] * factor, entropy_min, entropy_max)
                entropy_weights[name].assign(new_w)

            # commit last entropy update epoch
            last_entropy_update_epoch.assign(tf.cast(epoch_int, tf.int32))

            # Update prev_entropy to epoch-measured values AFTER entropy-weight update,
            # so next epoch's delta uses the correct baseline.
            for name in ["p1", "p2", "p3", "p4"]:
                prev_entropy[name].assign(tf.convert_to_tensor(epoch_mean_entropies[name], tf.float32))

            # (Temperature controllers are skipped this epoch) -> no temperature assignments here

        # -----------------------
        # CASE B: Regular epoch -> run temperature controllers (fast) and do not change entropy-weights
        # -----------------------
        else:
            for name in ["p1", "p2", "p3", "p4"]:
                if frozen_vars[name]:
                    # still update prev_entropy baseline so freeze detection has a reference
                    prev_entropy[name].assign(tf.convert_to_tensor(epoch_mean_entropies[name], tf.float32))
                    continue

                # build controller state from epoch-aggregated signals
                state = tf.stack([
                    loss_delta,
                    tf.convert_to_tensor(epoch_mean_entropies[name], tf.float32),
                    tf.convert_to_tensor(epoch_mean_entropies[name], tf.float32) - prev_entropy[name],
                    tf.math.log(temperatures[name] + eps),
                ])[None, :]

                # Train temperature controller in eager mode with small clipping
                with tf.GradientTape() as ctape:
                    pred_delta = temp_controllers[name](state)  # (1,1)
                    pred_delta = tf.clip_by_value(pred_delta, -MAX_CONTROLLER_DELTA, MAX_CONTROLLER_DELTA)
                    pred_temp = tf.clip_by_value(temperatures[name] * tf.exp(pred_delta[0,0]), temp_min, temp_max)

                    # simulate predicted entropy under predicted temp (use layer's sinkhorn in eager)
                    perm = getattr(model, f"permute{name[-1]}")
                    logits_stop = tf.stop_gradient(perm.logits)
                    if hasattr(perm, "_sinkhorn"):
                        P_pred = perm._sinkhorn(logits_stop / pred_temp)
                    else:
                        P_pred = _tf_log_sinkhorn(logits_stop / pred_temp, perm.num_iters)
                    P_pred = tf.clip_by_value(P_pred, SINKHORN_EPS, 1.0)
                    pred_entropy = -tf.reduce_mean(tf.reduce_sum(P_pred * tf.math.log(P_pred + 1e-12), axis=-1))

                    # target entropy based on epoch mean entropies
                    mean_entropy = sum([epoch_mean_entropies[n] for n in ["p1","p2","p3","p4"]]) / 4.0
                    target_entropy = mean_entropy * 0.7

                    controller_loss = tf.reduce_mean(tf.square(pred_entropy - target_entropy))

                # apply controller gradients
                c_vars = temp_controllers[name].trainable_variables
                if c_vars:
                    c_grads = ctape.gradient(controller_loss, c_vars)
                    safe_c_grads = []
                    for g, v in zip(c_grads, c_vars):
                        if g is None:
                            safe_c_grads.append(tf.zeros_like(v))
                        else:
                            safe_c_grads.append(tf.clip_by_norm(g, 1.0))
                    optimizers["temperature"][name].apply_gradients(zip(safe_c_grads, c_vars))

                # commit temperature change (fast update)
                temperatures[name].assign(pred_temp)

                # update prev_entropy baseline for next epoch delta calculation
                prev_entropy[name].assign(tf.convert_to_tensor(epoch_mean_entropies[name], tf.float32))

            # Note: entropy_weights are NOT updated on this path

    # -----------------------
    # Freeze counters and freeze decisions are evaluated every epoch (same behavior)
    # -----------------------
    for name in ["p1","p2","p3","p4"]:
        ent = epoch_mean_entropies[name]
        stable = tf.logical_and(ent < ENTROPY_FREEZE_THRESH,
                                tf.abs(ent - prev_entropy[name]) < 1e-6)
        prev_cnt = entropy_stable_count[name]
        entropy_stable_count[name].assign(tf.where(stable, prev_cnt + 1, tf.constant(0)))

        should_freeze = tf.logical_and(
            entropy_stable_count[name] >= STABLE_REQUIRED,
            tf.clip_by_value(temperatures[name], temp_min, temp_max) < TEMP_FREEZE_THRESH
        )

        if should_freeze and not frozen_vars[name].numpy():
            frozen_vars[name].assign(True)
            # Optionally: reduce perm optimizer LR to near-zero to softly freeze instead of hard stop.
            # opt = optimizers['perm'][name]; opt.learning_rate.assign(0.0)

    # commit prev_loss for next epoch
    prev_loss.assign(loss_tensor)

    # return diagnostics
    return {
        "epoch_loss": float(loss_tensor.numpy()),
        "epoch_ent": {n: float(epoch_mean_entropies[n]) for n in epoch_mean_entropies},
        "temperatures": {n: float(temperatures[n].numpy()) for n in temperatures},
        "entropy_weights": {n: float(entropy_weights[n].numpy()) for n in entropy_weights},
        "frozen_vars": {n: bool(frozen_vars[n].numpy()) for n in frozen_vars},
        "last_entropy_update_epoch": int(last_entropy_update_epoch.numpy()),
        "entropy_update_happened": bool(do_entropy_update),
    }
    
# Run once, in eager mode, after model and optimizers are created but BEFORE the first call to train_step.
def initialize_optimizer_slots():
    # Model optimizer (may create Adam slots for all model vars)
    model_vars = model.model_vars
    if len(model_vars) > 0:
        zero_grads = [tf.zeros_like(v) for v in model_vars]
        optimizers["model"].apply_gradients(zip(zero_grads, model_vars))

    # Permutation optimizers (per branch)
    for name, opt in optimizers["perm"].items():
        perm_layer = getattr(model, f"permute{name[-1]}")
        vars_list = perm_layer.trainable_variables
        if len(vars_list) > 0:
            zero_grads = [tf.zeros_like(v) for v in vars_list]
            opt.apply_gradients(zip(zero_grads, vars_list))

    # Temperature/controller optimizers
    for name, opt in optimizers["temperature"].items():
        c_vars = temp_controllers[name].trainable_variables
        if c_vars:
            zero_grads = [tf.zeros_like(v) for v in c_vars]
            opt.apply_gradients(zip(zero_grads, c_vars))

    print("\nOptimizer slot variables initialized.")

def snapshot_permutation_state():
    """Copy perm logits, temps, entropy_weights to a plain dict for revert if needed."""
    return {
        "p1": model.permute1.logits.numpy().copy(),
        "p2": model.permute2.logits.numpy().copy(),
        "p3": model.permute3.logits.numpy().copy(),
        "p4": model.permute4.logits.numpy().copy(),
        "temps": {n: float(temperatures[n].numpy()) for n in temperatures},
        "entropy_weights": {n: float(entropy_weights[n].numpy()) for n in entropy_weights},
    }

def revert_permutation_state(snapshot):
    """Re-assign logits, temps and entropy weights from a snapshot."""
    model.permute1.logits.assign(snapshot["p1"])
    model.permute2.logits.assign(snapshot["p2"])
    model.permute3.logits.assign(snapshot["p3"])
    model.permute4.logits.assign(snapshot["p4"])
    for n in snapshot["temps"]:
        temperatures[n].assign(snapshot["temps"][n])
    for n in snapshot["entropy_weights"]:
        entropy_weights[n].assign(snapshot["entropy_weights"][n])

def simple_propose_gradient_step(step_size=1.0, batch=None, clip_norm=1.0):
    """
    Simple proposal: single small gradient step on the *sum* perm losses using one batch.
    If batch not provided, grabs next batch from train_ds iterator.
    This performs an in-place update to perm logits.
    """
    if batch is None:
        batch = next(iter(train_ds))
    xb, yb = batch
    with tf.GradientTape() as t:
        y_pred = model(xb, training=True)
        task_loss = loss_fn(yb, y_pred)
        # compute proxy perm losses (same structure as inside train_step)
        perm_losses = []
        for name in ["p1","p2","p3","p4"]:
            perm = getattr(model, f"permute{name[-1]}")
            temp = tf.clip_by_value(temperatures[name], temp_min, temp_max)
            P = perm._sinkhorn(perm.logits / temp)
            row_ent = safe_entropy(P, 1e-8)
            perm_losses.append(task_loss + entropy_weights[name] * row_ent)
        obj = tf.add_n(perm_losses)
    # grads wrt all perm logits (perm.logits are first var in each perm.trainable_variables here)
    perm_vars = model.permute1.trainable_variables + model.permute2.trainable_variables + model.permute3.trainable_variables + model.permute4.trainable_variables
    grads = t.gradient(obj, perm_vars)
    # apply small manual step
    idx = 0
    lr = 1e-3 * step_size
    for perm_name in ["p1","p2","p3","p4"]:
        perm = getattr(model, f"permute{perm_name[-1]}")
        g = grads[idx]
        idx += 1
        if g is None:
            continue
        perm.logits.assign(perm.logits - lr * tf.clip_by_norm(g, clip_norm))

# @tf.function
def run_epochs(num_epochs, apply_model_updates=True, apply_perm_updates=True, start_epoch=0):
    """
    Run `num_epochs` epochs, using train_step with flags.
    Returns a list of epoch summaries (dicts) for the run.
    """
    summaries = []
    for e in range(num_epochs):
        epoch_loss_sum = 0.0
        epoch_ent_sums = {n: 0.0 for n in ["p1","p2","p3","p4"]}
        steps_local = 0
        for xb, yb in train_ds:
            res = train_step(xb, yb, apply_model_updates=apply_model_updates, apply_perm_updates=apply_perm_updates)
            # increment step counter
            step.assign_add(1)
            epoch_loss_sum += float(res["loss"].numpy())
            steps_local += 1
            for n in ["p1","p2","p3","p4"]:
                epoch_ent_sums[n] += float(res["row_entropy"][n].numpy())
        mean_loss = epoch_loss_sum / max(1, steps_local)
        mean_ents = {n: epoch_ent_sums[n] / max(1, steps_local) for n in epoch_ent_sums}
        # update controllers on epoch boundary using the aggregated stats
        epoch_controller_update(start_epoch + e, mean_loss, mean_ents, steps_local, apply_perm_updates=apply_perm_updates)
        summaries.append({"epoch": start_epoch + e, "train_loss": mean_loss, "entropies": mean_ents})
    return summaries

def evaluate_on_val():
    """Return (val_loss, mae) averaged over val_ds."""
    metric.reset_state()
    val_loss_acc = 0.0
    val_steps = 0
    for xv, yv in val_ds:
        vp = model(xv, training=False)
        val_loss_acc += float(loss_fn(yv, vp).numpy())
        metric.update_state(yv, vp)
        val_steps += 1
    return val_loss_acc / max(1, val_steps), float(metric.result().numpy())

def reinit_optimizers_and_slots():
    """
    Optional helper: re-create optimizer objects and reinitialize their slots.
    Called if we want to fully reset optimizer internal state after a big perm change.
    """
    # recreate optimizer instances
    global optimizers
    optimizers["model"] = tf.keras.optimizers.AdamW(1e-3)
    optimizers["perm"]["p1"] = tf.keras.optimizers.AdamW(1e-5)
    optimizers["perm"]["p2"] = tf.keras.optimizers.AdamW(1e-5)
    optimizers["perm"]["p3"] = tf.keras.optimizers.AdamW(1e-5)
    optimizers["perm"]["p4"] = tf.keras.optimizers.AdamW(1e-5)
    optimizers["temperature"]["p1"] = tf.keras.optimizers.AdamW(1e-4)
    optimizers["temperature"]["p2"] = tf.keras.optimizers.AdamW(1e-4)
    optimizers["temperature"]["p3"] = tf.keras.optimizers.AdamW(1e-4)
    optimizers["temperature"]["p4"] = tf.keras.optimizers.AdamW(1e-4)
    # re-run slot initializer
    initialize_optimizer_slots()

# -----------------------------
# High-level permutation-search driver
# -----------------------------
def permutation_search_cycle(total_cycles=50,
                             baseline_epochs=3,
                             adaptation_epochs=6,
                             accept_rel_improve=0.003,
                             patience_epochs=2,
                             save_dir="perm_search_checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    global epoch_global, best_val_loss

    for cycle in range(total_cycles):
        print(f"\n=== CYCLE {cycle} — BASELINE ({baseline_epochs} epochs) ===")
        # baseline: let model train with fixed permutations (no perm updates)
        run_epochs(baseline_epochs, apply_model_updates=True, apply_perm_updates=False, start_epoch=epoch_global)
        epoch_global += baseline_epochs

        baseline_val, baseline_mae = evaluate_on_val()
        print(f"Baseline val_loss: {baseline_val:.6g} mae: {baseline_mae:.6g}")

        # snapshot state before proposing
        snap = snapshot_permutation_state()

        # propose a small permutation change (gradient step or other)
        print("Proposing permutation update...")
        simple_propose_gradient_step(step_size=1.0)

        # adaptation: let the model adapt to the proposed permutation (perms now fixed during adaptation)
        print(f"Adaptation ({adaptation_epochs} epochs)...")
        run_epochs(adaptation_epochs, apply_model_updates=True, apply_perm_updates=False, start_epoch=epoch_global)
        epoch_global += adaptation_epochs

        new_val, new_mae = evaluate_on_val()
        print(f"Post-adapt val_loss: {new_val:.6g} mae: {new_mae:.6g}")

        rel_imp = (baseline_val - new_val) / max(1e-12, baseline_val)
        print(f"Relative improvement: {rel_imp:.6%}")

        accepted = False
        if rel_imp >= accept_rel_improve:
            # stability quick check
            print("Tentatively accepted. Running stability window...")
            run_epochs(patience_epochs, apply_model_updates=True, apply_perm_updates=False, start_epoch=epoch_global)
            epoch_global += patience_epochs
            val_after = evaluate_on_val()[0]
            if val_after <= new_val * 1.001:
                accepted = True
                print("Permutation accepted.")
                # save accepted permutation and model weights
                np.save(os.path.join(save_dir, f"perms_accepted_cycle_{cycle}.npy"), {
                    "p1": model.permute1.logits.numpy(),
                    "p2": model.permute2.logits.numpy(),
                    "p3": model.permute3.logits.numpy(),
                    "p4": model.permute4.logits.numpy()
                })
                model.save_weights(os.path.join(save_dir, f"model_after_cycle_{cycle}.weights.h5"))
            else:
                print("Stability check failed. Reverting.")
        else:
            print("Rejected: insufficient improvement.")

        if not accepted:
            revert_permutation_state(snap)
            # Optionally reinitialize optimizer state if a clean slate is required:
            # reinit_optimizers_and_slots()

        # optional: run temperature-controller update after accepted permutation
        if accepted:
            # we must supply epoch-aggregated stats from the last adaptation epoch for a correct update;
            # for brevity, compute a fresh evaluation as input (controller uses epoch metrics)
            val_loss_now, _ = evaluate_on_val()
            # compute dummy entropies for controller update (use current perms)
            ent_summ = {}
            for name in ["p1","p2","p3","p4"]:
                Pcur = getattr(model, f"permute{name[-1]}")._sinkhorn(getattr(model, f"permute{name[-1]}").logits / temperatures[name])
                ent_summ[name] = float(safe_entropy(Pcur, 1e-8).numpy())
            epoch_controller_update(epoch_global, val_loss_now, ent_summ, steps_in_epoch=1, apply_perm_updates=False)

        # update best_val_loss
        if new_val < best_val_loss:
            best_val_loss = new_val
            print("New best validation loss:", best_val_loss)

        print(f"End of cycle {cycle}; epoch_global={epoch_global}")

    print("Permutation search complete.")    

WL = 8
DIMS = 4
EPOCHS = 10_000
NUM_SAMPLES = 10_000
DATASET = 'USDCHF_M1_245.csv'
PATH = f'datasets/{DATASET}'

X, y, features = load_data(PATH, WL, NUM_SAMPLES)

split = int(0.9 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

BATCH_SIZE = 32 #compute_batch_size(len(X_train))

norm = tf.keras.layers.Normalization()
norm.adapt(X_train)

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_ds = val_ds.batch(BATCH_SIZE)

model = TimeSeriesModel(num_features=len(features), dims=DIMS, norm=norm)

loss_fn = tf.keras.losses.Huber()
metric  = tf.keras.metrics.MeanAbsoluteError()

# Optimizers
opt_model = tf.keras.optimizers.AdamW(1e-3)

opt_perm_p1 = tf.keras.optimizers.AdamW(1e-5)
opt_perm_p2 = tf.keras.optimizers.AdamW(1e-5)
opt_perm_p3 = tf.keras.optimizers.AdamW(1e-5)
opt_perm_p4 = tf.keras.optimizers.AdamW(1e-5)

opt_temp_p1 = tf.keras.optimizers.AdamW(1e-4)
opt_temp_p2 = tf.keras.optimizers.AdamW(1e-4)
opt_temp_p3 = tf.keras.optimizers.AdamW(1e-4)
opt_temp_p4 = tf.keras.optimizers.AdamW(1e-4)

# Meta Controllers
ent_controller1 = MetaController(a_max=0.2, dim=4)
ent_controller2 = MetaController(a_max=0.2, dim=4)
ent_controller3 = MetaController(a_max=0.2, dim=4)
ent_controller4 = MetaController(a_max=0.2, dim=4)

temp_controller1 = MetaController(a_max=0.1, dim=4)
temp_controller2 = MetaController(a_max=0.1, dim=4)
temp_controller3 = MetaController(a_max=0.1, dim=4)
temp_controller4 = MetaController(a_max=0.1, dim=4)

optimizers = {
    "model": opt_model,

    "perm": {
        "p1": opt_perm_p1,
        "p2": opt_perm_p2,
        "p3": opt_perm_p3,
        "p4": opt_perm_p4,
    },

    "temperature": {
        "p1": opt_temp_p1,
        "p2": opt_temp_p2,
        "p3": opt_temp_p3,
        "p4": opt_temp_p4,
    },
}

entropy_weights = {
    "p1": tf.Variable(1e-3, trainable=False),
    "p2": tf.Variable(1e-3, trainable=False),
    "p3": tf.Variable(1e-3, trainable=False),
    "p4": tf.Variable(1e-3, trainable=False),
}

temperatures = {
    "p1": model.permute1.temperature,
    "p2": model.permute2.temperature,
    "p3": model.permute3.temperature,
    "p4": model.permute4.temperature,
}

temp_controllers = {
    "p1": temp_controller1,
    "p2": temp_controller2,
    "p3": temp_controller3,
    "p4": temp_controller4,
}

entropy_controllers = {
    "p1": ent_controller1,
    "p2": ent_controller2,
    "p3": ent_controller3,
    "p4": ent_controller4,
}

# Memory
prev_loss = tf.Variable(1e9, trainable=False)
prev_entropy = {
    "p1": tf.Variable(0.0, trainable=False),
    "p2": tf.Variable(0.0, trainable=False),
    "p3": tf.Variable(0.0, trainable=False),
    "p4": tf.Variable(0.0, trainable=False),
}
prev_perm_grad_norm = {
    "p1": tf.Variable(0.0, trainable=False),
    "p2": tf.Variable(0.0, trainable=False),
    "p3": tf.Variable(0.0, trainable=False),
    "p4": tf.Variable(0.0, trainable=False),
}

entropy_min, entropy_max = 1e-6, 1.0
temp_min, temp_max = 5e-3, 5.0

BETA_ADV = 0.1
GAMMA_REPEL = 0.1

TEMP_INTERVAL = 6
ENTROPY_INTERVAL = 50
assert ENTROPY_INTERVAL % TEMP_INTERVAL != 0

ENTROPY_FREEZE_THRESH = 5e-4
TEMP_FREEZE_THRESH = 5e-2

PERM_GRAD_CLIP = 1.0
SINKHORN_EPS = 1e-6

MODEL_GRAD_CLIP = 5.0

# Controller hyperparams
TARGET_ENTROPY_FACTOR = 0.7  # want each perm entropy ~ 0.7 * mean_entropy
MAX_CONTROLLER_DELTA = 0.05   # bound controller output
TEMP_STEP = 1.08              # deterministic step fallback
ENT_STEP = 1.10               # deterministic entropy-weight step
CONTROLLER_GRAD_CLIP = 1.0

# Stability-based freeze counters
entropy_stable_count = {
"p1": tf.Variable(0, trainable=False, dtype=tf.int32),
"p2": tf.Variable(0, trainable=False, dtype=tf.int32),
"p3": tf.Variable(0, trainable=False, dtype=tf.int32),
"p4": tf.Variable(0, trainable=False, dtype=tf.int32),
}

# Tune this to require N consecutive "stable" steps before freezing
STABLE_REQUIRED = 100

# Clip magnitude for controller predicted delta (small negative values prevent collapse)
# controller can decrease temp by at most ~1% per controller update
MAX_CONTROLLER_DELTA = 0.01

frozen_vars = {
    "p1": tf.Variable(False, trainable=False),
    "p2": tf.Variable(False, trainable=False),
    "p3": tf.Variable(False, trainable=False),
    "p4": tf.Variable(False, trainable=False),
}

# Keep MAX_CONTROLLER_DELTA small to avoid single-epoch catastrophes
MAX_CONTROLLER_DELTA = 0.01

# Track last epoch where entropy weights were updated
last_entropy_update_epoch = tf.Variable(-999, trainable=False, dtype=tf.int32)

# Keep small inertia factor when entropy updates happen (so it's gentle)
ENTROPY_UPDATE_FACTOR = 1.05  # multiplicative factor; use >1 to increase when ent high, <1 to decrease.

# -----------------------------
# Outer loop: Alternating phases (perm-phase then model-phase)
# -----------------------------
# Tuning hyperparams
ENTROPY_UPDATE_GAP = 5   # try 3-10; 5 is a safe default
PERM_PHASE_EPOCHS = 5    # number of epochs to run perm-search
MODEL_PHASE_EPOCHS = 10   # number of epochs to train model between perm-searches
# cycle_len = PERM_PHASE_EPOCHS + MODEL_PHASE_EPOCHS
TOTAL_EPOCHS = EPOCHS
assert PERM_PHASE_EPOCHS >= 1 and MODEL_PHASE_EPOCHS >= 1

# step = tf.Variable(0, dtype=tf.int32) if 'step' not in globals() else step
best_val_loss = np.inf
history = {
    'epoch': [], 'phase': [], 'epoch_loss': [], 'val_loss': [], 'mae': [],
    'ent_p1': [], 'ent_p2': [], 'ent_p3': [], 'ent_p4': [],
    'temp_p1': [], 'temp_p2': [], 'temp_p3': [], 'temp_p4': [],
    'entw_p1': [], 'entw_p2': [], 'entw_p3': [], 'entw_p4': [],
    'frozen': []
}

# ensure model variables exist
dummy_b = 1
dummy_T = WL           # window length
dummy_F = len(features)  # number of features
_ = model(tf.zeros([dummy_b, dummy_T, dummy_F], dtype=tf.float32))  # builds layers

# Also call controllers once
for name in ["p1","p2","p3","p4"]:
    # ensure perm layers built
    _ = getattr(model, f"permute{name[-1]}").logits  # logits variable exists by add_weight in layer
    # ensure controller submodels exist (if not yet called)
    _ = temp_controllers[name](tf.zeros([1,4], dtype=tf.float32))
    _ = entropy_controllers[name](tf.zeros([1,4], dtype=tf.float32))


# Called once to initialize optimizers outside @tf.function
initialize_optimizer_slots()

step = tf.Variable(0, dtype=tf.int32)
epoch_global = 0



# Run the search loop instead of the previous while loop
permutation_search_cycle(total_cycles=10_000,
                         baseline_epochs=10,
                         adaptation_epochs=10,
                         accept_rel_improve=0.003,
                         patience_epochs=10,
                         save_dir="perm_search_checkpoints")


# -----------------------------
# After training: save history for plotting/analysis
# -----------------------------
import json
with open("training_history.json", "w") as fh:
    json.dump(history, fh)

print("Training finished. History saved to training_history.json")
