# feature_ordering_v25.py

import pandas as pd
import numpy as np
import tensorflow as tf
import random
import math

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
        P = tf.exp(log_alpha)
        for _ in range(self.num_iters):
            P /= tf.reduce_sum(P, axis=-1, keepdims=True) + 1e-8
            P /= tf.reduce_sum(P, axis=-2, keepdims=True) + 1e-8
        return P

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
            tf.keras.layers.Dense(8, activation="gelu"),
            tf.keras.layers.Dense(4, activation="gelu"),
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
        # self.gru2 = tf.keras.layers.GRU(dims, return_sequences=True, seed=SEED+1)
        # self.lstm = tf.keras.layers.LSTM(dims, return_sequences=True, seed=SEED)
        # self.conv1 = tf.keras.layers.Conv1D(dims, 3, padding='same')
        # self.conv2 = tf.keras.layers.Conv1D(dims, 3, padding='same')
        # self.act1 = tf.keras.layers.Activation('gelu')
        # self.act2 = tf.keras.layers.Activation('gelu')
        # self.t_dense = tf.keras.layers.TimeDistributed(
        #     tf.keras.layers.Dense(dims, activation='gelu'))
        
        # self.conc = tf.keras.layers.Concatenate()
        # self.dense = tf.keras.layers.Dense(4, activation='softmax')
        # self.add = tf.keras.layers.Add()
        self.head = tf.keras.layers.Dense(1)

        self.drop1 = tf.keras.layers.Dropout(0.0)
        # self.drop2 = tf.keras.layers.Dropout(0.1)
        # self.drop3 = tf.keras.layers.Dropout(0.1)
        # self.drop4 = tf.keras.layers.Dropout(0.1)
        
    def call(self, x, training=False):
        x = self.norm(x)
        
        x1 = self.gru1(self.permute1(x))
        # x1= self.act1(x1)
        x = self.drop1(x1)
        
        # x2 = self.lstm(self.permute2(x))
        # x2 = self.drop2(x2)
        
        # x3 = self.gru2(self.permute3(x))
        # # x3= self.act2(x3)
        # x3 = self.drop3(x3)
        
        # x4 = self.t_dense(self.permute4(x))
        # x4 = self.drop4(x4)
        
        # comb = [x1,x2,x3,x4]
        # fusion = self.conc(comb)
        # weights = self.dense(fusion)

        # w1 = weights[:, :, 0:1]
        # w2 = weights[:, :, 1:2]
        # w3 = weights[:, :, 2:3]
        # w4 = weights[:, :, 3:4]
        
        # x = self.add([x1*w1, x2*w2, x3*w3, x4*w4])
        x = self.agr(x)
        return self.head(x)

    @property
    def perm_vars(self):
        return (
            self.permute1.trainable_variables
            # self.permute2.trainable_variables +
            # self.permute3.trainable_variables +
            # self.permute4.trainable_variables
        )

    @property
    def model_vars(self):
        return (
            self.agr.trainable_variables +
            self.gru1.trainable_variables +
            # self.lstm.trainable_variables +
            # self.gru2.trainable_variables +
            # self.t_dense.trainable_variables +
            # self.dense.trainable_variables +
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
    
def load_data(PATH, WL, NUM_SAMPLES, extra_features=True):
    data = load_csv_file(PATH, NUM_SAMPLES)
    if extra_features:
        df = create_features(data.copy(), WL)
    else:
        df = data.copy()
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
def train_step(x, y, step, eps=1e-8):
    with tf.GradientTape(persistent=True) as tape:
        y_pred = model(x, training=True)
        task_loss = loss_fn(y, y_pred)

        P = {}
        row_entropy = {}
        frozen = {}

        # ---------- Build permutations ----------
        for name, perm in zip(
            ['p1'],
            [model.permute1] #, model.permute2, model.permute3, model.permute4],
        ):
            temp = tf.clip_by_value(temperatures[name], temp_min, temp_max)
            logits = perm.logits / temp

            Pi = perm._sinkhorn(logits)
            Pi = tf.clip_by_value(Pi, SINKHORN_EPS, 1.0)

            P[name] = Pi
            row_entropy[name] = safe_entropy(Pi, eps)

            frozen[name] = tf.logical_and(
                row_entropy[name] < ENTROPY_FREEZE_THRESH,
                temp < TEMP_FREEZE_THRESH,
            )

        # ---------- Repulsion ----------
        repulsion = {}
        for i in P:
            repulsion[i] = 0.0
            for j in P:
                if i != j:
                    repulsion[i] += tf.reduce_mean(tf.abs(P[i] - tf.stop_gradient(P[j])))

        # ---------- Advantage (relative entropy contribution proxy) ----------
        mean_entropy = tf.add_n(list(row_entropy.values())) / len(row_entropy)
        advantage = {
            k: mean_entropy - row_entropy[k]
            for k in row_entropy
        }

        # ---------- Branch-specific permutation losses ----------
        perm_loss = {}
        for name in P:
            perm_loss[name] = (
                task_loss
                + entropy_weights[name] * row_entropy[name]
                - BETA_ADV * advantage[name]
                + GAMMA_REPEL * repulsion[name]
            )

    # ---------- Model update ----------
    grads_model = tape.gradient(task_loss, model.model_vars)
    optimizers["model"].apply_gradients(
        zip(grads_model, model.model_vars)
    )
    
    # ---------- Permutation updates ----------
    for name in P:
        if frozen[name]:
            continue

        perm = getattr(model, f"permute{name[-1]}")
        # use specific scalar loss for this perm
        single_perm_loss = perm_loss[name]
        grads = tape.gradient(single_perm_loss, perm.trainable_variables)

        # Replace None grads and clip
        safe_grads = []
        for g, v in zip(grads, perm.trainable_variables):
            if g is None:
                safe_grads.append(tf.zeros_like(v))
            else:
                safe_grads.append(tf.clip_by_norm(g, PERM_GRAD_CLIP))

        optimizers["perm"][name].apply_gradients(zip(safe_grads, perm.trainable_variables))
        
    # release persistent tape to avoid memory leaks
    del tape

    # Calculate loss delta for later use
    loss_delta = tf.math.log(task_loss + eps) - tf.math.log(prev_loss + eps)
    prev_loss.assign(task_loss)
    
    # ---------- Temperature controllers ----------
    if tf.equal(step % TEMP_INTERVAL, 0):
        for name in P:
            if frozen[name]:
                continue
                
            state = tf.stack([
                loss_delta,
                row_entropy[name],
                row_entropy[name] - prev_entropy[name],
                # tf.math.log(perm_grad_norm[name] + eps),
                tf.math.log(temperatures[name] + eps),
            ])[None, :]

            delta = temp_controllers[name](state)
            delta = tf.clip_by_value(delta, -0.05, 0.05)
            
            temperatures[name].assign(
                tf.clip_by_value(
                    temperatures[name] * tf.exp(delta[0, 0]),
                    temp_min,
                    temp_max
                )
            )

            prev_entropy[name].assign(row_entropy[name])

    # ---------- Entropy controllers ----------
    if tf.equal(step % ENTROPY_INTERVAL, 0):
        # loss_delta = tf.math.log(task_loss + eps) - tf.math.log(prev_loss + eps)
        for name in P:
            if frozen[name]:
                continue
                
            state = tf.stack([
                loss_delta,
                row_entropy[name],
                row_entropy[name] - prev_entropy[name],
                tf.math.log(task_loss + eps),               
                tf.math.log(temperatures[name] + eps),
            ])[None, :]

            delta = entropy_controllers[name](state)
            delta = tf.clip_by_value(delta, -0.05, 0.05)
            
            entropy_weights[name].assign(
                tf.clip_by_value(
                    entropy_weights[name] * tf.exp(delta[0, 0]),
                    entropy_min,
                    entropy_max
                )
            )

            prev_entropy[name].assign(row_entropy[name])
        # prev_loss.assign(task_loss)

    return {
        "loss": task_loss,
        "row_entropy": row_entropy,
        "entropy_weights": entropy_weights,
        "temperatures": temperatures,
        "repulsion": repulsion,
        "advantage": advantage,
        "frozen": frozen,
    }


WL = 8
DIMS = 4
EPOCHS = 10_000
NUM_SAMPLES = 10_000
DATASET = 'USDCHF_M1_245.csv'
PATH = f'datasets/{DATASET}'
BRANCHES = ['p1'] #, 'p2', 'p3, 'p4']

X, y, features = load_data(PATH, WL, NUM_SAMPLES, extra_features=False)

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
ent_controller1 = MetaController(a_max=0.2, dim=5)
ent_controller2 = MetaController(a_max=0.2, dim=5)
ent_controller3 = MetaController(a_max=0.2, dim=5)
ent_controller4 = MetaController(a_max=0.2, dim=5)

temp_controller1 = MetaController(a_max=0.1, dim=5)
temp_controller2 = MetaController(a_max=0.1, dim=5)
temp_controller3 = MetaController(a_max=0.1, dim=5)
temp_controller4 = MetaController(a_max=0.1, dim=5)

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
temp_min, temp_max = 1e-3, 5.0

BETA_ADV = 0.1
GAMMA_REPEL = 0.01

TEMP_INTERVAL = 10
ENTROPY_INTERVAL = 25
assert ENTROPY_INTERVAL % TEMP_INTERVAL != 0

ENTROPY_FREEZE_THRESH = 5e-4
TEMP_FREEZE_THRESH = 5e-2

PERM_GRAD_CLIP = 1.0
SINKHORN_EPS = 1e-6

step = 0

for epoch in range(EPOCHS):
    for x_batch, y_batch in train_ds:
        results = train_step(x_batch, y_batch, tf.constant(step))
        step += 1

    loss  = results['loss'].numpy()
    
    ent1  = results['row_entropy']['p1'].numpy()
    # ent2  = results['row_entropy']['p2'].numpy()
    # ent3  = results['row_entropy']['p3'].numpy()
    # ent4  = results['row_entropy']['p4'].numpy()
    
    ent_w1 = results['entropy_weights']['p1'].numpy()
    # ent_w2 = results['entropy_weights']['p2'].numpy()
    # ent_w3 = results['entropy_weights']['p3'].numpy()
    # ent_w4 = results['entropy_weights']['p4'].numpy()

    temp1 = results['temperatures']['p1'].numpy()
    # temp2 = results['temperatures']['p2'].numpy()
    # temp3 = results['temperatures']['p3'].numpy()
    # temp4 = results['temperatures']['p4'].numpy()

    frozen1 = results['frozen']['p1']
    # frozen2 = results['frozen']['p2']
    # frozen3 = results['frozen']['p3']
    # frozen4 = results['frozen']['p4']
    
    metric.reset_state()
    
    for x_val, y_val in val_ds:
        val_pred = model(x_val, training=False)
        val_loss = loss_fn(y_val, val_pred)
        metric.update_state(y_val, val_pred)
        
    mae = metric.result()
    lr = get_lr(opt_model)

    print(f"\nEpoch {epoch}\nloss: {loss:.3g} | val_loss: {val_loss.numpy():.3g} | mae: {mae.numpy():.3g} | lr: {lr:.3e}\n"
          f"P1:\tEntropy: {ent1:.3g} | Entropy Weight: {ent_w1:.3g} | Temp: {temp1:.3g} | Frozen: {frozen1}\n"
          # f"P2:\tEntropy: {ent2:.3g} | Entropy Weight: {ent_w2:.3g} | Temp: {temp2:.3g} | Frozen: {frozen2}\n"
          # f"P3:\tEntropy: {ent3:.3g} | Entropy Weight: {ent_w3:.3g} | Temp: {temp3:.3g} | Frozen: {frozen3}\n"
          # f"P4:\tEntropy: {ent4:.3g} | Entropy Weight: {ent_w4:.3g} | Temp: {temp4:.3g} | Frozen: {frozen4}"
          )


