from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers


@dataclass
class TransformerConfig:
    d_model: int = 64
    num_heads: int = 4
    num_layers: int = 2
    ff_dim: int = 128
    dropout_rate: float = 0.1
    learning_rate: float = 0.0003
    weight_decay: float = 1e-4
    batch_size: int = 128
    epochs: int = 100
    patience: int = 15
    validation_split: float = 0.2
    random_state: int = 42
    use_class_weights: bool = True


@dataclass
class TransformerResult:
    model: keras.Model
    history: dict
    y_pred: np.ndarray
    y_pred_proba: np.ndarray
    y_test: np.ndarray
    test_accuracy: float
    test_loss: float
    config: TransformerConfig


class _LearnablePositionalEmbedding(layers.Layer):
    """Adds a learnable position vector to each token in the sequence."""

    def __init__(self, seq_len: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.d_model = d_model

    def build(self, input_shape):
        self.pos_emb = self.add_weight(
            name="pos_embedding",
            shape=(self.seq_len, self.d_model),
            initializer="random_normal",
            trainable=True,
        )

    def call(self, x):
        return x + self.pos_emb

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"seq_len": self.seq_len, "d_model": self.d_model})
        return cfg


class _PrependCLSToken(layers.Layer):
    """Prepends a learnable [CLS] token to a sequence."""

    def __init__(self, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model

    def build(self, input_shape):
        self.cls_token = self.add_weight(
            name="cls_token",
            shape=(1, 1, self.d_model),
            initializer="zeros",
            trainable=True,
        )

    def call(self, x):
        cls = tf.tile(self.cls_token, [tf.shape(x)[0], 1, 1])
        return tf.concat([cls, x], axis=1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"d_model": self.d_model})
        return cfg


class _ExtractFirstToken(layers.Layer):
    """Extracts the [CLS] token (position 0) from a sequence."""

    def call(self, x):
        return x[:, 0, :]


def reshape_for_transformer(X: np.ndarray) -> np.ndarray:
    """
    (N, 25) → (N, 5, 5): 5 channels as tokens, 5 bands as features per token.

    Column layout is channel-major: [AF3_Theta…AF3_Gamma, AF4_Theta…Pz_Gamma].
    reshape(-1, 5, 5) yields (N, 5_channels, 5_bands) directly — no transpose.
    Attention operates across channels; each token carries the full spectral profile
    of one electrode.
    """
    return X.reshape(-1, 5, 5)


def build_transformer_model(config: TransformerConfig) -> keras.Model:
    """
    Encoder-only Transformer for EEG spectral classification.

    Architecture:
      Input (5 channels × 5 bands)
      → Linear embedding (5 → d_model)
      → Learnable positional embedding
      → Prepend [CLS] token  →  sequence length: 6
      → N × (pre-norm self-attention + pre-norm FFN)
      → Extract [CLS] token
      → LayerNorm → Dropout → Dense(3, softmax)
    """
    inputs = layers.Input(shape=(5, 5))

    # Project each channel token from dim=5 (bands) to d_model
    x = layers.Dense(config.d_model)(inputs)

    # Learnable positional embedding — one vector per channel position
    x = _LearnablePositionalEmbedding(seq_len=5, d_model=config.d_model)(x)

    # Prepend [CLS] token → sequence of 6
    x = _PrependCLSToken(config.d_model)(x)

    # Transformer encoder blocks
    for _ in range(config.num_layers):
        # Pre-norm self-attention
        residual = x
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.MultiHeadAttention(
            num_heads=config.num_heads,
            key_dim=config.d_model // config.num_heads,
            dropout=config.dropout_rate,
        )(x, x)
        x = layers.Dropout(config.dropout_rate)(x)
        x = layers.Add()([residual, x])

        # Pre-norm FFN
        residual = x
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.Dense(config.ff_dim, activation="relu")(x)
        x = layers.Dropout(config.dropout_rate)(x)
        x = layers.Dense(config.d_model)(x)
        x = layers.Add()([residual, x])

    # Extract [CLS] token and classify
    x = _ExtractFirstToken()(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dropout(config.dropout_rate)(x)
    outputs = layers.Dense(3, activation="softmax")(x)

    return keras.Model(inputs=inputs, outputs=outputs)


def train_transformer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: TransformerConfig | None = None,
) -> TransformerResult:
    if config is None:
        config = TransformerConfig()

    tf.random.set_seed(config.random_state)
    np.random.seed(config.random_state)

    X_train_t = reshape_for_transformer(X_train)
    X_test_t = reshape_for_transformer(X_test)

    class_weight = None
    if config.use_class_weights:
        weights = compute_class_weight(
            "balanced",
            classes=np.unique(y_train),
            y=y_train,
        )
        class_weight = dict(enumerate(weights))

    model = build_transformer_model(config)
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=config.patience,
        restore_best_weights=True,
    )
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )

    history = model.fit(
        X_train_t,
        y_train,
        validation_split=config.validation_split,
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=[early_stopping, lr_scheduler],
        class_weight=class_weight,
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(X_test_t, y_test, verbose=0)
    y_pred_proba = model.predict(X_test_t, verbose=0)
    y_pred = y_pred_proba.argmax(axis=1)

    return TransformerResult(
        model=model,
        history=history.history,
        y_pred=y_pred,
        y_pred_proba=y_pred_proba,
        y_test=y_test,
        test_accuracy=test_accuracy,
        test_loss=test_loss,
        config=config,
    )
