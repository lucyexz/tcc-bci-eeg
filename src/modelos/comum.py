"""Peças comuns aos cinco modelos: camadas do Transformer, configuração de treino e o laço de treino.

Todos os modelos recebem a mesma entrada: uma janela causal (N, T, 25) com os
T registros mais recentes de um participante, cada um com as 25 potências
espectrais (5 eletrodos × 5 bandas, ordem canal-major: AF3_Theta … AF4_Gamma).
Cada modelo só reorganiza essa janela do seu jeito (`reshape`).

Treino igual para todos: pesos de classe balanceados, Adam com taxa fixa,
EarlyStopping na perda de validação (paciência por modelo) restaurando os
pesos da melhor época.
"""

from dataclasses import dataclass

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers

N_FEATURES = 25
N_CLASSES = 3


@dataclass
class TreinoConfig:
    """Hiperparâmetros de treino comuns. Cada modelo herda e define os seus."""

    window: int = 32
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 100
    patience: int = 10
    random_state: int = 42
    use_class_weights: bool = True
    verbose: int = 2


@dataclass
class TreinoResultado:
    model: keras.Model
    history: dict
    y_pred: np.ndarray
    y_pred_proba: np.ndarray
    y_test: np.ndarray
    test_accuracy: float
    test_loss: float
    config: object


# ---------------------------------------------------------------------------
# Camadas do Transformer (os nomes das classes ficam fixos: são gravados nos .keras)
# ---------------------------------------------------------------------------

class _LearnablePositionalEmbedding(layers.Layer):
    """Soma um vetor de posição aprendido a cada token."""

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
    """Põe um token [CLS] aprendido no início da sequência."""

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
    """Devolve o token [CLS] (posição 0)."""

    def call(self, x):
        return x[:, 0, :]


CUSTOM_OBJECTS = {
    "_LearnablePositionalEmbedding": _LearnablePositionalEmbedding,
    "_PrependCLSToken": _PrependCLSToken,
    "_ExtractFirstToken": _ExtractFirstToken,
}


def transformer_encoder_block(x, d_model: int, num_heads: int, ff_dim: int, dropout_rate: float):
    """Um bloco pre-norm: autoatenção + FFN, cada um com conexão residual."""
    residual = x
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        dropout=dropout_rate,
    )(x, x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Add()([residual, x])

    residual = x
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dense(ff_dim, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(d_model)(x)
    return layers.Add()([residual, x])


def encoder_head(x, n_tokens: int, config) -> keras.KerasTensor:
    """Posicional → [CLS] → N blocos → [CLS] → LayerNorm → Dropout → Dense(3, softmax)."""
    x = _LearnablePositionalEmbedding(seq_len=n_tokens, d_model=config.d_model)(x)
    x = _PrependCLSToken(config.d_model)(x)
    for _ in range(config.num_layers):
        x = transformer_encoder_block(
            x,
            d_model=config.d_model,
            num_heads=config.num_heads,
            ff_dim=config.ff_dim,
            dropout_rate=config.dropout_rate,
        )
    x = _ExtractFirstToken()(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dropout(config.dropout_rate)(x)
    return layers.Dense(N_CLASSES, activation="softmax")(x)


# ---------------------------------------------------------------------------
# Treino
# ---------------------------------------------------------------------------

def treinar(
    build,
    reshape,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config,
    validation_data: tuple[np.ndarray, np.ndarray],
) -> TreinoResultado:
    """Treina um modelo com janelas (N, T, 25) e avalia no teste.

    Sementes → reshape → pesos de classe → modelo → Adam → EarlyStopping
    (val_loss, restaura a melhor época) → avaliação.
    """
    tf.random.set_seed(config.random_state)
    np.random.seed(config.random_state)

    X_train_m = reshape(X_train)
    X_test_m = reshape(X_test)
    X_val, y_val = validation_data

    class_weight = None
    if config.use_class_weights:
        weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
        class_weight = dict(enumerate(weights))

    model = build(config)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=config.patience,
        restore_best_weights=True,
    )
    history = model.fit(
        X_train_m,
        y_train,
        validation_data=(reshape(X_val), y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=[early_stopping],
        class_weight=class_weight,
        verbose=config.verbose,
    )

    test_loss, test_accuracy = model.evaluate(X_test_m, y_test, verbose=0)
    y_pred_proba = model.predict(X_test_m, verbose=0)

    return TreinoResultado(
        model=model,
        history=history.history,
        y_pred=y_pred_proba.argmax(axis=1),
        y_pred_proba=y_pred_proba,
        y_test=y_test,
        test_accuracy=test_accuracy,
        test_loss=test_loss,
        config=config,
    )
