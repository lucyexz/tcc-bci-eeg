"""Transformer: cada registro da janela é um token de tempo.

    (T registros, 25 valores)
    → Dense(32) por registro (embedding)
    → posicional aprendido (T) → [CLS] no início (T + 1 tokens)
    → 1 bloco pre-norm (autoatenção com 2 cabeças + FFN 64)
    → [CLS] → LayerNorm → Dropout → Dense(3, softmax)

Tamanho pequeno de propósito (d_model 32, 1 bloco): a base tem pouca
informação independente e um encoder maior só sobreajusta.
"""

from dataclasses import dataclass

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from src.modelos.comum import N_FEATURES, TreinoConfig, encoder_head


@dataclass
class TransformerConfig(TreinoConfig):
    d_model: int = 32
    num_heads: int = 2
    num_layers: int = 1
    ff_dim: int = 64
    dropout_rate: float = 0.1
    learning_rate: float = 0.0003
    batch_size: int = 128
    patience: int = 15


def reshape(X: np.ndarray) -> np.ndarray:
    """A janela já está no formato (N, T tokens, 25 valores)."""
    return X


def build(config: TransformerConfig) -> keras.Model:
    inputs = layers.Input(shape=(config.window, N_FEATURES))
    x = layers.Dense(config.d_model)(inputs)
    return keras.Model(inputs=inputs, outputs=encoder_head(x, config.window, config))
