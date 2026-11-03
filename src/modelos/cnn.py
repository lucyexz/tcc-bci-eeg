"""CNN: a janela vista como uma imagem 5 × 5 com T canais.

Cada registro é uma grade eletrodo × banda (5 × 5); os T registros da janela
são empilhados como canais da imagem, como os canais de cor de uma foto.

    (5 eletrodos, 5 bandas, T registros)
    → Conv2D(32, 4×4, same) → BatchNorm → ReLU
    → Conv2D(64, 4×4, same) → BatchNorm → ReLU
    → MaxPool(2×2) → Flatten → Dropout(0,4) → Dense(64, ReLU) → Dense(3, softmax)
"""

from dataclasses import dataclass

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from src.modelos.comum import N_CLASSES, TreinoConfig


@dataclass
class CNNConfig(TreinoConfig):
    conv_filters: tuple[int, ...] = (32, 64)
    dropout_rate: float = 0.4
    dense_units: int = 64
    learning_rate: float = 0.0003
    batch_size: int = 128
    patience: int = 10


def reshape(X: np.ndarray) -> np.ndarray:
    """(N, T, 25) → (N, 5, 5, T): o canal t da imagem é o registro t da janela."""
    n, window, _ = X.shape
    return np.ascontiguousarray(X.reshape(n, window, 5, 5).transpose(0, 2, 3, 1))


def build(config: CNNConfig) -> keras.Model:
    # O Keras sorteia a semente de cada camada quando ela é criada; a camada
    # densa oculta é criada primeiro de propósito, para manter a mesma
    # inicialização (e os mesmos resultados) de sempre.
    hidden = layers.Dense(config.dense_units, activation="relu")
    return keras.Sequential([
        layers.Input(shape=(5, 5, config.window)),
        layers.Conv2D(config.conv_filters[0], (4, 4), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv2D(config.conv_filters[1], (4, 4), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(config.dropout_rate),
        hidden,
        layers.Dense(N_CLASSES, activation="softmax"),
    ])
