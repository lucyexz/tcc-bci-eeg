"""Bi-LSTM e Bi-GRU: a janela lida registro a registro, nos dois sentidos.

Cada passo de tempo é um registro com os 25 valores espectrais.

    (T registros, 25 valores)
    → Bidirecional(LSTM ou GRU 64, devolve a sequência) → Dropout(0,4)
    → Bidirecional(LSTM ou GRU 32) → Dropout(0,4)
    → Dense(64, ReLU) → Dropout(0,4) → Dense(3, softmax)

As duas redes têm as mesmas camadas e o mesmo treino; só a célula recorrente
muda (a GRU tem menos portas e menos parâmetros que a LSTM).
"""

from dataclasses import dataclass

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from src.modelos.comum import N_CLASSES, N_FEATURES, TreinoConfig


@dataclass
class RecorrenteConfig(TreinoConfig):
    units: tuple[int, ...] = (64, 32)
    dropout_rate: float = 0.4
    recurrent_dropout: float = 0.1
    dense_units: int = 64
    learning_rate: float = 0.001
    batch_size: int = 64
    patience: int = 10


@dataclass
class BiLSTMConfig(RecorrenteConfig):
    pass


@dataclass
class BiGRUConfig(RecorrenteConfig):
    pass


def reshape(X: np.ndarray) -> np.ndarray:
    """A janela já está no formato (N, T passos, 25 valores)."""
    return X


def _build(config: RecorrenteConfig, cell) -> keras.Model:
    rnn_layers = []
    for i, n_units in enumerate(config.units):
        rnn = cell(n_units, return_sequences=i < len(config.units) - 1,
                   recurrent_dropout=config.recurrent_dropout)
        rnn_layers.append(layers.Bidirectional(rnn))
        rnn_layers.append(layers.Dropout(config.dropout_rate))
    # A ordem de criação das camadas define as sementes sorteadas pelo Keras.
    hidden = [layers.Dense(config.dense_units, activation="relu"), layers.Dropout(config.dropout_rate)]
    return keras.Sequential([
        layers.Input(shape=(config.window, N_FEATURES)),
        *rnn_layers,
        *hidden,
        layers.Dense(N_CLASSES, activation="softmax"),
    ])


def build_bilstm(config: BiLSTMConfig) -> keras.Model:
    return _build(config, layers.LSTM)


def build_bigru(config: BiGRUConfig) -> keras.Model:
    return _build(config, layers.GRU)
