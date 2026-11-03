"""CNN + Transformer: convoluções extraem tokens de tempo e um Transformer os combina.

Adaptação do EEG Conformer (Song et al., 2023) à potência por banda. A janela
é vista como 25 séries (eletrodo × banda) de T registros:

    (25 séries, T registros, 1)
    → Conv2D(16, (1, 4), valid)     temporal: o mesmo filtro em todas as séries
    → Conv2D(16, (25, 1), valid)    espacial: mistura todos os eletrodos × bandas
    → BatchNorm → ELU → AvgPool(1, 4, passo 2) → Dropout
    → Dense(32) por token
    → posicional aprendido + [CLS] → 1 bloco Transformer (o mesmo do Transformer)
    → [CLS] → LayerNorm → Dropout → Dense(3, softmax)

Diferenças em relação ao Conformer original, por causa dos dados: kernels em
registros (~8 por segundo; 4 registros ≈ 0,5 s) e não em amostras do sinal
bruto; rede mais estreita e rasa; encoder e cabeça iguais aos do Transformer,
para que só a extração convolucional mude entre os dois modelos.
Com T = 16/32/64 o pooling deixa 5/13/29 tokens.
"""

from dataclasses import dataclass

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from src.modelos.comum import N_FEATURES, encoder_head
from src.modelos.transformer import TransformerConfig

RECORDS_PER_SECOND = 8


@dataclass
class CNNTransformerConfig(TransformerConfig):
    conv_filters: int = 16
    temporal_kernel: int = 4
    pool_size: int = 4
    pool_stride: int = 2


def n_tokens(config: CNNTransformerConfig) -> int:
    """Tokens de tempo depois da convolução temporal (valid) e do pooling."""
    after_conv = config.window - config.temporal_kernel + 1
    return (after_conv - config.pool_size) // config.pool_stride + 1


def token_offsets_s(config: CNNTransformerConfig) -> np.ndarray:
    """Centro de cada token, em segundos antes do registro classificado (0 = o atual)."""
    span = config.temporal_kernel + config.pool_size - 1
    centers = np.arange(n_tokens(config)) * config.pool_stride + (span - 1) / 2
    return (centers - (config.window - 1)) / RECORDS_PER_SECOND


def reshape(X: np.ndarray) -> np.ndarray:
    """(N, T, 25) → (N, 25, T, 1): uma linha por série eletrodo × banda."""
    return X.transpose(0, 2, 1)[..., None]


def build(config: CNNTransformerConfig) -> keras.Model:
    tokens = n_tokens(config)
    if tokens < 1:
        raise ValueError(f"Janela T={config.window} curta demais para a convolução e o pooling")

    inputs = layers.Input(shape=(N_FEATURES, config.window, 1))
    x = layers.Conv2D(config.conv_filters, (1, config.temporal_kernel), padding="valid")(inputs)
    x = layers.Conv2D(config.conv_filters, (N_FEATURES, 1), padding="valid")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("elu")(x)
    x = layers.AveragePooling2D((1, config.pool_size), strides=(1, config.pool_stride))(x)
    x = layers.Dropout(config.dropout_rate)(x)
    x = layers.Reshape((tokens, config.conv_filters))(x)
    x = layers.Dense(config.d_model)(x)
    return keras.Model(inputs=inputs, outputs=encoder_head(x, tokens, config))
