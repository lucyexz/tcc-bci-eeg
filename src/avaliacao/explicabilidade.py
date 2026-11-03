"""Explicabilidade dos modelos treinados: Integrated Gradients e atenção do [CLS].

- Integrated Gradients (IG): quanto cada valor da janela de entrada (eletrodo ×
  banda × registro) contribuiu para a probabilidade da classe, integrando o
  gradiente do caminho entre a linha de base (zero = a média do treino depois
  do StandardScaler) e a amostra.
- Atenção do [CLS] (só nos modelos com Transformer): quanto o token [CLS], que
  vira a representação usada na classificação, olha para cada token de tempo.
"""

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.modelos.comum import CUSTOM_OBJECTS


def load_model(path: Path) -> keras.Model:
    return keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS, compile=False)


def amostra_estratificada(y: np.ndarray, por_classe: int, seed: int = 42) -> np.ndarray:
    """Índices de até `por_classe` amostras sorteadas de cada classe (em ordem crescente).

    O teste está ordenado por participante e bloco, então os primeiros
    registros não representam as classes.
    """
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([
        rng.choice(np.flatnonzero(y == c), min(por_classe, int((y == c).sum())), replace=False)
        for c in range(3)]))


def window_integrated_gradients(model, X_in: np.ndarray, target: np.ndarray, steps: int = 32,
                                batch_size: int = 128) -> np.ndarray:
    """IG de cada valor da entrada (já no formato do modelo) para a classe `target`.

    Retorna um array com a forma de `X_in`.
    """
    alphas = tf.reshape(tf.linspace(0.0, 1.0, steps + 1)[1:], (-1,) + (1,) * X_in.ndim)
    out = []
    for start in range(0, len(X_in), batch_size):
        xb = tf.constant(X_in[start:start + batch_size], dtype=tf.float32)
        tb = target[start:start + batch_size]
        path = tf.reshape(alphas * xb[None], (-1, *xb.shape[1:]))  # (steps·B, ...)
        with tf.GradientTape() as tape:
            tape.watch(path)
            proba = model(path, training=False)
            chosen = tf.gather(proba, np.tile(tb, steps), axis=1, batch_dims=1)
        grads = tf.reshape(tape.gradient(chosen, path), (steps, -1, *xb.shape[1:]))
        out.append((tf.reduce_mean(grads, axis=0) * xb).numpy())
    return np.concatenate(out)


def _cls_attention_camada(model, X_in: np.ndarray, i: int) -> np.ndarray:
    """Atenção do [CLS] para cada token de tempo na camada `i`, por amostra: (N, n_tokens).

    Média das cabeças, renormalizada entre os tokens (sem o próprio [CLS]):
    1/n_tokens = atenção uniforme.
    """
    ln_layers = [layer for layer in model.layers if isinstance(layer, keras.layers.LayerNormalization)]
    mha = [layer for layer in model.layers if isinstance(layer, keras.layers.MultiHeadAttention)][i]
    pre = keras.Model(model.input, ln_layers[2 * i].output)
    hidden = pre.predict(X_in, batch_size=1024, verbose=0)
    _, scores = mha(hidden, hidden, return_attention_scores=True, training=False)
    cls_to_tokens = scores.numpy().mean(axis=1)[:, 0, 1:]
    return cls_to_tokens / cls_to_tokens.sum(axis=1, keepdims=True)


def n_camadas_atencao(model) -> int:
    return sum(isinstance(layer, keras.layers.MultiHeadAttention) for layer in model.layers)


def cls_attention_por_amostra(model, X_in: np.ndarray) -> np.ndarray:
    """Atenção do [CLS] na última camada, para cada janela: (N, n_tokens). Usada no simulador."""
    return _cls_attention_camada(model, X_in, n_camadas_atencao(model) - 1)


def cls_attention_over_tokens(model, X_in: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Atenção do [CLS] para cada token de tempo, por camada e classe: (camadas, 3, n_tokens)."""
    result = []
    for i in range(n_camadas_atencao(model)):
        cls_to_tokens = _cls_attention_camada(model, X_in, i)
        result.append(np.stack([cls_to_tokens[y == c].mean(axis=0) for c in range(3)]))
    return np.stack(result)
