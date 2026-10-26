"""Explicabilidade dos modelos treinados: o que cada rede usa para decidir.

Técnicas (todas sem dependências novas — só TF, sklearn e numpy):

- Importância por permutação (model-agnostic): quanto a acurácia balanceada
  cai quando embaralhamos as features de um par canal × banda. Calculada no
  treino (o que o modelo aprendeu) e no teste (o que generaliza).
- Integrated Gradients: atribuição de cada feature à probabilidade de cada
  classe, integrando o gradiente do baseline (média do treino = 0 após o
  StandardScaler) até a amostra.
- Atenção do Transformer: quanto o token [CLS] — que decide a classe — olha
  para cada canal, por camada e por classe.
- Embeddings: t-SNE da penúltima camada, por classe e por sujeito.
- Timeline: probabilidade prevista ao longo do teste vs. rótulo verdadeiro.
- Calibração: confiança prevista vs. acurácia observada.

Toda importância é agregada na grade 5 canais × 5 bandas; no feature set
`log_rel` (2 mapas), os dois mapas do mesmo canal × banda são somados ou
permutados juntos.
"""

from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
from sklearn.metrics import balanced_accuracy_score
from tensorflow import keras

from src.constants import EEG_BANDS, EEG_CHANNELS
from src.explainability import plot_style as ps
from src.models.cnn import create_temporal_windows, reshape_for_cnn
from src.models.lstm import reshape_for_lstm
from src.models.transformer import (
    _ExtractFirstToken,
    _LearnablePositionalEmbedding,
    _PrependCLSToken,
    reshape_for_transformer,
)

plt = ps.plt

CUSTOM_OBJECTS = {
    "_LearnablePositionalEmbedding": _LearnablePositionalEmbedding,
    "_PrependCLSToken": _PrependCLSToken,
    "_ExtractFirstToken": _ExtractFirstToken,
}


def load_model(path: Path) -> keras.Model:
    return keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS, compile=False)


def to_model_input(model_name: str, X: np.ndarray) -> np.ndarray:
    if model_name == "cnn":
        return reshape_for_cnn(create_temporal_windows(X, 1), 1)
    if model_name in ("lstm", "gru"):
        return reshape_for_lstm(X)
    return reshape_for_transformer(X)


def _tf_reshape(model_name: str, X: tf.Tensor, n_maps: int) -> tf.Tensor:
    """Versão TensorFlow dos reshape_for_*, para derivar em relação às features planas."""
    grid = tf.reshape(X, (-1, n_maps, 5, 5))  # (N, mapa, canal, banda)
    if model_name == "cnn":
        return tf.reshape(tf.transpose(grid, (0, 2, 3, 1)), (-1, 5, 5, n_maps))
    if model_name in ("lstm", "gru"):
        return tf.reshape(tf.transpose(grid, (0, 3, 1, 2)), (-1, 5, 5 * n_maps))
    return tf.reshape(tf.transpose(grid, (0, 2, 1, 3)), (-1, 5, 5 * n_maps))


def predict_proba(model, model_name: str, X: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    return model.predict(to_model_input(model_name, X), batch_size=batch_size, verbose=0)


def _to_grid(values_flat: np.ndarray) -> np.ndarray:
    """(…, 25·k) → (…, 5, 5) somando os mapas."""
    n_maps = values_flat.shape[-1] // 25
    return values_flat.reshape(*values_flat.shape[:-1], n_maps, 5, 5).sum(axis=-3)


# ---------------------------------------------------------------------------
# Importância por permutação
# ---------------------------------------------------------------------------

def permutation_importance(
    model, model_name: str, X: np.ndarray, y: np.ndarray,
    n_repeats: int = 3, seed: int = 42, max_samples: int = 12000,
) -> dict:
    """Queda de acurácia balanceada ao embaralhar cada par canal × banda.

    Retorna a grade 5×5 (média das repetições) e a acurácia de referência.
    Também agrega por canal (5 bandas juntas) e por banda (5 canais juntos).
    """
    rng = np.random.default_rng(seed)
    if len(X) > max_samples:
        idx = np.sort(rng.choice(len(X), max_samples, replace=False))
        X, y = X[idx], y[idx]

    n_maps = X.shape[1] // 25
    base = balanced_accuracy_score(y, predict_proba(model, model_name, X).argmax(1))

    def drop_for(columns: list[int]) -> float:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            perm = rng.permutation(len(X))
            Xp[:, columns] = X[perm][:, columns]
            score = balanced_accuracy_score(y, predict_proba(model, model_name, Xp).argmax(1))
            drops.append(base - score)
        return float(np.mean(drops))

    def cols(channels, bands):
        return [m * 25 + c * 5 + b for m in range(n_maps) for c in channels for b in bands]

    grid = np.zeros((5, 5))
    for c in range(5):
        for b in range(5):
            grid[c, b] = drop_for(cols([c], [b]))

    by_channel = {EEG_CHANNELS[c]: drop_for(cols([c], range(5))) for c in range(5)}
    by_band = {EEG_BANDS[b]: drop_for(cols(range(5), [b])) for b in range(5)}

    return {"baseline_balanced_accuracy": float(base), "grid": grid,
            "by_channel": by_channel, "by_band": by_band}


# ---------------------------------------------------------------------------
# Integrated Gradients
# ---------------------------------------------------------------------------

def integrated_gradients(
    model, model_name: str, X: np.ndarray, target: np.ndarray, steps: int = 64,
    batch_size: int = 256,
) -> np.ndarray:
    """Atribuição IG de cada feature plana à probabilidade da classe `target`.

    Baseline = vetor nulo, que após o StandardScaler é a média do treino.
    Retorna (N, 25·k).
    """
    n_maps = X.shape[1] // 25
    alphas = tf.reshape(tf.linspace(0.0, 1.0, steps + 1)[1:], (-1, 1, 1))
    out = []

    for start in range(0, len(X), batch_size):
        xb = tf.constant(X[start:start + batch_size], dtype=tf.float32)
        tb = target[start:start + batch_size]
        path = alphas * xb[None]  # (steps, B, F)
        flat = tf.reshape(path, (-1, xb.shape[1]))
        with tf.GradientTape() as tape:
            tape.watch(flat)
            proba = model(_tf_reshape(model_name, flat, n_maps), training=False)
            chosen = tf.gather(proba, np.tile(tb, steps), axis=1, batch_dims=1)
        grads = tape.gradient(chosen, flat)
        grads = tf.reshape(grads, (steps, -1, xb.shape[1]))
        out.append((tf.reduce_mean(grads, axis=0) * xb).numpy())

    return np.concatenate(out)


def class_attribution_grids(
    model, model_name: str, X: np.ndarray, y: np.ndarray,
    per_class: int = 600, seed: int = 42,
) -> np.ndarray:
    """Atribuição IG média por classe, na grade 5×5. Retorna (3, 5, 5).

    Para cada classe, usa amostras cujo rótulo verdadeiro é aquela classe e
    mede a evidência que cada canal × banda dá a favor dela.
    """
    rng = np.random.default_rng(seed)
    grids = []
    for label in range(3):
        idx = np.flatnonzero(y == label)
        if len(idx) > per_class:
            idx = rng.choice(idx, per_class, replace=False)
        attributions = integrated_gradients(model, model_name, X[idx], np.full(len(idx), label))
        grids.append(_to_grid(attributions.mean(axis=0)))
    return np.stack(grids)


# ---------------------------------------------------------------------------
# Atenção do Transformer
# ---------------------------------------------------------------------------

def transformer_cls_attention(model, X: np.ndarray, y: np.ndarray, max_samples: int = 3000) -> np.ndarray:
    """Atenção do [CLS] para cada canal, por camada e classe. Retorna (camadas, 3, 5)."""
    ln_layers = [layer for layer in model.layers if isinstance(layer, keras.layers.LayerNormalization)]
    mha_layers = [layer for layer in model.layers if isinstance(layer, keras.layers.MultiHeadAttention)]

    X_in = reshape_for_transformer(X[:max_samples])
    y = y[:max_samples]
    result = []

    for i, mha in enumerate(mha_layers):
        # Em cada bloco pré-norm, a 1ª LayerNorm alimenta a atenção.
        pre = keras.Model(model.input, ln_layers[2 * i].output)
        hidden = pre.predict(X_in, batch_size=1024, verbose=0)
        _, scores = mha(hidden, hidden, return_attention_scores=True, training=False)
        cls_to_channels = scores.numpy().mean(axis=1)[:, 0, 1:]  # (N, 5) média das cabeças
        # Renormaliza entre os 5 canais (descarta a atenção do CLS para si
        # mesmo): assim 0,2 significa atenção uniforme.
        cls_to_channels = cls_to_channels / cls_to_channels.sum(axis=1, keepdims=True)
        result.append(np.stack([cls_to_channels[y == c].mean(axis=0) for c in range(3)]))

    return np.stack(result)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def penultimate_embeddings(model, model_name: str, X: np.ndarray) -> np.ndarray:
    if isinstance(model, keras.Sequential):
        # Sequential carregado do disco não tem grafo de entrada definido:
        # remonta a passagem por todas as camadas menos a de saída.
        inputs = keras.Input(shape=model.input_shape[1:])
        x = inputs
        for layer in model.layers[:-1]:
            x = layer(x)
        feature_model = keras.Model(inputs, x)
    else:
        feature_model = keras.Model(model.inputs, model.layers[-1].input)
    return feature_model.predict(to_model_input(model_name, X), batch_size=2048, verbose=0)


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

def _grid_axes(ax, grid, vmin, vmax, cmap, show_y=True, show_x=True):
    image = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ps.no_grid(ax)
    ax.set_xticks(range(5))
    ax.set_xticklabels(EEG_BANDS if show_x else [], rotation=45, fontsize=7)
    ax.set_yticks(range(5))
    ax.set_yticklabels(EEG_CHANNELS if show_y else [], fontsize=7)
    return image


def plot_importance_grids(panels: list[tuple[str, np.ndarray]], title: str, path: Path,
                          ncols: int = 4, label: str = "Queda na acurácia balanceada (pp)") -> None:
    """Pequenos múltiplos de grades 5×5 com escala comum (pontos percentuais)."""
    n = len(panels)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols + 1, 2.5 * nrows + 0.6), squeeze=False)
    vmax = max(0.5, max(float(np.max(g)) for _, g in panels) * 100)
    image = None
    for k, ax in enumerate(axes.flat):
        if k >= n:
            ax.set_visible(False)
            continue
        name, grid = panels[k]
        image = _grid_axes(ax, np.clip(grid * 100, 0, None), 0, vmax, ps.SEQUENTIAL,
                           show_y=k % ncols == 0, show_x=k >= n - ncols)
        ax.set_title(name, fontsize=8.5)
    cbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label(label, fontsize=8)
    cbar.outline.set_visible(False)
    fig.suptitle(title, x=0.02, y=1.04, ha="left", fontsize=10, fontweight="semibold")
    ps.save(fig, path)


def plot_class_attributions(rows: list[tuple[str, np.ndarray]], title: str, path: Path) -> None:
    """Linhas = modelos/sujeitos; colunas = classes. Escala divergente comum."""
    fig, axes = plt.subplots(len(rows), 3, figsize=(7.4, 2.3 * len(rows) + 0.6), squeeze=False)
    vmax = max(float(np.abs(g).max()) for _, g in rows) or 1e-6
    image = None
    for r, (name, grids) in enumerate(rows):
        for c in range(3):
            ax = axes[r, c]
            image = _grid_axes(ax, grids[c], -vmax, vmax, ps.DIVERGING,
                               show_y=c == 0, show_x=r == len(rows) - 1)
            if r == 0:
                ax.set_title(ps.CLASS_LABELS[c], fontsize=9)
            if c == 0:
                ax.set_ylabel(name, fontsize=8.5, color=ps.INK, rotation=0, ha="right", va="center")
    cbar = fig.colorbar(image, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label("Integrated Gradients médio\n(azul = evidência a favor da classe;\nvermelho = contra)", fontsize=8)
    cbar.outline.set_visible(False)
    fig.suptitle(title, x=0.02, y=1.0 + 0.12 / len(rows), ha="left", fontsize=10, fontweight="semibold")
    ps.save(fig, path)


def plot_attention(rows: list[tuple[str, np.ndarray]], path: Path) -> None:
    """Atenção do [CLS] por canal: linhas = sujeitos/modelos, colunas = camadas."""
    n_layers = rows[0][1].shape[0]
    fig, axes = plt.subplots(len(rows), n_layers, figsize=(3.4 * n_layers + 1, 1.5 * len(rows) + 0.8),
                             squeeze=False)
    vmax = max(float(a.max()) for _, a in rows)
    image = None
    for r, (name, attention) in enumerate(rows):
        for layer in range(n_layers):
            ax = axes[r, layer]
            image = ax.imshow(attention[layer], cmap=ps.SEQUENTIAL, vmin=0, vmax=vmax, aspect="auto")
            ps.no_grid(ax)
            ax.set_xticks(range(5))
            ax.set_xticklabels(EEG_CHANNELS if r == len(rows) - 1 else [], fontsize=7)
            ax.set_yticks(range(3))
            ax.set_yticklabels(ps.CLASS_LABELS if layer == 0 else [], fontsize=7)
            if r == 0:
                ax.set_title(f"Camada {layer + 1}", fontsize=9)
            if layer == 0:
                ax.set_ylabel(name, fontsize=8.5, color=ps.INK, rotation=0, ha="right", va="center",
                              labelpad=40)
    cbar = fig.colorbar(image, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label("Peso de atenção [CLS] → canal\n(média das cabeças; 0,2 = uniforme)", fontsize=8)
    cbar.outline.set_visible(False)
    fig.suptitle("Transformer: para quais eletrodos o token de decisão olha",
                 x=0.02, y=0.93, ha="left", fontsize=10, fontweight="semibold")
    ps.save(fig, path)


def plot_embeddings(panels: list[tuple[str, np.ndarray]], users: np.ndarray, labels: np.ndarray,
                    title: str, path: Path, seed: int = 42) -> dict:
    """t-SNE de cada embedding; linha 1 colorida por classe, linha 2 por sujeito."""
    from sklearn.metrics import silhouette_score

    fig, axes = plt.subplots(2, len(panels), figsize=(3.6 * len(panels), 6.4), squeeze=False)
    summary = {}
    for k, (name, emb) in enumerate(panels):
        coords = TSNE(2, perplexity=30, random_state=seed, init="pca").fit_transform(emb)
        summary[name] = {
            "silhouette_by_class": float(silhouette_score(emb, labels)),
            "silhouette_by_subject": float(silhouette_score(emb, users)),
        }
        for row, (groups, colors, names) in enumerate([
            (labels, ps.CLASS_COLORS, dict(enumerate(ps.CLASS_LABELS))),
            (users, ps.USER_COLORS, {u: u for u in ps.USERS}),
        ]):
            ax = axes[row, k]
            ps.no_grid(ax)
            ax.set_xticks([])
            ax.set_yticks([])
            for g in names:
                mask = groups == g
                if mask.any():
                    ax.scatter(coords[mask, 0], coords[mask, 1], s=3, color=colors[g],
                               linewidth=0, label=names[g], alpha=0.8)
            if row == 0:
                ax.set_title(name, fontsize=9)
        axes[0, k].text(0.0, -0.04, f"silhueta classe {summary[name]['silhouette_by_class']:+.2f}",
                        transform=axes[0, k].transAxes, fontsize=7.5, color=ps.INK_SECONDARY, va="top")
        axes[1, k].text(0.0, -0.04, f"silhueta sujeito {summary[name]['silhouette_by_subject']:+.2f}",
                        transform=axes[1, k].transAxes, fontsize=7.5, color=ps.INK_SECONDARY, va="top")
    axes[0, -1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), markerscale=4)
    axes[1, -1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), markerscale=4)
    fig.suptitle(title, x=0.02, ha="left", fontsize=10, fontweight="semibold")
    fig.tight_layout()
    ps.save(fig, path)
    return summary


def plot_timelines(rows: list[tuple[str, dict]], path: Path, smooth: int = 16) -> None:
    """Probabilidades previstas ao longo do teste, com o rótulo verdadeiro no topo.

    `rows`: (título, npz carregado com y_true, y_pred_proba, positions, blocks).
    As probabilidades são suavizadas por média móvel causal de `smooth`
    amostras só para legibilidade.
    """
    fig, axes = plt.subplots(len(rows), 1, figsize=(8.4, 1.9 * len(rows) + 0.6), squeeze=False)
    for ax, (name, data) in zip(axes[:, 0], rows):
        order = np.argsort(data["positions"])
        t = np.arange(len(order))
        proba = data["y_pred_proba"][order]
        truth = data["y_true"][order]
        for c in range(3):
            # média móvel causal; no início usa só as amostras já disponíveis
            cumsum = np.cumsum(np.insert(proba[:, c], 0, 0.0))
            width = np.minimum(np.arange(1, len(t) + 1), smooth)
            smoothed = (cumsum[1:] - cumsum[np.arange(1, len(t) + 1) - width]) / width
            ax.plot(t, smoothed, color=ps.CLASS_COLORS[c], linewidth=1.2, label=ps.CLASS_LABELS[c])
        # faixa do rótulo verdadeiro no topo
        for c in range(3):
            mask = truth == c
            ax.fill_between(t, 1.04, 1.14, where=mask, color=ps.CLASS_COLORS[c], linewidth=0, step="mid")
        ax.set_ylim(0, 1.16)
        ax.set_yticks([0, 0.5, 1])
        ax.set_xlim(0, len(t))
        ax.set_ylabel("P(classe)")
        acc = float((data["y_pred"] == data["y_true"]).mean())
        ax.set_title(f"{name}  —  acurácia {acc:.1%}", fontsize=9)
    axes[-1, 0].set_xlabel("Amostras do conjunto de teste, em ordem de gravação "
                           "(faixa superior = rótulo verdadeiro)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"O que o modelo prevê ao longo do tempo (média móvel de {smooth} amostras)",
                 x=0.02, ha="left", fontsize=10, fontweight="semibold")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    ps.save(fig, path)


def calibration_curve(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> dict:
    confidence = proba.max(axis=1)
    correct = proba.argmax(axis=1) == y_true
    edges = np.linspace(1 / 3, 1, n_bins + 1)
    centers, accuracy, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence >= lo) & (confidence < hi if hi < 1 else confidence <= hi)
        if mask.sum() >= 20:
            centers.append(float(confidence[mask].mean()))
            accuracy.append(float(correct[mask].mean()))
            counts.append(int(mask.sum()))
    ece = sum(c * abs(a - m) for m, a, c in zip(centers, accuracy, counts)) / max(1, sum(counts))
    return {"confidence": centers, "accuracy": accuracy, "counts": counts,
            "ece": float(ece), "mean_confidence": float(confidence.mean()),
            "accuracy_overall": float(correct.mean())}


def plot_calibration(curves: dict[str, dict], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.plot([1 / 3, 1], [1 / 3, 1], color=ps.AXIS, linewidth=1)
    ax.text(0.97, 0.93, "calibração perfeita", rotation=45, ha="right", va="top",
            fontsize=7.5, color=ps.INK_MUTED, transform=ax.transData)
    for model_name, curve in curves.items():
        ax.plot(curve["confidence"], curve["accuracy"], color=ps.MODEL_COLORS[model_name],
                marker="o", markersize=4, markeredgecolor=ps.SURFACE,
                label=f"{ps.MODEL_LABELS[model_name]} (ECE {curve['ece']:.2f})")
    ax.set_xlim(0.3, 1.0)
    ax.set_ylim(0.0, 1.0)
    ps.percent_axis(ax, "x")
    ps.percent_axis(ax, "y")
    ax.set_xlabel("Confiança prevista (probabilidade máxima)")
    ax.set_ylabel("Acurácia observada")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ps.save(fig, path)
