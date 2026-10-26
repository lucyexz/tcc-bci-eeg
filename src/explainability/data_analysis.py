"""Análise do dataset independente de modelo: o que o sinal oferece.

Responde, com figuras e números, às perguntas que explicam os resultados:

1. Por que o split aleatório vazava?           → autocorrelação das features
2. Por que o teste cronológico é tão difícil?  → deriva ao longo dos blocos
3. Onde existe informação de classe?           → separabilidade por canal/banda
4. Por que o modelo agrupado falha?            → os dados se agrupam por sujeito
5. Como é o espectro de cada classe?           → espectro médio por classe

Todas as estatísticas de separabilidade são calculadas sobre **médias por
bloco de trial** (36 por sujeito), não sobre amostras: amostras vizinhas são
quase cópias, e tratá-las como independentes inflaria os testes.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.constants import BLOCK_COLUMN, EEG_BANDS, EEG_CHANNELS, LABEL_COLUMN, USER_COLUMN
from src.data.loader import load_all_users
from src.explainability import plot_style as ps
from src.features.spectral_features import asymmetry_features, transform_features
from src.preprocessing.base_preprocessing import (
    assign_trial_blocks,
    remove_duplicate_rows,
    remove_rows_without_label_or_user,
)

plt = ps.plt


def load_clean() -> pd.DataFrame:
    """Mesma limpeza do pré-processamento, mais bloco e posição."""
    df = load_all_users()
    df, _ = remove_rows_without_label_or_user(df)
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)
    df[USER_COLUMN] = df[USER_COLUMN].astype(str)
    df, _ = remove_duplicate_rows(df)
    df = df.dropna().reset_index(drop=True)
    return assign_trial_blocks(df)


def _log_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    return transform_features(df, "log")


# ---------------------------------------------------------------------------
# 1. Autocorrelação
# ---------------------------------------------------------------------------

def _acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = x - x.mean()
    n = len(x)
    size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(x, size)
    acov = np.fft.irfft(spectrum * np.conj(spectrum), size)[: max_lag + 1]
    return acov / acov[0]


def autocorrelation(df_log: pd.DataFrame, features: list[str], out_dir: Path) -> dict:
    max_lag = 1200
    lags = np.arange(1, max_lag + 1)
    block_len = int(df_log.groupby([USER_COLUMN, BLOCK_COLUMN]).size().median())

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    summary = {"median_block_length": block_len, "per_user": {}}

    for user in ps.USERS:
        values = df_log.loc[df_log[USER_COLUMN] == user, features].to_numpy()
        acf = np.mean([_acf(values[:, j], max_lag) for j in range(values.shape[1])], axis=0)[1:]
        ax.plot(lags, acf, color=ps.USER_COLORS[user], label=user)
        summary["per_user"][user] = {
            "lag_1": float(acf[0]),
            "lag_8": float(acf[7]),
            "lag_block": float(acf[block_len - 1]),
        }

    ax.axvline(block_len, color=ps.INK_MUTED, linewidth=1)
    ax.text(block_len * 1.04, 0.92, f"duração de 1 trial\n(~{block_len} amostras)",
            color=ps.INK_SECONDARY, fontsize=8, va="top")
    ax.set_xscale("log")
    ax.set_ylim(-0.1, 1.0)
    ax.set_xlabel("Distância entre amostras (lag, em amostras — escala log)")
    ax.set_ylabel("Autocorrelação média\ndas 25 features (log)")
    ax.set_title("Amostras vizinhas são quase cópias")
    ax.legend(ncol=5, loc="lower left", bbox_to_anchor=(0, -0.42))
    ps.save(fig, out_dir / "dados_01_autocorrelacao.png")
    return summary


# ---------------------------------------------------------------------------
# 2. Deriva temporal
# ---------------------------------------------------------------------------

def _eta_squared(values: np.ndarray, groups: np.ndarray) -> float:
    grand = values.mean()
    total = ((values - grand) ** 2).sum()
    between = sum(
        (groups == g).sum() * (values[groups == g].mean() - grand) ** 2
        for g in np.unique(groups)
    )
    return float(between / total) if total > 0 else 0.0


def temporal_drift(df_log: pd.DataFrame, features: list[str], out_dir: Path) -> dict:
    fig, axes = plt.subplots(5, 1, figsize=(6.8, 8.2), sharex=True)
    summary = {}

    for ax, user in zip(axes, ps.USERS):
        user_df = df_log[df_log[USER_COLUMN] == user]
        level = user_df[features].mean(axis=1)
        block_means = level.groupby(user_df[BLOCK_COLUMN]).mean()
        block_labels = user_df.groupby(BLOCK_COLUMN)[LABEL_COLUMN].first()

        ax.plot(block_means.index, block_means.values, color=ps.AXIS, linewidth=1, zorder=1)
        for label in range(3):
            idx = block_labels.index[block_labels == label]
            ax.scatter(idx, block_means.loc[idx], s=22, color=ps.CLASS_COLORS[label],
                       edgecolor=ps.SURFACE, linewidth=1.2, zorder=2,
                       label=ps.CLASS_LABELS[label])

        cut = int(len(user_df) * 0.8)
        cut_block = user_df[BLOCK_COLUMN].iloc[cut]
        ax.axvline(cut_block - 0.5, color=ps.INK_MUTED, linewidth=1)
        ax.set_ylabel(user, rotation=0, ha="right", va="center", color=ps.INK)
        ax.tick_params(axis="y", labelsize=7)

        eta_class = []
        eta_block = []
        for feature in features:
            values = user_df[feature].to_numpy()
            eta_class.append(_eta_squared(values, user_df[LABEL_COLUMN].to_numpy()))
            eta_block.append(_eta_squared(values, user_df[BLOCK_COLUMN].to_numpy()))

        summary[user] = {
            "eta2_class_median": float(np.median(eta_class)),
            "eta2_block_median": float(np.median(eta_block)),
            "eta2_block_not_class_median": float(np.median(np.array(eta_block) - np.array(eta_class))),
        }

    axes[0].set_title("Nível médio do sinal (log da potência) em cada trial")
    axes[-1].set_xlabel("Bloco de trial (ordem de gravação) — linha vertical: início do teste cronológico (20% finais)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    ps.save(fig, out_dir / "dados_02_deriva_temporal.png")
    return summary


# ---------------------------------------------------------------------------
# 3. Separabilidade por canal × banda (ANOVA sobre médias de bloco)
# ---------------------------------------------------------------------------

def _block_anova(user_df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    block_means = user_df.groupby(BLOCK_COLUMN)[columns].mean()
    block_labels = user_df.groupby(BLOCK_COLUMN)[LABEL_COLUMN].first()
    rows = []
    for column in columns:
        groups = [block_means.loc[block_labels == c, column].to_numpy() for c in range(3)]
        f_stat, p_value = stats.f_oneway(*groups)
        rows.append({"feature": column, "F": float(f_stat), "p": float(p_value)})
    return pd.DataFrame(rows).set_index("feature")


def separability(df: pd.DataFrame, df_log: pd.DataFrame, features: list[str], out_dir: Path) -> dict:
    asym = asymmetry_features(df)
    asym_df = pd.concat([asym, df[[USER_COLUMN, BLOCK_COLUMN, LABEL_COLUMN]]], axis=1)
    asym_cols = list(asym.columns)

    results = {}
    grids = {}
    asym_grids = {}
    for user in ps.USERS:
        res = _block_anova(df_log[df_log[USER_COLUMN] == user], features)
        res_asym = _block_anova(asym_df[asym_df[USER_COLUMN] == user], asym_cols)
        grids[user] = -np.log10(res["p"].to_numpy()).reshape(5, 5)
        asym_grids[user] = -np.log10(res_asym["p"].to_numpy()).reshape(2, 5)
        combined = pd.concat([res, res_asym])
        results[user] = {
            "n_significant_p05": int((res["p"] < 0.05).sum()),
            "n_significant_asym_p05": int((res_asym["p"] < 0.05).sum()),
            "top_features": [
                {"feature": f, "F": round(r.F, 2), "p": float(f"{r.p:.2e}")}
                for f, r in combined.sort_values("p").head(5).iterrows()
            ],
        }

    vmax = max(4.0, max(g.max() for g in grids.values()), max(g.max() for g in asym_grids.values()))
    fig, axes = plt.subplots(
        2, 5, figsize=(11, 4.4),
        gridspec_kw={"height_ratios": [5, 2.2], "hspace": 0.35, "wspace": 0.12},
    )
    threshold = -np.log10(0.05)

    for col, user in enumerate(ps.USERS):
        for row, (grid, ylabels) in enumerate([
            (grids[user], EEG_CHANNELS),
            (asym_grids[user], ["AF3−AF4", "T7−T8"]),
        ]):
            ax = axes[row, col]
            image = ax.imshow(grid, cmap=ps.SEQUENTIAL, vmin=0, vmax=vmax, aspect="auto")
            ps.no_grid(ax)
            for (i, j), value in np.ndenumerate(grid):
                if value >= threshold:
                    ax.text(j, i, "●", ha="center", va="center", fontsize=6,
                            color=ps.SURFACE if value > vmax * 0.55 else ps.INK)
            ax.set_xticks(range(5))
            ax.set_xticklabels(EEG_BANDS if row == 1 else [], rotation=45, fontsize=7)
            ax.set_yticks(range(len(ylabels)))
            ax.set_yticklabels(ylabels if col == 0 else [], fontsize=7)
            if row == 0:
                ax.set_title(f"{user}  ({results[user]['n_significant_p05']}/25)", fontsize=9)

    cbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("−log10(p)  ANOVA entre classes\n(● = p < 0,05)", fontsize=8)
    cbar.outline.set_visible(False)
    fig.suptitle("Onde existe informação de classe: canal × banda (topo) e assimetria hemisférica (base)",
                 x=0.06, y=1.0, ha="left", fontsize=10, fontweight="semibold")
    ps.save(fig, out_dir / "dados_03_separabilidade.png")

    results["_note"] = (
        "ANOVA de 1 via sobre as médias de bloco (12 trials por classe). Sem "
        "correção para múltiplas comparações: com 25 testes, ~1,25 'significativos' "
        "são esperados só pelo acaso."
    )
    return results


# ---------------------------------------------------------------------------
# 4. Os dados se agrupam por sujeito ou por classe?
# ---------------------------------------------------------------------------

def _embedding_figure(emb: np.ndarray, users: np.ndarray, labels: np.ndarray, title: str, path: Path) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(11, 4.8))
    for ax in axes.flat:
        ps.no_grid(ax)
        ax.set_xticks([])
        ax.set_yticks([])

    for ax, user in zip(axes[0], ps.USERS):
        mask = users == user
        ax.scatter(emb[~mask, 0], emb[~mask, 1], s=2, color=ps.GRID, linewidth=0)
        ax.scatter(emb[mask, 0], emb[mask, 1], s=3, color=ps.USER_COLORS[user], linewidth=0)
        ax.set_title(user, fontsize=9)

    for ax, label in zip(axes[1], range(3)):
        mask = labels == label
        ax.scatter(emb[~mask, 0], emb[~mask, 1], s=2, color=ps.GRID, linewidth=0)
        ax.scatter(emb[mask, 0], emb[mask, 1], s=3, color=ps.CLASS_COLORS[label], linewidth=0)
        ax.set_title(ps.CLASS_LABELS[label], fontsize=9)
    axes[1, 3].set_visible(False)
    axes[1, 4].set_visible(False)

    fig.text(0.01, 0.73, "Por sujeito", rotation=90, va="center", fontsize=9, color=ps.INK_SECONDARY)
    fig.text(0.01, 0.28, "Por classe", rotation=90, va="center", fontsize=9, color=ps.INK_SECONDARY)
    fig.suptitle(title, x=0.03, ha="left", fontsize=10, fontweight="semibold")
    fig.tight_layout(rect=(0.02, 0, 1, 0.95))
    ps.save(fig, path)


def subject_vs_class_structure(df_log: pd.DataFrame, features: list[str], out_dir: Path,
                               per_user: int = 600, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    idx = np.concatenate([
        rng.choice(np.flatnonzero(df_log[USER_COLUMN].to_numpy() == user), per_user, replace=False)
        for user in ps.USERS
    ])
    sample = df_log.iloc[idx]
    users = sample[USER_COLUMN].to_numpy()
    labels = sample[LABEL_COLUMN].to_numpy()

    X_global = StandardScaler().fit_transform(sample[features])
    X_subject = np.empty_like(X_global)
    for user in ps.USERS:
        mask = users == user
        X_subject[mask] = StandardScaler().fit_transform(sample.loc[mask, features])

    summary = {}
    for name, X, title in [
        ("global", X_global, "t-SNE das amostras (log, normalização global): os grupos são os sujeitos"),
        ("por_sujeito", X_subject, "t-SNE após normalizar cada sujeito separadamente"),
    ]:
        emb = TSNE(n_components=2, perplexity=30, random_state=seed, init="pca").fit_transform(X)
        _embedding_figure(emb, users, labels, title,
                          out_dir / f"dados_04_tsne_{name}.png")
        summary[name] = {
            "silhouette_by_subject": float(silhouette_score(X, users)),
            "silhouette_by_class": float(silhouette_score(X, labels)),
            "pca_explained_2d": float(PCA(2).fit(X).explained_variance_ratio_.sum()),
        }
    summary["_note"] = (
        "Silhueta próxima de 0 = sem estrutura; positiva = grupos separados. "
        "Calculada no espaço das 25 features padronizadas."
    )
    return summary


# ---------------------------------------------------------------------------
# 5. Espectro médio por classe
# ---------------------------------------------------------------------------

def class_spectra(df_log: pd.DataFrame, features: list[str], out_dir: Path) -> None:
    fig, axes = plt.subplots(5, 5, figsize=(10, 9), sharex=True)
    x = np.arange(5)

    for row, user in enumerate(ps.USERS):
        user_df = df_log[df_log[USER_COLUMN] == user].copy()
        user_df[features] = StandardScaler().fit_transform(user_df[features])
        block_means = user_df.groupby(BLOCK_COLUMN)[features].mean()
        block_labels = user_df.groupby(BLOCK_COLUMN)[LABEL_COLUMN].first()

        for col, channel in enumerate(EEG_CHANNELS):
            ax = axes[row, col]
            cols = [f"LOG.{channel}.{band}" for band in EEG_BANDS]
            for label in range(3):
                values = block_means.loc[block_labels == label, cols].to_numpy()
                mean = values.mean(axis=0)
                ci = 1.96 * values.std(axis=0, ddof=1) / np.sqrt(len(values))
                ax.plot(x, mean, color=ps.CLASS_COLORS[label], marker="o", markersize=3,
                        label=ps.CLASS_LABELS[label])
                ax.fill_between(x, mean - ci, mean + ci, color=ps.CLASS_COLORS[label],
                                alpha=0.10, linewidth=0)
            ax.axhline(0, color=ps.AXIS, linewidth=0.8)
            ax.tick_params(labelsize=7)
            if row == 0:
                ax.set_title(channel, fontsize=9)
            if col == 0:
                ax.set_ylabel(user, rotation=0, ha="right", va="center", color=ps.INK)
            if row == 4:
                ax.set_xticks(x)
                ax.set_xticklabels(EEG_BANDS, rotation=45, fontsize=7)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Espectro médio por classe (log da potência, z-score por sujeito; faixa = IC 95% entre trials)",
                 x=0.02, ha="left", fontsize=10, fontweight="semibold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    ps.save(fig, out_dir / "dados_05_espectro_por_classe.png")


def run_all(out_dir: Path, summary_path: Path) -> dict:
    df = load_clean()
    df_log, features = _log_frame(df)

    summary = {
        "autocorrelation": autocorrelation(df_log, features, out_dir),
        "temporal_drift": temporal_drift(df_log, features, out_dir),
        "separability": separability(df, df_log, features, out_dir),
        "subject_vs_class": subject_vs_class_structure(df_log, features, out_dir),
    }
    class_spectra(df_log, features, out_dir)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [ok] resumo salvo em {summary_path}")
    return summary
