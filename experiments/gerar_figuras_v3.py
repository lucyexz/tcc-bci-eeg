"""Gera as figuras da iteração v3 (LOSO e LOTO do agrupado), PNG 300 dpi.

    python experiments/gerar_figuras_v3.py

Mesmo estilo e mesma métrica da v2 (acurácia balanceada, acaso = 33,3%).
Tudo vai para outputs/iterations/v3/figures/resultados/.
"""

import numpy as np
import pandas as pd

from _v3_results import MODELS, USERS, V3_ROOT, collect_all, load_predictions
from gerar_figuras_v2 import (
    CHANCE,
    METRIC,
    METRIC_LABEL,
    POOLED_VARIANTS,
    VARIANT_COLORS,
    VARIANT_LABELS,
    _chance_line,
    _grouped_bars,
    ps,
)

plt = ps.plt

FIG_ROOT = V3_ROOT.parent / "figures" / "resultados"

# Do mais permissivo (sujeito e trecho vistos no treino) ao mais estrito.
PROTOCOLS = ["chronological", "blocked_cv", "loto", "loso"]
PROTOCOL_LABELS = {
    "chronological": "Hold-out cronológico",
    "blocked_cv": "CV por blocos (6 folds)",
    "loto": "LOTO (12 folds)",
    "loso": "LOSO (5 sujeitos)",
}
PROTOCOL_COLORS = dict(zip(PROTOCOLS, ps.SERIES))


def subject_level(df: pd.DataFrame, protocol: str, variants=POOLED_VARIANTS) -> pd.DataFrame:
    """Média sobre repetições (sementes ou folds) por (variante, modelo, sujeito)."""
    sub = df[(df.protocol == protocol) & (df.subject != "ALL") & df.variant.isin(variants)]
    if sub.empty:
        return pd.DataFrame(columns=["variant", "model", "subject", "value"])
    return sub.groupby(["variant", "model", "subject"], as_index=False).agg(
        value=(METRIC, "mean"), trial=("trial_accuracy", "mean"))


def best_variant(df: pd.DataFrame, protocol: str) -> str | None:
    sl = subject_level(df, protocol)
    return sl.groupby("variant").value.mean().idxmax() if not sl.empty else None


def fig_ablation(df, protocol, out):
    sl = subject_level(df, protocol)
    if sl.empty:
        return
    table = sl.pivot_table(index="model", columns="variant", values="value", aggfunc="mean")
    err = sl.pivot_table(index="model", columns="variant", values="value", aggfunc="std")

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    _grouped_bars(ax, table, MODELS, POOLED_VARIANTS, VARIANT_COLORS, VARIANT_LABELS, err)
    ax.set_xticklabels([ps.MODEL_LABELS[m] for m in MODELS])
    _chance_line(ax)
    ax.set_ylim(0.2, max(0.62, float((table + err.fillna(0)).max().max()) + 0.04))
    ps.percent_axis(ax)
    ax.set_ylabel(f"{METRIC_LABEL}\n(média entre sujeitos ± desvio)")
    ax.set_title(f"Efeito das features — modelo agrupado, {PROTOCOL_LABELS[protocol]}")
    ax.legend(ncol=len(POOLED_VARIANTS), loc="upper left", bbox_to_anchor=(0, -0.1))
    ps.save(fig, out / f"res_ablacao_{protocol}.png")


def fig_subject_bars(df, protocol, out):
    sl = subject_level(df, protocol)
    if sl.empty:
        return
    fig, axes = plt.subplots(1, 5, figsize=(12, 3.3), sharey=True)
    for ax, user in zip(axes, USERS):
        table = sl[sl.subject == user].pivot_table(index="model", columns="variant", values="value")
        _grouped_bars(ax, table, MODELS, POOLED_VARIANTS, VARIANT_COLORS, VARIANT_LABELS)
        ax.set_xticklabels([ps.MODEL_LABELS[m].replace("Transformer", "Transf.") for m in MODELS],
                           rotation=30, fontsize=7.5)
        _chance_line(ax, label=ax is axes[-1])
        ax.set_title(user)
    axes[0].set_ylim(0.15, 0.85)
    ps.percent_axis(axes[0])
    axes[0].set_ylabel(METRIC_LABEL)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(POOLED_VARIANTS), loc="lower left", bbox_to_anchor=(0.06, -0.08))
    detail = "sujeito de teste fora do treino" if protocol == "loso" else "recorte por sujeito"
    fig.suptitle(f"Por sujeito — agrupado, {PROTOCOL_LABELS[protocol]} ({detail})",
                 x=0.06, ha="left", fontsize=10, fontweight="semibold")
    fig.tight_layout()
    ps.save(fig, out / f"res_por_sujeito_{protocol}.png")


def fig_protocols(df, out):
    """Mesma variante, quatro protocolos: quanto custa cada grau de generalização."""
    rows = []
    for protocol in PROTOCOLS:
        sl = subject_level(df, protocol)
        for variant, g in sl.groupby("variant"):
            per_subject = g.groupby("subject").value.mean()
            rows.append(dict(protocol=protocol, variant=variant,
                             value=per_subject.mean(), err=per_subject.std()))
    res = pd.DataFrame(rows)
    if res.empty:
        return
    table = res.pivot(index="variant", columns="protocol", values="value")
    err = res.pivot(index="variant", columns="protocol", values="err")
    protocols = [p for p in PROTOCOLS if p in table.columns]

    fig, ax = plt.subplots(figsize=(7.6, 3.7))
    _grouped_bars(ax, table, POOLED_VARIANTS, protocols, PROTOCOL_COLORS, PROTOCOL_LABELS, err)
    ax.set_xticklabels([VARIANT_LABELS[v].replace(" + norm. sujeito", "\n+ norm. sujeito")
                        for v in POOLED_VARIANTS], fontsize=8)
    _chance_line(ax)
    ax.set_ylim(0.2, max(0.62, float((table + err.fillna(0)).max().max()) + 0.04))
    ps.percent_axis(ax)
    ax.set_ylabel(f"{METRIC_LABEL}\n(média dos 4 modelos; ± desvio entre sujeitos)")
    ax.set_title("Modelo agrupado sob os quatro protocolos de avaliação")
    ax.legend(ncol=2, loc="upper left", bbox_to_anchor=(0, -0.2))
    ps.save(fig, out / "res_protocolos.png")


def fig_loso_gap(df, out):
    """Por sujeito: o mesmo modelo quando o sujeito está no treino (LOTO) × fora dele (LOSO)."""
    variant = best_variant(df, "loso")
    if variant is None:
        return
    seen = subject_level(df, "loto", [variant]).groupby("subject").value.mean()
    unseen = subject_level(df, "loso", [variant]).groupby("subject").value.mean()
    if seen.empty or unseen.empty:
        return

    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    for i, user in enumerate(USERS):
        if user not in seen.index or user not in unseen.index:
            continue
        ax.plot([unseen[user], seen[user]], [i, i], color=ps.AXIS, linewidth=2, zorder=1)
        ax.scatter([seen[user]], [i], s=50, color=PROTOCOL_COLORS["loto"], edgecolor=ps.SURFACE,
                   linewidth=1.5, zorder=2, label="sujeito no treino (LOTO)" if i == 0 else None)
        ax.scatter([unseen[user]], [i], s=50, color=PROTOCOL_COLORS["loso"], edgecolor=ps.SURFACE,
                   linewidth=1.5, zorder=2, label="sujeito fora do treino (LOSO)" if i == 0 else None)
    ax.axvline(CHANCE, color=ps.INK_MUTED, linewidth=1)
    ax.set_yticks(range(5))
    ax.set_yticklabels(USERS)
    ax.invert_yaxis()
    ps.percent_axis(ax, "x")
    ax.set_xlabel(f"{METRIC_LABEL} (média dos 4 modelos) — linha vertical: acaso")
    ax.set_title(f"O custo de ser um usuário novo — {VARIANT_LABELS[variant]}")
    ax.legend(loc="lower right")
    ps.save(fig, out / "res_loso_vs_loto.png")


def fig_confusions(df, protocol, out):
    from sklearn.metrics import confusion_matrix

    variant = best_variant(df, protocol)
    if variant is None:
        return
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.0))
    for ax, model in zip(axes, MODELS):
        parts = [load_predictions(protocol, variant, model, u) for u in USERS]
        parts = [p for p in parts if p is not None]
        if not parts:
            ax.set_visible(False)
            continue
        y_true = np.concatenate([p["y_true"] for p in parts])
        y_pred = np.concatenate([p["y_pred"] for p in parts])
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2], normalize="true")
        ax.imshow(cm, cmap=ps.SEQUENTIAL, vmin=0, vmax=1)
        ps.no_grid(ax)
        for (i, j), value in np.ndenumerate(cm):
            ax.text(j, i, f"{value:.0%}", ha="center", va="center", fontsize=8.5,
                    color=ps.SURFACE if value > 0.55 else ps.INK)
        ax.set_xticks(range(3))
        ax.set_xticklabels(ps.CLASS_LABELS, fontsize=7.5)
        ax.set_yticks(range(3))
        ax.set_yticklabels(ps.CLASS_LABELS if ax is axes[0] else [], fontsize=7.5)
        ax.set_xlabel("Previsto", fontsize=8)
        ax.set_title(ps.MODEL_LABELS[model], fontsize=9)
    axes[0].set_ylabel("Verdadeiro", fontsize=8)
    fig.suptitle(f"Matrizes de confusão (recall por linha) — agrupado, {VARIANT_LABELS[variant]}, "
                 f"{PROTOCOL_LABELS[protocol]} (5 sujeitos somados)",
                 x=0.02, ha="left", fontsize=10, fontweight="semibold")
    fig.tight_layout()
    ps.save(fig, out / f"res_matrizes_confusao_{protocol}.png")


def main() -> None:
    df = collect_all()
    if df.empty or not (df.iteration == "v3").any():
        raise SystemExit(f"Nenhum resultado da v3 em {V3_ROOT}.")

    for protocol in ["loso", "loto"]:
        fig_ablation(df, protocol, FIG_ROOT)
        fig_subject_bars(df, protocol, FIG_ROOT)
        fig_confusions(df, protocol, FIG_ROOT)
    fig_protocols(df, FIG_ROOT)
    fig_loso_gap(df, FIG_ROOT)


if __name__ == "__main__":
    main()
