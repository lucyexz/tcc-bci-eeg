"""Gera todas as figuras da iteração v2 (PNG 300 dpi, para a monografia).

    python experiments/gerar_figuras_v2.py                 # tudo
    python experiments/gerar_figuras_v2.py --only resultados
    python experiments/gerar_figuras_v2.py --only dados explicabilidade

Grupos:
    dados            — análise do dataset, independente de modelo
    resultados       — v1 × v2, ablação, CV, trial vs amostra, matrizes, épocas
    explicabilidade  — permutação, Integrated Gradients, atenção, embeddings,
                       timelines e calibração

Métrica principal: acurácia balanceada (média do recall das 3 classes). O
acaso é 33,3% para qualquer distribuição de classes no teste — ao contrário da
acurácia simples, cuja linha de base (classe majoritária) muda por sujeito.

Tudo vai para outputs/iterations/v2/figures/ e os números que as figuras de
explicabilidade calculam vão para outputs/iterations/v2/analysis/.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _v2_results import (
    MODELS,
    USERS,
    V2_ROOT,
    collect_all,
    load_cv_predictions,
    load_predictions,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.explainability import plot_style as ps  # noqa: E402

plt = ps.plt

ITER_ROOT = V2_ROOT.parent
FIG_ROOT = ITER_ROOT / "figures"
ANALYSIS_ROOT = ITER_ROOT / "analysis"

CHANCE = 1 / 3
METRIC = "balanced_accuracy"
METRIC_LABEL = "Acurácia balanceada"

POOLED_VARIANTS = ["raw", "log", "log_subjnorm", "log_rel_subjnorm"]
PERSONAL_VARIANTS = ["raw", "log", "log_rel"]
VARIANT_LABELS = {
    "raw": "raw (v1)",
    "log": "log",
    "log_subjnorm": "log + norm. sujeito",
    "log_rel": "log + relativa",
    "log_rel_subjnorm": "log + relativa + norm. sujeito",
}
# Variantes são passos cumulativos: rampa ordinal (cinza = referência v1).
VARIANT_COLORS = {
    "raw": ps.V1_COLOR,
    "log": "#86b6ef",
    "log_subjnorm": "#3987e5",
    "log_rel": "#3987e5",
    "log_rel_subjnorm": "#184f95",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chance_line(ax, label=True):
    ax.axhline(CHANCE, color=ps.INK_MUTED, linewidth=1)
    if label:
        ax.text(1.0, CHANCE, " acaso", transform=ax.get_yaxis_transform(),
                va="center", ha="left", fontsize=7.5, color=ps.INK_MUTED)


def _grouped_bars(ax, table: pd.DataFrame, groups: list[str], series: list[str],
                  colors: dict, labels: dict, err: pd.DataFrame | None = None):
    """Barras agrupadas finas: `table` indexado por grupo, colunas = séries."""
    n = len(series)
    width = min(0.8 / n, 0.22)
    x = np.arange(len(groups))
    for k, s in enumerate(series):
        if s not in table.columns:
            continue
        offset = (k - (n - 1) / 2) * width
        values = table.reindex(groups)[s].to_numpy()
        yerr = err.reindex(groups)[s].to_numpy() if err is not None and s in err.columns else None
        ax.bar(x + offset, values, width * 0.88, color=colors[s], label=labels[s], zorder=2,
               yerr=yerr, error_kw=dict(ecolor=ps.INK_SECONDARY, elinewidth=0.8, capsize=0))
    ax.set_xticks(x)
    return x


def best_variant(df: pd.DataFrame, track: str, protocol: str) -> str:
    variants = POOLED_VARIANTS if track == "pooled" else PERSONAL_VARIANTS
    sub = df[(df.iteration == "v2") & (df.protocol == protocol) & (df.track == track)
             & (df.subject != "ALL") & (df.variant != "raw") & df.variant.isin(variants)]
    if sub.empty:
        return "log"
    return sub.groupby("variant")[METRIC].mean().idxmax()


def best_model(df: pd.DataFrame, track: str, protocol: str, variant: str) -> str:
    sub = df[(df.iteration == "v2") & (df.protocol == protocol) & (df.track == track)
             & (df.variant == variant) & (df.subject != "ALL")]
    if sub.empty:
        return "gru"
    return sub.groupby("model")[METRIC].mean().idxmax()


def subject_level(df, protocol, track, variants):
    """Média sobre repetições (sementes ou folds) por (variante, modelo, sujeito)."""
    sub = df[(df.iteration == "v2") & (df.protocol == protocol) & (df.track == track)
             & (df.subject != "ALL") & df.variant.isin(variants)]
    return sub.groupby(["variant", "model", "subject"], as_index=False).agg(
        value=(METRIC, "mean"), accuracy=("accuracy", "mean"),
        trial=("trial_accuracy", "mean"), majority=("majority_baseline", "mean"),
        n=(METRIC, "size"))


# ---------------------------------------------------------------------------
# Resultados
# ---------------------------------------------------------------------------

def fig_ablation(df, protocol, track, out):
    variants = POOLED_VARIANTS if track == "pooled" else PERSONAL_VARIANTS
    sl = subject_level(df, protocol, track, variants)
    if sl.empty:
        return
    # média entre sujeitos; barra de erro = desvio entre sujeitos
    table = sl.pivot_table(index="model", columns="variant", values="value", aggfunc="mean")
    err = sl.pivot_table(index="model", columns="variant", values="value", aggfunc="std")

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    _grouped_bars(ax, table, MODELS, variants, VARIANT_COLORS, VARIANT_LABELS, err)
    ax.set_xticklabels([ps.MODEL_LABELS[m] for m in MODELS])
    _chance_line(ax)
    ax.set_ylim(0.2, max(0.62, float((table + err.fillna(0)).max().max()) + 0.04))
    ps.percent_axis(ax)
    ax.set_ylabel(f"{METRIC_LABEL}\n(média entre sujeitos ± desvio)")
    proto = "hold-out cronológico, 3 sementes" if protocol == "chronological" else "CV por blocos, 6 folds"
    trk = "modelo agrupado" if track == "pooled" else "modelos personalizados"
    ax.set_title(f"Efeito das features — {trk} ({proto})")
    ax.legend(ncol=len(variants), loc="upper left", bbox_to_anchor=(0, -0.1))
    ps.save(fig, out / f"res_ablacao_{track}_{protocol}.png")


def fig_subject_bars(df, protocol, track, out):
    variants = POOLED_VARIANTS if track == "pooled" else PERSONAL_VARIANTS
    sl = subject_level(df, protocol, track, variants)
    if sl.empty:
        return
    fig, axes = plt.subplots(1, 5, figsize=(12, 3.3), sharey=True)
    for ax, user in zip(axes, USERS):
        table = sl[sl.subject == user].pivot_table(index="model", columns="variant", values="value")
        _grouped_bars(ax, table, MODELS, variants, VARIANT_COLORS, VARIANT_LABELS)
        ax.set_xticklabels([ps.MODEL_LABELS[m].replace("Transformer", "Transf.") for m in MODELS],
                           rotation=30, fontsize=7.5)
        _chance_line(ax, label=ax is axes[-1])
        ax.set_title(user)
    axes[0].set_ylim(0.15, 0.85)
    ps.percent_axis(axes[0])
    axes[0].set_ylabel(METRIC_LABEL)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(variants), loc="lower left", bbox_to_anchor=(0.06, -0.08))
    proto = "hold-out cronológico" if protocol == "chronological" else "CV por blocos"
    trk = "agrupado (recorte por sujeito)" if track == "pooled" else "personalizado"
    fig.suptitle(f"Por sujeito — {trk}, {proto}", x=0.06, ha="left", fontsize=10, fontweight="semibold")
    fig.tight_layout()
    ps.save(fig, out / f"res_por_sujeito_{track}_{protocol}.png")


def fig_v1_vs_v2(df, out):
    """v1 (números congelados) × v2 (melhor variante), mesmo protocolo cronológico."""
    rows = []
    for track in ["pooled", "personalized"]:
        variant = best_variant(df, track, "chronological")
        for metric in ["accuracy", METRIC]:
            v1 = df[(df.iteration == "v1") & (df.track == track) & (df.subject != "ALL")]
            v2 = df[(df.iteration == "v2") & (df.protocol == "chronological") & (df.track == track)
                    & (df.variant == variant) & (df.subject != "ALL")]
            if v1.empty or v2.empty or metric not in v1 or v1[metric].isna().all():
                continue
            for model in MODELS:
                rows.append(dict(track=track, metric=metric, model=model, variant=variant,
                                 v1=v1[v1.model == model][metric].mean(),
                                 v2=v2[v2.model == model].groupby("subject")[metric].mean().mean()))
    if not rows:
        return
    res = pd.DataFrame(rows)
    panels = [(t, m) for t in ["pooled", "personalized"] for m in ["accuracy", METRIC]
              if not res[(res.track == t) & (res.metric == m)].empty]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (track, metric) in zip(axes, panels):
        sub = res[(res.track == track) & (res.metric == metric)].set_index("model")
        table = sub[["v1", "v2"]]
        _grouped_bars(ax, table, MODELS, ["v1", "v2"], {"v1": ps.V1_COLOR, "v2": ps.V2_COLOR},
                      {"v1": "v1", "v2": f"v2 ({VARIANT_LABELS[sub.variant.iloc[0]]})"})
        ax.set_xticklabels([ps.MODEL_LABELS[m].replace("Transformer", "Transf.") for m in MODELS],
                           rotation=30, fontsize=7.5)
        if metric == METRIC:
            _chance_line(ax, label=False)
        trk = "Agrupado" if track == "pooled" else "Personalizado"
        ax.set_title(f"{trk} — {'acurácia' if metric == 'accuracy' else 'acur. balanceada'}", fontsize=9)
        ax.legend(loc="upper left", fontsize=7)
    axes[0].set_ylim(0.2, 0.62)
    ps.percent_axis(axes[0])
    axes[0].set_ylabel("Média entre sujeitos")
    fig.suptitle("v1 × v2 no mesmo hold-out cronológico", x=0.02, ha="left",
                 fontsize=10, fontweight="semibold")
    fig.tight_layout()
    ps.save(fig, out / "res_v1_vs_v2.png")


def fig_delta_heatmap(df, protocol, out):
    """Personalizado: sujeito × modelo, raw | melhor variante | Δ."""
    variant = best_variant(df, "personalized", protocol)
    sl = subject_level(df, protocol, "personalized", ["raw", variant])
    if sl.empty or variant not in set(sl.variant):
        return
    raw = sl[sl.variant == "raw"].pivot_table(index="subject", columns="model", values="value").reindex(USERS)[MODELS]
    new = sl[sl.variant == variant].pivot_table(index="subject", columns="model", values="value").reindex(USERS)[MODELS]
    delta = new - raw

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    for ax, (title, data, cmap, vmin, vmax, fmt) in zip(axes, [
        ("raw (v1)", raw, ps.SEQUENTIAL, 0.25, 0.75, "{:.0%}"),
        (VARIANT_LABELS[variant], new, ps.SEQUENTIAL, 0.25, 0.75, "{:.0%}"),
        ("Δ (pp)", delta * 100, ps.DIVERGING, -15, 15, "{:+.1f}"),
    ]):
        image = ax.imshow(data.to_numpy(), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ps.no_grid(ax)
        for (i, j), value in np.ndenumerate(data.to_numpy()):
            if np.isnan(value):
                continue
            level = (value - vmin) / (vmax - vmin)
            light = level > 0.62 if cmap is ps.SEQUENTIAL else abs(level - 0.5) > 0.36
            ax.text(j, i, fmt.format(value), ha="center", va="center", fontsize=8,
                    color=ps.SURFACE if light else ps.INK)
        ax.set_xticks(range(4))
        ax.set_xticklabels([ps.MODEL_LABELS[m] for m in MODELS], rotation=30, fontsize=8)
        ax.set_yticks(range(5))
        ax.set_yticklabels(USERS if ax is axes[0] else [], fontsize=8)
        ax.set_title(title, fontsize=9)
    proto = "hold-out cronológico" if protocol == "chronological" else "CV por blocos"
    fig.suptitle(f"Personalizado — {METRIC_LABEL.lower()} por sujeito e modelo ({proto})",
                 x=0.02, ha="left", fontsize=10, fontweight="semibold")
    fig.tight_layout()
    ps.save(fig, out / f"res_heatmap_delta_{protocol}.png")


def fig_cv_folds(df, out):
    """Distribuição dos 6 folds por sujeito: raw × melhor variante (melhor modelo)."""
    variant = best_variant(df, "personalized", "blocked_cv")
    model = best_model(df, "personalized", "blocked_cv", variant)
    sub = df[(df.iteration == "v2") & (df.protocol == "blocked_cv") & (df.track == "personalized")
             & (df.model == model) & df.variant.isin(["raw", variant])]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    rng = np.random.default_rng(0)
    for k, v in enumerate(["raw", variant]):
        for i, user in enumerate(USERS):
            values = sub[(sub.variant == v) & (sub.subject == user)][METRIC].to_numpy()
            if len(values) == 0:
                continue
            x = i + (k - 0.5) * 0.3
            ax.scatter(x + rng.uniform(-0.05, 0.05, len(values)), values, s=16,
                       color=VARIANT_COLORS[v], edgecolor=ps.SURFACE, linewidth=0.8, zorder=3,
                       label=VARIANT_LABELS[v] if i == 0 else None)
            ax.plot([x - 0.1, x + 0.1], [values.mean()] * 2, color=ps.INK, linewidth=1.6, zorder=4)
    _chance_line(ax)
    ax.set_xticks(range(5))
    ax.set_xticklabels(USERS)
    ps.percent_axis(ax)
    ax.set_ylabel(METRIC_LABEL)
    ax.set_title(f"CV por blocos — cada ponto é um fold (6 trials); traço = média  [{ps.MODEL_LABELS[model]}]")
    ax.legend(loc="upper left")
    ps.save(fig, out / "res_cv_folds_personalizado.png")


def fig_trial_vs_sample(df, out):
    """Acurácia por amostra × por trial (média das probabilidades no trial)."""
    variant = best_variant(df, "personalized", "blocked_cv")
    sl = subject_level(df, "blocked_cv", "personalized", [variant])
    if sl.empty or sl.trial.isna().all():
        return
    table = sl.groupby("subject")[["accuracy", "trial"]].mean().reindex(USERS)
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    _grouped_bars(ax, table, USERS, ["accuracy", "trial"],
                  {"accuracy": "#86b6ef", "trial": "#184f95"},
                  {"accuracy": "por amostra", "trial": "por trial (decisão sobre o trial inteiro)"})
    ax.set_xticklabels(USERS)
    _chance_line(ax)
    ps.percent_axis(ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Acurácia (média dos 4 modelos)")
    ax.set_title(f"Decidir por trial em vez de por amostra — {VARIANT_LABELS[variant]}, CV")
    ax.legend(loc="upper left")
    ps.save(fig, out / "res_trial_vs_amostra.png")


def fig_confusions(out, df):
    from sklearn.metrics import confusion_matrix

    variant = best_variant(df, "personalized", "blocked_cv")
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.0))
    for ax, model in zip(axes, MODELS):
        parts = [load_cv_predictions(variant, "personalized", model, u) for u in USERS]
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
    fig.suptitle(f"Matrizes de confusão (recall por linha) — personalizado, {VARIANT_LABELS[variant]}, "
                 "CV (5 sujeitos somados)", x=0.02, ha="left", fontsize=10, fontweight="semibold")
    fig.tight_layout()
    ps.save(fig, out / "res_matrizes_confusao.png")


def fig_best_epoch(df, out):
    """Distribuição da melhor época (min val_loss): diagnóstico de overfitting precoce."""
    sub = df[(df.subject != "ALL") | (df.track == "pooled")]
    sub = sub[((sub.iteration == "v1") | ((sub.protocol == "chronological") & (sub.rep == "seed_42")))
              & sub.best_epoch.notna()]
    sub = sub.drop_duplicates(["iteration", "variant", "track", "model", "subject", "rep"])
    groups = [("v1", "raw"), ("v2", "log")] + [
        ("v2", v) for v in ["log_rel", "log_subjnorm", "log_rel_subjnorm"]]
    data = [(f"{it} · {VARIANT_LABELS[v]}", sub[(sub.iteration == it) & (sub.variant == v)].best_epoch)
            for it, v in groups]
    data = [(n, d) for n, d in data if len(d)]
    if not data:
        return
    fig, ax = plt.subplots(figsize=(6.4, 0.55 * len(data) + 1.2))
    for i, (name, values) in enumerate(data):
        values = values.to_numpy(dtype=float)
        jitter = np.random.default_rng(i).uniform(-0.15, 0.15, len(values))
        ax.scatter(values, np.full(len(values), i) + jitter, s=14,
                   color=ps.V1_COLOR if name.startswith("v1") else ps.V2_COLOR,
                   edgecolor=ps.SURFACE, linewidth=0.6)
        ax.text(ax.get_xlim()[1] if False else 0.0, i, "", va="center")
        ax.plot([np.median(values)] * 2, [i - 0.3, i + 0.3], color=ps.INK, linewidth=1.6)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels([n for n, _ in data], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Época de menor val_loss (traço = mediana)")
    ax.set_title("Quando a validação para de melhorar (hold-out cronológico)")
    ps.save(fig, out / "res_melhor_epoca.png")


def fig_pooled_vs_personal(df, out):
    rows = []
    for track in ["pooled", "personalized"]:
        variant = best_variant(df, track, "blocked_cv")
        sl = subject_level(df, "blocked_cv", track, [variant])
        for user in USERS:
            rows.append(dict(track=track, subject=user,
                             value=sl[sl.subject == user].value.mean(), variant=variant))
    res = pd.DataFrame(rows).dropna()
    if res.empty:
        return
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    for i, user in enumerate(USERS):
        p = res[(res.subject == user) & (res.track == "pooled")].value
        q = res[(res.subject == user) & (res.track == "personalized")].value
        if p.empty or q.empty:
            continue
        ax.plot([p.iloc[0], q.iloc[0]], [i, i], color=ps.AXIS, linewidth=2, zorder=1)
        ax.scatter(p, [i], s=50, color=ps.SERIES[1], edgecolor=ps.SURFACE, linewidth=1.5, zorder=2,
                   label="agrupado" if i == 0 else None)
        ax.scatter(q, [i], s=50, color=ps.SERIES[0], edgecolor=ps.SURFACE, linewidth=1.5, zorder=2,
                   label="personalizado" if i == 0 else None)
    ax.axvline(CHANCE, color=ps.INK_MUTED, linewidth=1)
    ax.set_yticks(range(5))
    ax.set_yticklabels(USERS)
    ax.invert_yaxis()
    ps.percent_axis(ax, "x")
    ax.set_xlabel(f"{METRIC_LABEL} (média dos 4 modelos, CV)  — linha vertical: acaso")
    ax.set_title("Agrupado × personalizado, melhor variante de cada track")
    ax.legend(loc="lower right")
    ps.save(fig, out / "res_agrupado_vs_personalizado.png")


def resultados(df):
    out = FIG_ROOT / "resultados"
    for protocol in ["chronological", "blocked_cv"]:
        for track in ["pooled", "personalized"]:
            fig_ablation(df, protocol, track, out)
            fig_subject_bars(df, protocol, track, out)
        fig_delta_heatmap(df, protocol, out)
    fig_v1_vs_v2(df, out)
    fig_cv_folds(df, out)
    fig_trial_vs_sample(df, out)
    fig_confusions(out, df)
    fig_best_epoch(df, out)
    fig_pooled_vs_personal(df, out)


# ---------------------------------------------------------------------------
# Explicabilidade
# ---------------------------------------------------------------------------

def explicabilidade(df):
    from src.explainability import model_explain as me
    from src.preprocessing.base_preprocessing import prepare_base_data
    from run_v2 import VARIANTS, build_config

    out = FIG_ROOT / "explicabilidade"
    models_root = ITER_ROOT / "models" / "chronological"
    summary: dict = {}

    # A CV é a estimativa principal: ela escolhe a variante e a arquitetura
    # analisadas. Os modelos analisados são os do hold-out cronológico
    # (semente 42), os únicos salvos em disco.
    variant = best_variant(df, "personalized", "blocked_cv")
    arch = best_model(df, "personalized", "blocked_cv", variant)
    summary["personalized_variant"] = variant
    summary["architecture"] = arch
    print(f"  explicabilidade: personalizado {variant}, arquitetura {arch}")

    preps = {}
    for user in USERS:
        preps[user] = prepare_base_data(build_config("chronological", VARIANTS[variant], user, 0))

    def model_path(var, track, model, user=None):
        path = models_root / var / track
        if user:
            path = path / user
        return path / "seed_42" / f"{model}_model.keras"

    # 1. Permutação (teste e treino) e 2. IG por classe — arquitetura escolhida, 5 sujeitos
    perm_test, perm_train, ig_rows = [], [], []
    summary["permutation"] = {}
    for user in USERS:
        path = model_path(variant, "personalized", arch, user)
        if not path.exists():
            continue
        prep = preps[user]
        model = me.load_model(path)
        test = me.permutation_importance(model, arch, prep.X_test, prep.y_test)
        train = me.permutation_importance(model, arch, prep.X_train, prep.y_train)
        perm_test.append((f"{user} ({test['baseline_balanced_accuracy']:.0%})", test["grid"]))
        perm_train.append((f"{user} ({train['baseline_balanced_accuracy']:.0%})", train["grid"]))
        summary["permutation"][user] = {
            "test": {k: v for k, v in test.items() if k != "grid"} | {"grid": test["grid"].tolist()},
            "train": {k: v for k, v in train.items() if k != "grid"} | {"grid": train["grid"].tolist()},
        }
        ig_rows.append((user, me.class_attribution_grids(model, arch, prep.X_test, prep.y_test)))

    if perm_test:
        me.plot_importance_grids(
            perm_test, f"Importância por permutação no TESTE — {ps.MODEL_LABELS[arch]} personalizado "
            f"({VARIANT_LABELS[variant]}); entre parênteses, acurácia balanceada",
            out / "exp_01_permutacao_teste.png", ncols=5)
        me.plot_importance_grids(
            perm_train, f"Importância por permutação no TREINO — o que o modelo aprendeu "
            f"({ps.MODEL_LABELS[arch]})", out / "exp_02_permutacao_treino.png", ncols=5)
        me.plot_class_attributions(
            ig_rows, f"Integrated Gradients por classe — {ps.MODEL_LABELS[arch]} personalizado (teste)",
            out / "exp_03_integrated_gradients.png")
        summary["integrated_gradients"] = {u: g.tolist() for u, g in ig_rows}

    # 3. Atenção do Transformer (personalizado, 5 sujeitos)
    att_rows = []
    for user in USERS:
        path = model_path(variant, "personalized", "transformer", user)
        if path.exists():
            prep = preps[user]
            att_rows.append((user, me.transformer_cls_attention(me.load_model(path), prep.X_test, prep.y_test)))
    if att_rows:
        me.plot_attention(att_rows, out / "exp_04_atencao_transformer.png")
        summary["attention"] = {u: a.tolist() for u, a in att_rows}

    # 4. Embeddings do modelo agrupado: raw × melhor variante
    pooled_variant = best_variant(df, "pooled", "blocked_cv")
    pooled_arch = best_model(df, "pooled", "blocked_cv", pooled_variant)
    panels = []
    sample_users = sample_labels = None
    for var in ["raw", pooled_variant]:
        path = model_path(var, "pooled", pooled_arch)
        if not path.exists():
            continue
        prep = prepare_base_data(build_config("chronological", VARIANTS[var], None, 0))
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(len(prep.y_test), min(3000, len(prep.y_test)), replace=False))
        emb = me.penultimate_embeddings(me.load_model(path), pooled_arch, prep.X_test[idx])
        panels.append((f"{ps.MODEL_LABELS[pooled_arch]} agrupado — {VARIANT_LABELS[var]}", emb))
        sample_users, sample_labels = prep.users_test[idx], prep.y_test[idx]
    if panels:
        summary["embeddings"] = me.plot_embeddings(
            panels, sample_users, sample_labels,
            "Representação interna (penúltima camada) do modelo agrupado, amostras de teste",
            out / "exp_05_embeddings_agrupado.png")

    # 5. Timelines (personalizado, arquitetura escolhida, hold-out cronológico)
    rows = []
    for user in USERS:
        data = load_predictions("chronological", variant, "personalized", arch, user)
        if data is not None:
            rows.append((user, data))
    if rows:
        me.plot_timelines(rows, out / "exp_06_timeline_predicoes.png")

    # 6. Calibração (personalizado, CV concatenada, 4 modelos)
    cv_variant = best_variant(df, "personalized", "blocked_cv")
    curves = {}
    for model in MODELS:
        parts = [load_cv_predictions(cv_variant, "personalized", model, u) for u in USERS]
        parts = [p for p in parts if p is not None]
        if parts:
            curves[model] = me.calibration_curve(
                np.concatenate([p["y_true"] for p in parts]),
                np.concatenate([p["y_pred_proba"] for p in parts]))
    if curves:
        me.plot_calibration(curves, f"Calibração — personalizado, {VARIANT_LABELS[cv_variant]}, CV",
                            out / "exp_07_calibracao.png")
        summary["calibration"] = curves

    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    (ANALYSIS_ROOT / "model_explain.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")


def dados():
    from src.explainability.data_analysis import run_all

    run_all(FIG_ROOT / "dados", ANALYSIS_ROOT / "data_analysis.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=["dados", "resultados", "explicabilidade"],
                        default=["dados", "resultados", "explicabilidade"])
    args = parser.parse_args()

    df = collect_all() if {"resultados", "explicabilidade"} & set(args.only) else None
    if df is not None:
        ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
        df.to_csv(ANALYSIS_ROOT / "resultados_longos.csv", index=False)

    if "dados" in args.only:
        dados()
    if "resultados" in args.only:
        resultados(df)
    if "explicabilidade" in args.only:
        explicabilidade(df)


if __name__ == "__main__":
    main()
