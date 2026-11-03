"""Estilo e figuras dos resultados.

Cores neutras, sem vermelho, verde, amarelo ou laranja (que leem como
certo/errado ou alerta). Paradigmas: generalista magenta (■), especialista
azul (●). Classes: Esquerda azul (●), Direita magenta (■), Neutro ocre (▲).
Divergente marrom ↔ cinza ↔ azul. Vírgula decimal.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, MultipleLocator  # noqa: E402

from src.constantes import EEG_BANDS, EEG_CHANNELS  # noqa: E402
from src.config import TRACK_NOME, TRACKS, USERS  # noqa: E402
from src.avaliacao.resultados import num  # noqa: E402

CHANCE = 1 / 3
TRACK_COLOR = {"pooled": "#b0478f", "personalized": "#2a78d6"}
TRACK_MARKER = {"pooled": "s", "personalized": "o"}
CLASS_NAMES = ["Esquerda", "Direita", "Neutro"]
CLASS_COLOR = ["#2a78d6", "#b0478f", "#a8742f"]
CLASS_MARKER = ["o", "s", "^"]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE, BAND = "#e1e0d9", "#c3c2b7", "#ffffff", "#efeee9"
DIVERGING = LinearSegmentedColormap.from_list(
    "marrom_azul", ["#5c3d14", "#8a5a22", "#b98546", "#dcc29b", "#f0efec",
                    "#9ec5f4", "#3987e5", "#184f95", "#0d366b"])
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "azul", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 9,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 10, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "legend.frameon": False, "lines.linewidth": 2,
})

pct_fmt = FuncFormatter(lambda v, _: f"{v * 100:.0f}%")


def salvar(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [figura] {path}")


def _jsonavel(obj):
    """numpy → listas e números do Python, com 4 casas; NaN vira null."""
    if isinstance(obj, dict):
        return {str(k): _jsonavel(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [_jsonavel(v) for v in (obj.tolist() if isinstance(obj, np.ndarray) else obj)]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if np.isnan(obj) else round(float(obj), 4)
    return obj


def salvar_dados(path: Path, dados: dict) -> None:
    """Os dados de uma figura, para o dashboard desenhar o gráfico interativo.

    <etapa>/figuras/<nome>.png → <etapa>/graficos/<nome>.json. Gravado pela mesma função
    que desenha a figura, com os mesmos números.
    """
    destino = path.parent.parent / "graficos" / f"{path.stem}.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(_jsonavel(dados), ensure_ascii=False), encoding="utf-8")


PROTOCOLO_NOME = {"holdout": "Hold-out por blocos", "cv": "Validação cruzada por blocos"}


def _serie(valores: dict, paradigma: str) -> dict:
    """Pontos por participante e a média (o formato das figuras de resultado)."""
    vals = {u: valores[u] for u in USERS if u in valores}
    return {"paradigma": paradigma, "valores": vals, "media": float(np.mean(list(vals.values())))}


def _dados_calibracao(cal: dict, rotulo: str, paradigma: str) -> dict:
    return {"rotulo": rotulo, "paradigma": paradigma, "confianca": cal["confidence"], "acuracia": cal["accuracy"],
            "contagens": cal["counts"], "ece": cal["ece"], "confianca_media": cal["mean_confidence"],
            "acuracia_geral": cal["accuracy_overall"]}


def linha_acaso(ax, axis="x", label=True):
    (ax.axvline if axis == "x" else ax.axhline)(CHANCE, color=INK2, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    if label:
        if axis == "x":
            ax.text(CHANCE, 1.0, " acaso (33,3%)", transform=ax.get_xaxis_transform(),
                    fontsize=7.5, color=INK2, va="bottom", ha="center")
        else:
            ax.text(1.0, CHANCE, " acaso", transform=ax.get_yaxis_transform(), fontsize=7.5, color=INK2, va="center")


def _mean(d: dict) -> float:
    return float(np.mean(list(d.values())))


# ---------------------------------------------------------------------------
# Figuras de um modelo
# ---------------------------------------------------------------------------

def fig_resultado_modelo(nome: str, final: dict, baseline: dict | None, path: Path) -> None:
    """final[(protocolo, track)] = {participante: acurácia}; baseline idem (ou None)."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 2.9), sharey=True, gridspec_kw={"wspace": 0.15})
    for ax, protocolo in zip(axes, ("holdout", "cv")):
        for i, track in enumerate(TRACKS):
            s = final[(protocolo, track)]
            vals = np.array([s[u] for u in USERS if u in s])
            ax.scatter(vals, np.full(len(vals), i), s=18, color=TRACK_COLOR[track], alpha=0.4, linewidths=0,
                       marker=TRACK_MARKER[track], zorder=2)
            m = vals.mean()
            ax.scatter([m], [i], s=80, color=TRACK_COLOR[track], marker=TRACK_MARKER[track],
                       edgecolors=SURFACE, linewidths=1.2, zorder=3)
            ax.text(m, i - 0.28, num(m * 100), ha="center", fontsize=8, color=TRACK_COLOR[track],
                    fontweight="semibold")
            if baseline:
                b = _mean(baseline[(protocolo, track)])
                ax.scatter([b], [i], s=60, marker="D", facecolor=SURFACE, edgecolor=INK2, linewidths=1.3, zorder=3)
        linha_acaso(ax)
        ax.set_xlim(0.1, 0.8)
        ax.xaxis.set_major_formatter(pct_fmt)
        ax.set_yticks(range(len(TRACKS)), [TRACK_NOME[t] for t in TRACKS])
        ax.set_ylim(len(TRACKS) - 0.5, -0.6)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Acurácia balanceada no teste")
        ax.set_title({"holdout": "Hold-out por blocos", "cv": "Validação cruzada por blocos"}[protocolo], pad=14)
    handles = [Line2D([], [], color=MUTED, marker="o", linestyle="", markersize=8, label="média"),
               Line2D([], [], color=MUTED, marker="o", linestyle="", markersize=4, alpha=0.5, label="cada participante")]
    if baseline:
        handles.append(Line2D([], [], marker="D", linestyle="", markersize=6, markerfacecolor=SURFACE,
                              markeredgecolor=INK2, label="regressão logística (baseline, média)"))
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle(nome, x=0.06, ha="left", fontsize=11, fontweight="semibold", y=1.06)
    salvar(fig, path)
    salvar_dados(path, {"protocolos": [
        {"protocolo": PROTOCOLO_NOME[p],
         "linhas": [{"rotulo": nome, "series": [_serie(final[(p, t)], TRACK_NOME[t]) for t in TRACKS]}]}
        for p in ("holdout", "cv")]})


def fig_matriz_confusao(nome: str, matrizes: dict, path: Path) -> None:
    """matrizes[track] = matriz 3×3 normalizada por linha (hold-out, todos os treinos)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
    for ax, track in zip(axes, TRACKS):
        cm = matrizes[track]
        ax.imshow(cm, cmap=SEQUENTIAL, vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cm[i, j] * 100:.0f}%", ha="center", va="center", fontsize=9,
                        color=SURFACE if cm[i, j] > 0.55 else INK)
        ax.set_xticks(range(3), CLASS_NAMES)
        ax.set_yticks(range(3), CLASS_NAMES)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        ax.set_xlabel("Classe prevista")
        ax.set_ylabel("Classe verdadeira")
        ax.set_title(f"{TRACK_NOME[track]} (hold-out)")
    fig.suptitle(f"{nome}: matriz de confusão (% de cada classe verdadeira)", x=0.02, ha="left",
                 fontsize=10, fontweight="semibold")
    fig.tight_layout()
    salvar(fig, path)
    salvar_dados(path, {"classes": CLASS_NAMES,
                        "paradigmas": [{"paradigma": TRACK_NOME[t], "matriz": matrizes[t]} for t in TRACKS]})


def _curva_mediana(hists: list[dict], key: str):
    n_max = max(len(h[key]) for h in hists)
    med, epochs = [], []
    for e in range(n_max):
        vals = [h[key][e] for h in hists if len(h[key]) > e]
        if len(vals) >= len(hists) / 2:
            med.append(np.median(vals))
            epochs.append(e + 1)
    return epochs, med


def _dados_curvas(rotulo: str, runs: list) -> dict:
    hists = [r.meta["history"] for r in runs]
    treino, validacao = _curva_mediana(hists, "accuracy"), _curva_mediana(hists, "val_accuracy")
    return {"rotulo": rotulo, "treino": {"epocas": treino[0], "valores": treino[1]},
            "validacao": {"epocas": validacao[0], "valores": validacao[1]},
            "melhor_epoca": int(np.median([r.meta["best_epoch"] for r in runs]))}


def fig_curvas(nome: str, runs_por_track: dict, path: Path) -> None:
    """runs_por_track[track] = lista de Treino (hold-out + CV)."""
    salvar_dados(path, {"paradigmas": [{"paradigma": TRACK_NOME[t], "linhas": [_dados_curvas(nome, runs_por_track[t])]}
                                       for t in TRACKS]})
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.0), sharey=True)
    for ax, track in zip(axes, TRACKS):
        runs = runs_por_track[track]
        hists = [r.meta["history"] for r in runs]
        for key, color, label in (("accuracy", INK2, "treino"), ("val_accuracy", TRACK_COLOR[track], "validação")):
            epochs, med = _curva_mediana(hists, key)
            ax.plot(epochs, med, color=color, label=label, marker="o", markersize=2.5)
        best = int(np.median([r.meta["best_epoch"] for r in runs]))
        ax.axvline(best, color=MUTED, linewidth=1, linestyle=(0, (2, 2)))
        ax.text(best, 1.0, f" melhor época\n (mediana {best})", fontsize=7, color=MUTED, va="top")
        linha_acaso(ax, axis="y", label=False)
        ax.set_ylim(0.25, 1.02)
        ax.yaxis.set_major_formatter(pct_fmt)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.set_xlabel("Época")
        ax.set_title(TRACK_NOME[track])
        ax.legend(loc="lower right")
    axes[0].set_ylabel("Acurácia (mediana dos treinos)")
    fig.suptitle(f"{nome}: curvas de treino", x=0.02, ha="left", fontsize=10, fontweight="semibold")
    fig.tight_layout()
    salvar(fig, path)


def fig_permutacao_modelo(nome: str, real: dict, permutado: dict, path: Path) -> None:
    """real[track], permutado[track] = {participante: acurácia} (hold-out)."""
    salvar_dados(path, {"paradigmas": [
        {"paradigma": TRACK_NOME[t],
         "linhas": [{"rotulo": nome, "verdadeiro": _mean(real[t]), "permutado": _mean(permutado[t])}]}
        for t in TRACKS]})
    fig, ax = plt.subplots(figsize=(6.4, 2.2))
    for i, track in enumerate(TRACKS):
        r, p = _mean(real[track]), _mean(permutado[track])
        ax.plot([p, r], [i, i], color=AXIS, linewidth=2, zorder=2)
        ax.scatter([p], [i], s=55, facecolor=SURFACE, edgecolor=INK2, linewidths=1.5, zorder=3)
        ax.scatter([r], [i], s=60, color=TRACK_COLOR[track], marker=TRACK_MARKER[track], zorder=3)
        ax.text(r + 0.008, i - 0.25, num(r * 100), fontsize=8, color=INK)
        ax.text(p - 0.008, i - 0.25, num(p * 100), fontsize=8, color=INK2, ha="right")
    linha_acaso(ax)
    ax.set_xlim(0.15, 0.6)
    ax.xaxis.set_major_formatter(pct_fmt)
    ax.set_yticks(range(len(TRACKS)), [TRACK_NOME[t] for t in TRACKS])
    ax.set_ylim(len(TRACKS) - 0.5, -0.7)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Acurácia balanceada no teste (hold-out)")
    ax.set_title(f"{nome}: rótulos verdadeiros × permutados entre blocos", pad=14)
    handles = [Line2D([], [], marker="o", linestyle="", markersize=7, markerfacecolor=SURFACE,
                      markeredgecolor=INK2, label="rótulos permutados"),
               Line2D([], [], marker="o", linestyle="", markersize=7, color=MUTED, label="rótulos verdadeiros")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.18))
    salvar(fig, path)


def fig_calibracao_modelo(nome: str, curvas: dict, path: Path) -> None:
    """curvas[track] = saída de resultados.calibracao (CV, rodadas concatenadas)."""
    salvar_dados(path, {"series": [_dados_calibracao(curvas[t], TRACK_NOME[t], TRACK_NOME[t]) for t in TRACKS]})
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.3), sharey=True)
    for ax, track in zip(axes, TRACKS):
        cal = curvas[track]
        ax.plot([CHANCE, 1], [CHANCE, 1], color=AXIS, linewidth=1.2)
        ax.plot(cal["confidence"], cal["accuracy"], color=TRACK_COLOR[track], marker=TRACK_MARKER[track],
                markersize=4, markeredgecolor=SURFACE)
        ax.set_xlim(0.3, 1.0)
        ax.set_ylim(0.2, 1.0)
        ax.set_aspect("equal")
        ax.xaxis.set_major_formatter(pct_fmt)
        ax.yaxis.set_major_formatter(pct_fmt)
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.set_title(f"{TRACK_NOME[track]} · ECE {num(cal['ece'], 2)}", fontsize=9.5)
        ax.set_xlabel("Confiança prevista")
        ax.text(0.98, 0.04, f"confiança média {num(cal['mean_confidence'] * 100, 0)}%\n"
                f"acurácia {num(cal['accuracy_overall'] * 100, 0)}%", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7, color=INK2)
    axes[0].set_ylabel("Acurácia observada")
    fig.suptitle(f"{nome}: calibração (validação cruzada)", x=0.02, ha="left", fontsize=10, fontweight="semibold")
    fig.tight_layout()
    salvar(fig, path)


def fig_xai_grades(nome: str, grades: dict, path: Path) -> None:
    """grades[track] = (3, 5, 5): IG médio por classe, eletrodo (linhas) × banda (colunas)."""
    vmax = max(float(np.abs(g).max()) for g in grades.values()) or 1.0
    salvar_dados(path, {"eletrodos": EEG_CHANNELS, "bandas": EEG_BANDS, "classes": CLASS_NAMES, "vmax": vmax,
                        "paradigmas": [{"paradigma": TRACK_NOME[t], "grades": grades[t]} for t in TRACKS]})
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    fig, axes = plt.subplots(2, 3, figsize=(8.4, 5.6), layout="constrained")
    for r, track in enumerate(TRACKS):
        for c in range(3):
            ax = axes[r, c]
            im = ax.imshow(grades[track][c], cmap=DIVERGING, norm=norm)
            ax.set_xticks(range(5), EEG_BANDS, fontsize=7.5, rotation=30)
            ax.set_yticks(range(5), EEG_CHANNELS, fontsize=7.5)
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.tick_params(length=0)
            ax.set_title(f"{TRACK_NOME[track]} · {CLASS_NAMES[c]}", fontsize=9)
    cbar = fig.colorbar(im, ax=axes, shrink=0.7)
    cbar.outline.set_visible(False)
    cbar.set_label("Contribuição média para a classe (IG)\nazul = a favor · marrom = contra", color=INK2)
    fig.suptitle(f"{nome}: Integrated Gradients por eletrodo × banda (soma na janela)", x=0.02, ha="left",
                 fontsize=10, fontweight="semibold")
    salvar(fig, path)


def fig_xai_tempo(nome: str, tempos: dict, path: Path, records_per_second: float = 8) -> None:
    """tempos[track] = (3, T): |IG| médio por posição da janela, por classe."""
    salvar_dados(path, {"classes": CLASS_NAMES, "paradigmas": [
        {"paradigma": TRACK_NOME[t], "T": tempos[t].shape[1],
         "segundos": (np.arange(tempos[t].shape[1]) - (tempos[t].shape[1] - 1)) / records_per_second,
         "fracoes": tempos[t] / tempos[t].sum(axis=1, keepdims=True)}
        for t in TRACKS]})
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 2.9))
    for ax, track in zip(axes, TRACKS):
        t = tempos[track]
        x = (np.arange(t.shape[1]) - (t.shape[1] - 1)) / records_per_second
        for c in range(3):
            share = t[c] / t[c].sum()
            ax.plot(x, share, color=CLASS_COLOR[c], marker=CLASS_MARKER[c], markersize=2.5, linewidth=1.4,
                    label=CLASS_NAMES[c])
        ax.axhline(1 / t.shape[1], color=INK2, linewidth=1, linestyle=(0, (4, 3)))
        ax.set_xlabel("Segundos antes do registro classificado (0 = atual)")
        ax.set_title(f"{TRACK_NOME[track]} (T = {t.shape[1]})")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:.0f}%".replace(".", ",")))
    axes[0].set_ylabel("Fração do |IG| total")
    axes[1].legend(loc="upper left")
    fig.suptitle(f"{nome}: onde na janela está a informação usada (linha tracejada = uniforme)", x=0.02,
                 ha="left", fontsize=10, fontweight="semibold")
    fig.tight_layout()
    salvar(fig, path)


def fig_atencao(nome: str, atencao: dict, offsets: dict, path: Path) -> None:
    """atencao[track] = (3, n_tokens) da última camada; offsets[track] = tempo de cada token (s)."""
    salvar_dados(path, {"classes": CLASS_NAMES, "paradigmas": [
        {"paradigma": TRACK_NOME[t], "n_tokens": atencao[t].shape[1], "segundos": offsets[t], "atencao": atencao[t]}
        for t in TRACKS]})
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 2.9))
    for ax, track in zip(axes, TRACKS):
        a = atencao[track]
        for c in range(3):
            ax.plot(offsets[track], a[c], color=CLASS_COLOR[c], marker=CLASS_MARKER[c], markersize=3,
                    linewidth=1.4, label=CLASS_NAMES[c])
        ax.axhline(1 / a.shape[1], color=INK2, linewidth=1, linestyle=(0, (4, 3)))
        ax.set_xlabel("Segundos antes do registro classificado (0 = atual)")
        ax.set_title(f"{TRACK_NOME[track]} ({a.shape[1]} tokens)")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))
    axes[0].set_ylabel("Atenção do [CLS]")
    axes[1].legend(loc="upper left")
    fig.suptitle(f"{nome}: atenção do [CLS] sobre os tokens de tempo (linha tracejada = uniforme)", x=0.02,
                 ha="left", fontsize=10, fontweight="semibold")
    fig.tight_layout()
    salvar(fig, path)


# ---------------------------------------------------------------------------
# Figuras comparativas (todos os modelos)
# ---------------------------------------------------------------------------

def fig_principal(linhas: list[str], rotulos: list[str], final: dict, path: Path,
                  baseline_key: str | None = None) -> None:
    """final[(linha, protocolo, track)] = {participante: acurácia}."""
    salvar_dados(path, {"protocolos": [
        {"protocolo": PROTOCOLO_NOME[p],
         "linhas": [{"rotulo": rot, "series": [_serie(final[(linha, p, t)], TRACK_NOME[t]) for t in TRACKS]}
                    for linha, rot in zip(linhas, rotulos)]}
        for p in ("holdout", "cv")]})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True, gridspec_kw={"wspace": 0.18})
    for ax, protocolo in zip(axes, ("holdout", "cv")):
        for i, linha in enumerate(linhas):
            if linha == baseline_key:
                ax.axhspan(i - 0.45, i + 0.45, color=BAND, zorder=0, linewidth=0)
            for track, dy in (("pooled", -0.17), ("personalized", 0.17)):
                s = final[(linha, protocolo, track)]
                vals = np.array([s[u] for u in USERS if u in s])
                ax.scatter(vals, np.full(len(vals), i + dy), s=14, color=TRACK_COLOR[track], alpha=0.35,
                           linewidths=0, marker=TRACK_MARKER[track], zorder=2)
                m = vals.mean()
                ax.scatter([m], [i + dy], s=70, color=TRACK_COLOR[track], marker=TRACK_MARKER[track],
                           edgecolors=SURFACE, linewidths=1.2, zorder=3)
                ax.text(1.06, i + dy, num(m * 100), transform=ax.get_yaxis_transform(), va="center",
                        ha="right", fontsize=8, color=TRACK_COLOR[track], fontweight="semibold")
        linha_acaso(ax)
        ax.set_xlim(0.0, 0.80)
        ax.xaxis.set_major_formatter(pct_fmt)
        ax.xaxis.set_major_locator(MultipleLocator(0.1))
        ax.set_yticks(range(len(linhas)), rotulos)
        ax.set_ylim(len(linhas) - 0.5, -0.5)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Acurácia balanceada no teste")
        ax.set_title({"holdout": "Hold-out por blocos", "cv": "Validação cruzada por blocos"}[protocolo], pad=14)
        ax.text(1.06, -0.62, "média", transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                fontsize=7.5, color=MUTED)
    handles = [Line2D([], [], color=TRACK_COLOR[t], marker=TRACK_MARKER[t], linestyle="", markersize=7,
                      label=f"{TRACK_NOME[t]} (média)") for t in TRACKS]
    handles += [Line2D([], [], color=MUTED, marker="o", linestyle="", markersize=4, alpha=0.5,
                       label="cada participante")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    salvar(fig, path)


def fig_por_participante(colunas: list[str], rotulos: list[str], final: dict, n_modelos: int, path: Path) -> None:
    salvar_dados(path, {"participantes": USERS, "modelos": rotulos[:n_modelos], "cenarios": [
        {"protocolo": PROTOCOLO_NOME[p], "paradigma": TRACK_NOME[t],
         "matriz": [[final[(m, p, t)].get(u, np.nan) for m in colunas[:n_modelos]] for u in USERS]}
        for p in ("holdout", "cv") for t in TRACKS]})
    norm = TwoSlopeNorm(vmin=0.10, vcenter=CHANCE, vmax=0.80)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), layout="constrained")
    for r, protocolo in enumerate(("holdout", "cv")):
        for c, track in enumerate(TRACKS):
            ax = axes[r, c]
            M = np.array([[final[(m, protocolo, track)].get(u, np.nan) for m in colunas] for u in USERS])
            im = ax.imshow(M, cmap=DIVERGING, norm=norm, aspect="auto")
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    v = M[i, j]
                    ax.text(j, i, f"{v * 100:.0f}", ha="center", va="center", fontsize=8,
                            color=SURFACE if (v > 0.60 or v < 0.17) else INK)
            if n_modelos < len(colunas):  # separa as linhas extras (ex.: baseline) dos modelos
                ax.axvline(n_modelos - 0.5, color=SURFACE, linewidth=3)
            ax.set_xticks(range(len(colunas)), rotulos, rotation=30, ha="right", fontsize=8)
            ax.set_yticks(range(len(USERS)), USERS)
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.tick_params(length=0)
            ax.set_title(f"{'Hold-out' if protocolo == 'holdout' else 'Validação cruzada'} · {TRACK_NOME[track]}",
                         fontsize=9.5)
            if r == 0:
                ax.set_xticklabels([])
    cbar = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.03, ticks=[0.1, 0.2, CHANCE, 0.5, 0.6, 0.7, 0.8])
    cbar.ax.set_yticklabels(["10%", "20%", "33% (acaso)", "50%", "60%", "70%", "80%"])
    cbar.outline.set_visible(False)
    cbar.set_label("Acurácia balanceada no teste", color=INK2)
    salvar(fig, path)


def fig_permutacao(linhas: list[str], rotulos: list[str], real: dict, perm: dict, path: Path,
                   baseline_key: str | None = None) -> None:
    """real[(linha, track)], perm[(linha, track)] = {participante: acurácia} (hold-out)."""
    salvar_dados(path, {"paradigmas": [
        {"paradigma": TRACK_NOME[t],
         "linhas": [{"rotulo": rot, "verdadeiro": _mean(real[(linha, t)]), "permutado": _mean(perm[(linha, t)])}
                    for linha, rot in zip(linhas, rotulos)]}
        for t in TRACKS]})
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9), sharey=True)
    for ax, track in zip(axes, TRACKS):
        for i, linha in enumerate(linhas):
            if linha == baseline_key:
                ax.axhspan(i - 0.45, i + 0.45, color=BAND, zorder=0, linewidth=0)
            r, p = _mean(real[(linha, track)]), _mean(perm[(linha, track)])
            ax.plot([p, r], [i, i], color=AXIS, linewidth=2, zorder=2, solid_capstyle="round")
            ax.scatter([p], [i], s=55, facecolor=SURFACE, edgecolor=INK2, linewidths=1.5, zorder=3)
            ax.scatter([r], [i], s=60, color=TRACK_COLOR[track], marker=TRACK_MARKER[track], zorder=3,
                       edgecolors=SURFACE, linewidths=1)
            ax.text(r + 0.008, i - 0.22, num(r * 100), fontsize=7.5, color=INK, ha="left")
            ax.text(p - 0.008, i - 0.22, num(p * 100), fontsize=7.5, color=INK2, ha="right")
        linha_acaso(ax)
        ax.set_xlim(0.18, 0.56)
        ax.xaxis.set_major_formatter(pct_fmt)
        ax.set_yticks(range(len(linhas)), rotulos)
        ax.set_ylim(len(linhas) - 0.5, -0.6)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Acurácia balanceada no teste (hold-out)")
        ax.set_title(TRACK_NOME[track], pad=14)
    handles = [Line2D([], [], marker="o", linestyle="", markersize=7, markerfacecolor=SURFACE,
                      markeredgecolor=INK2, markeredgewidth=1.5, label="rótulos permutados entre blocos"),
               Line2D([], [], marker="o", linestyle="", markersize=7, color=TRACK_COLOR["personalized"],
                      label="rótulos verdadeiros (especialista)"),
               Line2D([], [], marker="s", linestyle="", markersize=7, color=TRACK_COLOR["pooled"],
                      label="rótulos verdadeiros (generalista)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    salvar(fig, path)


def fig_curvas_todos(modelos: list[str], rotulos: dict, runs: dict, path: Path) -> None:
    """runs[(modelo, track)] = lista de Treino (hold-out + CV)."""
    salvar_dados(path, {"paradigmas": [
        {"paradigma": TRACK_NOME[t], "linhas": [_dados_curvas(rotulos[m], runs[(m, t)]) for m in modelos]}
        for t in TRACKS]})
    fig, axes = plt.subplots(2, len(modelos), figsize=(12, 5.0), sharex=True, sharey=True)
    for r, track in enumerate(TRACKS):
        for c, modelo in enumerate(modelos):
            ax = axes[r, c]
            hists = [run.meta["history"] for run in runs[(modelo, track)]]
            for key, color in (("accuracy", INK2), ("val_accuracy", TRACK_COLOR[track])):
                epochs, med = _curva_mediana(hists, key)
                ax.plot(epochs, med, color=color, marker="o", markersize=2.5)
            best = int(np.median([run.meta["best_epoch"] for run in runs[(modelo, track)]]))
            ax.axvline(best, color=MUTED, linewidth=1, linestyle=(0, (2, 2)))
            linha_acaso(ax, axis="y", label=False)
            ax.set_ylim(0.25, 1.02)
            ax.yaxis.set_major_formatter(pct_fmt)
            ax.xaxis.set_major_locator(MultipleLocator(5))
            if r == 0:
                ax.set_title(rotulos[modelo], fontsize=9.5)
            if c == 0:
                ax.set_ylabel(f"{TRACK_NOME[track]}\nacurácia (mediana)")
            if r == 1:
                ax.set_xlabel("Época")
    handles = [Line2D([], [], color=INK2, linewidth=2, label="treino"),
               Line2D([], [], color=TRACK_COLOR["pooled"], linewidth=2, label="validação (generalista)"),
               Line2D([], [], color=TRACK_COLOR["personalized"], linewidth=2, label="validação (especialista)"),
               Line2D([], [], color=MUTED, linewidth=1, linestyle=(0, (2, 2)), label="melhor época (mediana)"),
               Line2D([], [], color=INK2, linewidth=1, linestyle=(0, (4, 3)), label="acaso")]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    salvar(fig, path)


def fig_calibracao_todos(modelos: list[str], rotulos: dict, curvas: dict, path: Path) -> None:
    """curvas[modelo] = calibração do especialista na CV (rodadas concatenadas)."""
    salvar_dados(path, {"series": [_dados_calibracao(curvas[m], rotulos[m], TRACK_NOME["personalized"])
                                   for m in modelos]})
    fig, axes = plt.subplots(1, len(modelos), figsize=(12, 2.9), sharey=True)
    for ax, modelo in zip(axes, modelos):
        cal = curvas[modelo]
        ax.plot([CHANCE, 1], [CHANCE, 1], color=AXIS, linewidth=1.2)
        ax.plot(cal["confidence"], cal["accuracy"], color=TRACK_COLOR["personalized"], marker="o", markersize=4,
                markeredgecolor=SURFACE)
        ax.set_xlim(0.3, 1.0)
        ax.set_ylim(0.2, 1.0)
        ax.set_aspect("equal")
        ax.xaxis.set_major_formatter(pct_fmt)
        ax.yaxis.set_major_formatter(pct_fmt)
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.set_title(f"{rotulos[modelo]}\nECE {num(cal['ece'], 2)}", fontsize=9.5)
        ax.set_xlabel("Confiança prevista")
        ax.text(0.98, 0.04, f"confiança média {num(cal['mean_confidence'] * 100, 0)}%\n"
                f"acurácia {num(cal['accuracy_overall'] * 100, 0)}%", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7, color=INK2)
    axes[0].set_ylabel("Acurácia observada")
    fig.tight_layout()
    salvar(fig, path)
