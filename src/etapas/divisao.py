"""Divisão dos dados usada por todos os modelos (não treina nada).

Para cada registro: log10 das 25 potências (5 eletrodos × 5 bandas) e
StandardScaler ajustado só no treino (por participante no generalista).
A divisão é sempre por blocos (trecho contínuo de uma classe; 36 por
participante, em 6 grupos de 6):

    hold-out   teste = grupo 6 (fim da sessão), validação = grupo 5, treino = grupos 1–4
    CV         cada grupo é teste uma vez (6 folds), com o grupo seguinte como validação

40 registros de purge separam as partições. Depois vêm as janelas causais de
T registros (sem cruzar bloco nem partição) e o teste comum: só registros com
pelo menos 63 anteriores no mesmo bloco, iguais para todos os modelos e T.

Confere as garantias anti-vazamento e grava as contagens de cada partição e a
linha do tempo da divisão.

Saída: resultados/divisao/ (tabelas/*.csv, figuras/)
"""

import numpy as np
from matplotlib.patches import Patch

from src.avaliacao import figuras as fg
from src.avaliacao.resultados import salvar_tabela
from src.config import EVAL_HISTORY, HOLDOUT_FOLD, N_FOLDS, PURGE, RESULTADOS, USERS
from src.dados.preparacao import janelas, preparar

SAIDA = RESULTADOS / "divisao"
PARTES = {"train": "treino", "val": "validação", "test": "teste"}
COR = {"train": "#c3c2b7", "val": "#a8742f", "test": "#2a78d6"}


def linha_do_tempo(prep, ax, user: str, title: str | None = None) -> None:
    """Cada registro do participante colorido pela partição (branco = purge)."""
    for parte, cor in COR.items():
        pos = getattr(prep, f"positions_{parte}")[getattr(prep, f"users_{parte}") == user]
        ax.vlines(pos, 0, 1, color=cor, linewidth=0.6)
    ax.set_yticks([])
    ax.set_ylabel(user, rotation=0, ha="right", va="center")
    ax.grid(False)
    for s in ("left", "top", "right"):
        ax.spines[s].set_visible(False)
    if title:
        ax.set_title(title)


def trechos(prep, user: str) -> list[list]:
    """Linha do tempo de um participante em trechos contínuos [início, fim, partição] (fim exclusivo).

    A partição é "train", "val", "test" ou "purge" (registros descartados entre partições).
    """
    pos = {p: getattr(prep, f"positions_{p}")[getattr(prep, f"users_{p}") == user] for p in PARTES}
    total = 1 + max(int(v.max()) for v in pos.values())
    rotulo = np.full(total, "purge", dtype=object)
    for p, v in pos.items():
        rotulo[v] = p
    inicio = np.flatnonzero(np.r_[True, rotulo[1:] != rotulo[:-1]])
    fim = np.r_[inicio[1:], total]
    return [[int(a), int(b), rotulo[a]] for a, b in zip(inicio, fim)]


def main(argv: list[str] | None = None) -> None:
    tab = SAIDA / "tabelas"

    print("Hold-out (generalista: os 5 participantes juntos)")
    _, prep, _ = preparar("holdout", "pooled", None, HOLDOUT_FOLD)
    rep = prep.preprocessing_report
    print(f"\nRegistros após a limpeza: {rep['shape_after_cleaning'][0]} | purge de {PURGE} registros entre "
          f"partições (total {rep.get('purged_rows', 0)}) | teste comum: registros com pelo menos "
          f"{EVAL_HISTORY - 1} anteriores no mesmo bloco")
    linhas = []
    for u in USERS:
        n = {p: int((getattr(prep, f"users_{p}") == u).sum()) for p in PARTES}
        total_u = 1 + max(int(getattr(prep, f"positions_{p}")[getattr(prep, f"users_{p}") == u].max()) for p in PARTES)
        blocos = rep["test_blocks_per_user"][u]
        linhas.append({"Participante": u, "Treino": n["train"], "Validação": n["val"], "Teste": n["test"],
                       "Purge": total_u - sum(n.values()), "Primeiro bloco de teste": blocos[0],
                       "Último bloco de teste": blocos[-1]})
    salvar_tabela(linhas, tab / "holdout.csv", "Hold-out por blocos: registros por partição")

    linhas = []
    for T in (16, 32, 64):
        w = janelas(prep, T, PURGE)
        linhas.append({"T (registros)": T, "Duração (s)": T / 8, "Janelas de treino": len(w.y_train),
                       "Validação": len(w.y_val), "Teste comum": len(w.y_test)})
    salvar_tabela(linhas, tab / "janelas.csv", "Janelas causais (hold-out, generalista)")

    print("\nValidação cruzada por blocos (generalista)")
    preps_cv, linhas = {}, []
    for fold in range(N_FOLDS):
        _, p, _ = preparar("cv", "pooled", None, fold)
        preps_cv[fold] = p
        r = p.preprocessing_report
        linhas.append({"Fold": fold, "Grupo de teste": fold + 1, "Grupo de validação": r["cv_val_fold"] + 1,
                       "Treino": len(p.y_train), "Validação": len(p.y_val), "Teste": len(p.y_test),
                       "Purge": r.get("purged_rows", 0)})
    salvar_tabela(linhas, tab / "validacao_cruzada.csv", "Validação cruzada por blocos (generalista)")

    print("\nEspecialista (um participante por vez)")
    linhas = []
    for u in USERS:
        _, p, _ = preparar("holdout", "personalized", u, HOLDOUT_FOLD)
        linhas.append({"Participante": u, "Treino": len(p.y_train), "Validação": len(p.y_val),
                       "Teste": len(p.y_test)})
    salvar_tabela(linhas, tab / "especialista.csv", "Especialista (hold-out, um modelo por participante)")

    # Figura: linha do tempo do hold-out e de um fold da CV
    fig, axes = fg.plt.subplots(len(USERS), 2, figsize=(11, 4.6), sharex="col")
    for i, u in enumerate(USERS):
        linha_do_tempo(prep, axes[i, 0], u, "Hold-out por blocos" if i == 0 else None)
        linha_do_tempo(preps_cv[2], axes[i, 1], u, "Validação cruzada: fold 3 de 6" if i == 0 else None)
        axes[i, 1].set_ylabel("")
    for ax in axes[-1]:
        ax.set_xlabel("Registro (ordem de gravação)")
    handles = [Patch(color=c, label=PARTES[p]) for p, c in COR.items()]
    handles.append(Patch(facecolor="white", edgecolor=fg.AXIS, label="purge (descartado)"))
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Divisão dos registros de cada participante", x=0.02, ha="left", fontsize=10, fontweight="semibold")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fg.salvar(fig, SAIDA / "figuras" / "01_divisao_blocos.png")
    # Dados para o dashboard: o hold-out e todos os folds da CV (a figura mostra só o fold 3)
    divisoes = [{"nome": "Hold-out", "prep": prep}] + [{"nome": f"CV: fold {f + 1} de {N_FOLDS}", "prep": preps_cv[f]}
                                                         for f in range(N_FOLDS)]
    fg.salvar_dados(SAIDA / "figuras" / "01_divisao_blocos.png", {
        "partes": PARTES, "participantes": USERS,
        "divisoes": [{"nome": d["nome"], "trechos": {u: trechos(d["prep"], u) for u in USERS}} for d in divisoes]})
    print(f"\nTabelas em {tab} e figura em {SAIDA / 'figuras'}")
