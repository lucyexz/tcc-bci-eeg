"""Baseline clássico: LDA e regressão logística sobre a média da janela.

Cada amostra vira a média, nos T registros da janela causal, das 25 features
(log, padronizadas exatamente como nas redes). Mesmos dados, partições, purge
e teste comum das redes; nenhuma rede neural.

    LDA                  solver lsqr, shrinkage automático (Ledoit-Wolf)
    Regressão logística  class_weight="balanced", C = 1, max_iter 2000

O T é escolhido entre {8, 16, 32, 64} pela maior acurácia balanceada média de
validação do hold-out, separadamente por classificador e paradigma; o teste do
hold-out e da validação cruzada usa esse T. Inclui o controle de permutação
no hold-out (sementes 42, 43 e 44). Leva ~1,5 min.

Saída: resultados/baseline/ (baseline.json, que os modelos e a comparação
leem; tabelas/*.csv; figuras/)
"""

import json
import time
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from src.avaliacao.resultados import CENARIOS, media_dp, pct, salvar_tabela
from src.config import (EVAL_HISTORY, HOLDOUT_FOLD, N_FOLDS, RESULTADOS, SEMENTES, TRACK_NOME, TRACKS, USERS,
                        subjects)
from src.dados.janelas import permute_block_labels, window_prep
from src.dados.preparacao import preparar
from src.modelos.treino import formatar_duracao

SAIDA = RESULTADOS / "baseline"
WINDOWS = [8, 16, 32, 64]
CLASSIFICADORES = {
    "LDA": lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
    "Regressão logística": lambda: LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000),
}


def scores_por_participante(y_true, y_pred, users, subject) -> dict[str, float]:
    out = {u: float(balanced_accuracy_score(y_true[users == u], y_pred[users == u])) for u in sorted(np.unique(users))}
    return out if subject is None else {subject: out[subject]}


def media_por_participante(entradas: list[dict], chave: str) -> dict[str, float]:
    acc = {}
    for e in entradas:
        for u, v in e[chave].items():
            acc.setdefault(u, []).append(v)
    return {u: float(np.mean(v)) for u, v in sorted(acc.items())}


def unidades():
    for protocolo in ("holdout", "cv"):
        for track in TRACKS:
            for subject in subjects(track):
                for fold in ([HOLDOUT_FOLD] if protocolo == "holdout" else range(N_FOLDS)):
                    yield protocolo, track, subject, fold


def main(argv: list[str] | None = None) -> None:
    inicio = time.perf_counter()
    print("Baseline clássico — LDA e regressão logística na média da janela")
    # bruto[clf][track][T][protocolo] = lista de {subject, fold, val, teste}
    bruto = {c: {t: {w: {"holdout": [], "cv": []} for w in WINDOWS} for t in TRACKS} for c in CLASSIFICADORES}
    extras = {}   # (clf, track, T, subject) → (y_true, y_pred, coef) do hold-out, para matriz e coeficientes
    for protocolo, track, subject, fold in unidades():
        _, prep, _ = preparar(protocolo, track, subject, fold, verbose=False)
        for window in WINDOWS:
            w = window_prep(prep, window, eval_history=EVAL_HISTORY)
            X_tr, X_va, X_te = w.X_train.mean(axis=1), w.X_val.mean(axis=1), w.X_test.mean(axis=1)
            for nome, make in CLASSIFICADORES.items():
                clf = make().fit(X_tr, w.y_train)
                y_te = clf.predict(X_te)
                bruto[nome][track][window][protocolo].append({
                    "subject": subject, "fold": fold,
                    "val": scores_por_participante(w.y_val, clf.predict(X_va), w.users_val, subject),
                    "teste": scores_por_participante(w.y_test, y_te, w.users_test, subject),
                })
                if protocolo == "holdout":
                    extras[(nome, track, window, subject)] = (w.y_test, y_te, getattr(clf, "coef_", None))
        print(f"  {protocolo} {TRACK_NOME[track]} {subject or ''} fold {fold}: ok "
              f"({formatar_duracao(time.perf_counter() - inicio)})", flush=True)

    resultados = {}
    for nome in CLASSIFICADORES:
        resultados[nome] = {}
        for track in TRACKS:
            por_T = bruto[nome][track]
            val = {w: media_dp(media_por_participante(por_T[w]["holdout"], "val"))[0] for w in WINDOWS}
            melhor = max(WINDOWS, key=lambda w: val[w])
            resultados[nome][track] = {
                "T": melhor, "validacao_por_T": val,
                "teste": {p: media_por_participante(por_T[melhor][p], "teste") for p in ("holdout", "cv")},
            }

    # Controle de permutação: rótulos embaralhados entre blocos no treino e na
    # validação, teste com os verdadeiros; T escolhido; sementes 42–44.
    permutacao = {}
    for nome, make in CLASSIFICADORES.items():
        permutacao[nome] = {}
        for track in TRACKS:
            T = resultados[nome][track]["T"]
            acc = {}
            for semente in SEMENTES:
                for subject in subjects(track):
                    _, prep, _ = preparar("holdout", track, subject, HOLDOUT_FOLD, verbose=False)
                    permute_block_labels(prep, seed=semente)
                    w = window_prep(prep, T, eval_history=EVAL_HISTORY)
                    clf = make().fit(w.X_train.mean(axis=1), w.y_train)
                    s = scores_por_participante(w.y_test, clf.predict(w.X_test.mean(axis=1)), w.users_test, subject)
                    for u, v in s.items():
                        acc.setdefault(u, []).append(v)
            permutacao[nome][track] = {u: float(np.mean(v)) for u, v in sorted(acc.items())}
    print(f"  permutação: ok ({formatar_duracao(time.perf_counter() - inicio)})")

    # Matriz de confusão e coeficientes da regressão logística (hold-out, T escolhido)
    rl = "Regressão logística"
    matrizes, coefs = {}, {}
    for track in TRACKS:
        T = resultados[rl][track]["T"]
        cm = np.zeros((3, 3))
        cf = []
        for subject in subjects(track):
            y_true, y_pred, coef = extras[(rl, track, T, subject)]
            cm += confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
            cf.append(coef.reshape(3, 5, 5))
        matrizes[track] = cm / cm.sum(axis=1, keepdims=True)
        coefs[track] = np.mean(cf, axis=0)

    SAIDA.mkdir(parents=True, exist_ok=True)
    (SAIDA / "baseline.json").write_text(json.dumps({
        "janelas": WINDOWS, "classificadores": list(CLASSIFICADORES), "eval_history": EVAL_HISTORY,
        "features": "média da janela das 25 features (log, padronizadas)",
        "resultados": resultados, "permutacao_holdout": permutacao,
        "matriz_confusao_holdout_rl": {t: matrizes[t].tolist() for t in TRACKS},
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    tab = SAIDA / "tabelas"
    linhas = []
    for nome in CLASSIFICADORES:
        for track in TRACKS:
            r = resultados[nome][track]
            linha = {"Classificador": nome, "Paradigma": TRACK_NOME[track], "T": r["T"],
                     "Validação hold-out (%)": pct(r["validacao_por_T"][r["T"]])}
            for p, rotulo in (("holdout", "Hold-out"), ("cv", "CV")):
                m, sd = media_dp(r["teste"][p])
                linha[f"{rotulo} (%)"], linha[f"{rotulo} DP (pp)"] = pct(m), pct(sd)
            m, sd = media_dp(permutacao[nome][track])
            linha["Permutação (%)"], linha["Permutação DP (pp)"] = pct(m), pct(sd)
            linhas.append(linha)
    salvar_tabela(linhas, tab / "resultado.csv",
                  "Baseline: acurácia balanceada no teste comum (o mesmo das redes), média e DP entre os 5 "
                  "participantes; permutação no hold-out com 3 sementes")

    salvar_tabela([{"Classificador": nome, "Paradigma": TRACK_NOME[track],
                    **{f"T = {w} (%)": pct(resultados[nome][track]["validacao_por_T"][w]) for w in WINDOWS}}
                   for nome in CLASSIFICADORES for track in TRACKS],
                  tab / "validacao_por_T.csv", "Baseline: validação do hold-out por T (escolha do T)")

    salvar_tabela([{"Participante": u, **{f"{rot} (%)": pct(resultados[rl][t]["teste"][p].get(u))
                                          for p, t, rot in CENARIOS}} for u in USERS],
                  tab / "por_participante.csv", "Baseline: por participante (regressão logística)")

    from src.avaliacao import figuras as fg
    fg.fig_matriz_confusao(rl, matrizes, SAIDA / "figuras" / "01_matriz_confusao.png")
    fig_coeficientes(coefs, SAIDA / "figuras" / "02_coeficientes.png")

    print(f"\nSalvo em {SAIDA} ({formatar_duracao(time.perf_counter() - inicio)})")


def fig_coeficientes(coefs: dict, path: Path) -> None:
    """Coeficientes da regressão logística por classe (eletrodo × banda), o equivalente ao XAI das redes."""
    from matplotlib.colors import TwoSlopeNorm

    from src.constantes import EEG_BANDS, EEG_CHANNELS
    from src.avaliacao import figuras as fg

    vmax = max(float(np.abs(c).max()) for c in coefs.values())
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    fig, axes = fg.plt.subplots(2, 3, figsize=(8.4, 5.6), layout="constrained")
    for r, track in enumerate(TRACKS):
        for c in range(3):
            ax = axes[r, c]
            im = ax.imshow(coefs[track][c], cmap=fg.DIVERGING, norm=norm)
            ax.set_xticks(range(5), EEG_BANDS, fontsize=7.5, rotation=30)
            ax.set_yticks(range(5), EEG_CHANNELS, fontsize=7.5)
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.tick_params(length=0)
            ax.set_title(f"{TRACK_NOME[track]} · {fg.CLASS_NAMES[c]}", fontsize=9)
    cbar = fig.colorbar(im, ax=axes, shrink=0.7)
    cbar.outline.set_visible(False)
    cbar.set_label("Coeficiente (features padronizadas)\nazul = a favor · marrom = contra", color=fg.INK2)
    fig.suptitle("Regressão logística: coeficientes por eletrodo × banda (hold-out)", x=0.02, ha="left",
                 fontsize=10, fontweight="semibold")
    fg.salvar(fig, path)
