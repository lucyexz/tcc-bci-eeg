"""Comparação dos cinco modelos e do baseline: tabelas e figuras gerais.

Lê os treinos de resultados/<modelo>/ e o baseline; nada é treinado. Se faltar
algum resultado, diz qual etapa rodar.

    tabelas/resultado_principal.csv   acurácia balanceada por modelo, paradigma e protocolo (+ baseline)
    tabelas/configuracao.csv          T e dropout finais de cada modelo
    tabelas/metricas_por_classe.csv   precisão, sensibilidade e F1 por classe (+ macro), só os cinco modelos
    tabelas/faixa_por_participante.csv menor e maior acurácia entre os 5 modelos, por participante
    tabelas/treino.csv                melhor época e diferença treino − teste
    tabelas/calibracao.csv            confiança, acurácia e ECE (validação cruzada)
    tabelas/permutacao.csv            controle de permutação de rótulos (hold-out)
    figuras/01_resultado_principal.png ... 05_calibracao.png  (só os cinco modelos, sem o baseline)

Saída: resultados/comparativo/
"""

import json
import statistics
import sys

import numpy as np

from src.avaliacao import figuras as fg
from src.avaliacao.resultados import (CENARIOS, calibracao_de, carregar, colunas_media_dp, esperado,
                                      gap_treino_teste, linhas_por_classe, media_dp, pct, por_participante,
                                      salvar_tabela)
from src.config import MODELOS, PROTOCOLOS, RESULTADOS, TRACK_NOME, TRACKS, USERS
from src.modelos import NOMES

SAIDA = RESULTADOS / "comparativo"
RL = "Regressão logística"
PROTOS = ("holdout", "cv")


def main(argv: list[str] | None = None) -> None:
    # Confere o que existe
    faltando = []
    runs = {}
    for modelo in MODELOS:
        for p in PROTOCOLOS:
            for t in TRACKS:
                runs[(modelo, p, t)] = carregar(RESULTADOS / modelo, p, t)
                if len(runs[(modelo, p, t)]) < esperado(p, t):
                    faltando.append(modelo)
    base_path = RESULTADOS / "baseline" / "baseline.json"
    if faltando or not base_path.exists():
        print("Resultados incompletos. Rode antes:")
        for modelo in dict.fromkeys(faltando):
            print(f"  python main.py {modelo}")
        if not base_path.exists():
            print("  python main.py baseline")
        sys.exit(1)
    base = json.loads(base_path.read_text(encoding="utf-8"))

    final = {(m, p, t): por_participante(runs[(m, p, t)]) for m in MODELOS for p in PROTOS for t in TRACKS}
    for clf in base["classificadores"]:
        for p in PROTOS:
            for t in TRACKS:
                final[(clf, p, t)] = base["resultados"][clf][t]["teste"][p]
    perm = {(m, t): por_participante(runs[(m, "holdout_permutado", t)]) for m in MODELOS for t in TRACKS}
    for t in TRACKS:
        perm[(RL, t)] = base["permutacao_holdout"][RL][t]

    def rotulo(linha: str) -> str:
        return NOMES.get(linha, f"{linha} (baseline)")

    tab = SAIDA / "tabelas"
    salvar_tabela([{"Modelo": rotulo(linha), **colunas_media_dp(final, lambda p, t, l=linha: (l, p, t))}
                   for linha in [*MODELOS, *base["classificadores"]]],
                  tab / "resultado_principal.csv",
                  "Acurácia balanceada no teste comum, média e DP entre os 5 participantes (primeiro a média das "
                  "sementes ou folds de cada um). Acaso = 33,3%")

    salvar_tabela([linha for m in MODELOS
                   for linha in linhas_por_classe({(p, t): runs[(m, p, t)] for p in PROTOS for t in TRACKS},
                                                  {"Modelo": NOMES[m]})],
                  tab / "metricas_por_classe.csv",
                  "Precisão, sensibilidade e F1 por classe no teste comum, média e DP entre os 5 participantes. "
                  "Macro = média das três classes (sensibilidade macro = acurácia balanceada)")

    linhas = []
    for m in MODELOS:
        cfg = {t: runs[(m, "holdout", t)][next(iter(runs[(m, "holdout", t)]))].meta["config"] for t in TRACKS}
        linhas.append({"Modelo": NOMES[m], "T generalista": cfg["pooled"]["window"],
                       "T especialista": cfg["personalized"]["window"],
                       "Dropout generalista": cfg["pooled"]["dropout_rate"],
                       "Dropout especialista": cfg["personalized"]["dropout_rate"]})
    salvar_tabela(linhas, tab / "configuracao.csv", "Configuração final")

    linhas = []
    for u in USERS:
        linha = {"Participante": u}
        for p, t, rot in CENARIOS:
            vals = [final[(m, p, t)][u] for m in MODELOS if u in final[(m, p, t)]]
            linha[f"{rot} mín. (%)"], linha[f"{rot} máx. (%)"] = pct(min(vals), 0), pct(max(vals), 0)
        linhas.append(linha)
    salvar_tabela(linhas, tab / "faixa_por_participante.csv", "Por participante: faixa entre os 5 modelos")

    linhas = []
    for m in MODELOS:
        for t in TRACKS:
            todos = {(p, *k): r for p in PROTOS for k, r in runs[(m, p, t)].items()}
            best = [r.meta["best_epoch"] for r in todos.values()]
            trained = [r.meta["epochs_trained"] for r in todos.values()]
            linhas.append({"Modelo": NOMES[m], "Paradigma": TRACK_NOME[t],
                           "Melhor época (mediana)": float(statistics.median(best)),
                           "Melhor época ≤ 2 (%)": pct(float(np.mean(np.array(best) <= 2)), 0),
                           "Épocas treinadas (mediana)": float(statistics.median(trained)),
                           "Treino − teste (pp)": pct(gap_treino_teste({k[1:]: r for k, r in todos.items()}))})
    salvar_tabela(linhas, tab / "treino.csv", "Melhor época e diferença treino − teste (hold-out + CV)")

    curvas, linhas = {}, []
    for m in MODELOS:
        for t in TRACKS:
            cal = calibracao_de(runs[(m, "cv", t)])
            if t == "personalized":
                curvas[m] = cal
            linhas.append({"Modelo": NOMES[m], "Paradigma": TRACK_NOME[t],
                           "Confiança média (%)": pct(cal["mean_confidence"]),
                           "Acurácia (%)": pct(cal["accuracy_overall"]), "ECE": round(cal["ece"], 3)})
    salvar_tabela(linhas, tab / "calibracao.csv", "Calibração (validação cruzada, rodadas concatenadas)")

    linhas = []
    for linha in [*MODELOS, RL]:
        for t in TRACKS:
            m, sd = media_dp(perm[(linha, t)])
            linhas.append({"Modelo": rotulo(linha), "Paradigma": TRACK_NOME[t], "Média (%)": pct(m),
                           "DP (pp)": pct(sd), **{f"{u} (%)": pct(v) for u, v in perm[(linha, t)].items()}})
    salvar_tabela(linhas, tab / "permutacao.csv",
                  "Permutação de rótulos (hold-out): rótulos embaralhados entre blocos no treino e na validação, "
                  "teste com os verdadeiros. Redes: semente 42; baseline: sementes 42–44. Sem vazamento, ~33% ou menos")

    # Figuras
    fig_dir = SAIDA / "figuras"
    linhas_fig = list(MODELOS)  # o baseline fica só nas tabelas
    rotulos = [NOMES[m] for m in linhas_fig]
    fg.fig_principal(linhas_fig, rotulos, final, fig_dir / "01_resultado_principal.png")
    fg.fig_por_participante(linhas_fig, rotulos, final, len(MODELOS), fig_dir / "02_por_participante.png")
    fg.fig_permutacao(linhas_fig, rotulos, {(m, t): final[(m, "holdout", t)] for m in linhas_fig for t in TRACKS},
                      perm, fig_dir / "03_permutacao.png")
    fg.fig_curvas_todos(MODELOS, NOMES, {(m, t): [r for p in PROTOS for r in runs[(m, p, t)].values()]
                                         for m in MODELOS for t in TRACKS}, fig_dir / "04_curvas_treino.png")
    fg.fig_calibracao_todos(MODELOS, NOMES, curvas, fig_dir / "05_calibracao.png")
    print(f"\nTabelas em {tab} e figuras em {fig_dir}")
