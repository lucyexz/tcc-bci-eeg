"""Treino e análise de uma das cinco redes (python main.py cnn, bilstm, bigru, transformer, cnn_transformer).

Sem opções: treina o que ainda não existe (hold-out com 3 sementes, validação
cruzada com 6 folds e controle de permutação) e depois analisa: tabelas (.csv),
figuras e explicabilidade em resultados/<modelo>/. Treinos já gravados são
pulados, então dá para interromper e continuar depois.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from src.config import CONFIG_FINAL, PROTOCOLOS, RESULTADOS, SEMENTES, TRACK_DIR, TRACK_NOME, TRACKS, USERS
from src.modelos import treino as tr


def main(modelo: str, argv: list[str] | None = None) -> None:
    from src.modelos import NOMES
    descricao = sys.modules[f"src.modelos.{'recorrentes' if modelo in ('bilstm', 'bigru') else modelo}"].__doc__
    parser = argparse.ArgumentParser(prog=f"python main.py {modelo}", description=descricao,
                                     epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--so-analisar", action="store_true", help="não treina; só analisa o que já existe")
    parser.add_argument("--estimar", action="store_true", help="só mostra o tempo previsto dos treinos que faltam")
    parser.add_argument("--protocolos", nargs="+", choices=PROTOCOLOS, default=PROTOCOLOS,
                        help="protocolos a treinar (padrão: os três)")
    parser.add_argument("--smoke", action="store_true", help="teste rápido do código: 2 épocas, em resultados/_smoke/")
    parser.add_argument("--saida", type=Path, help="grava em SAIDA em vez de resultados/<modelo>")
    parser.add_argument("--paradigmas", nargs="+", choices=["generalista", "especialista"],
                        help="restringe o treino (testes)")
    parser.add_argument("--participantes", nargs="+", choices=USERS, help="restringe o treino (testes)")
    parser.add_argument("--sementes", nargs="+", type=int, choices=SEMENTES, help="restringe o treino (testes)")
    parser.add_argument("--sem-xai", action="store_true", help="pula a explicabilidade na análise")
    args = parser.parse_args(argv)

    tr.configurar_tensorflow()
    raiz = args.saida or tr.raiz_modelo(modelo, smoke=args.smoke)
    tracks = [t for t in TRACKS if TRACK_DIR[t] in args.paradigmas] if args.paradigmas else None
    print(f"\n{'=' * 78}\n{NOMES[modelo]} — saída em {raiz}\n{'=' * 78}")
    for track in TRACKS:
        print(f"  {TRACK_NOME[track]:<13} " + ", ".join(f"{k} = {v}" for k, v in CONFIG_FINAL[modelo][track].items()))

    if not args.so_analisar:
        print("\nTempo previsto:")
        tr.estimar(modelo, args.protocolos, raiz)
        if args.estimar:
            return
        inicio = time.perf_counter()
        for protocolo in args.protocolos:
            tr.treinar_protocolo(modelo, protocolo, raiz, epochs=2 if args.smoke else None,
                                 tracks=tracks, users=args.participantes, sementes=args.sementes)
        print(f"\nTreinos concluídos em {tr.formatar_duracao(time.perf_counter() - inicio)}")
    elif args.estimar:
        tr.estimar(modelo, args.protocolos, raiz)
        return

    analisar(modelo, raiz, xai=not args.sem_xai)


# ---------------------------------------------------------------------------
# Análise
# ---------------------------------------------------------------------------

def carregar_baseline() -> dict | None:
    path = RESULTADOS / "baseline" / "baseline.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def analisar(modelo: str, raiz: Path, xai: bool = True) -> None:
    from src.avaliacao import figuras as fg
    from src.avaliacao.resultados import (CENARIOS, calibracao_de, carregar, colunas_media_dp, esperado,
                                          gap_treino_teste, linhas_por_classe, matriz_confusao, media_dp, pct,
                                          por_participante, salvar_tabela)
    from src.modelos import NOMES

    nome = NOMES[modelo]
    tab = raiz / "tabelas"
    print(f"\nAnálise — {nome}")
    runs = {(p, t): carregar(raiz, p, t) for p in PROTOCOLOS for t in TRACKS}
    faltando = [f"{p}/{TRACK_DIR[t]} ({len(r)}/{esperado(p, t)})" for (p, t), r in runs.items()
                if len(r) < esperado(p, t)]
    if not runs[("holdout", "pooled")] and not runs[("holdout", "personalized")]:
        print(f"  Nenhum treino de hold-out encontrado; rode python main.py {modelo}.")
        return
    if faltando:
        print("  [aviso] treinos incompletos: " + "; ".join(faltando))

    base = carregar_baseline()
    clf = "Regressão logística"
    final = {(p, t): por_participante(runs[(p, t)]) for p in ("holdout", "cv") for t in TRACKS}
    perm = {t: por_participante(runs[("holdout_permutado", t)]) for t in TRACKS}
    base_final = None
    if base:
        base_final = {(p, t): base["resultados"][clf][t]["teste"][p] for p in ("holdout", "cv") for t in TRACKS}
    else:
        print("  [aviso] baseline ainda não rodado (python main.py baseline): a tabela sai sem ele.")

    linhas = []
    for track in TRACKS:
        cfg = tr.config_modelo(modelo, track, 42)
        linhas.append({"Paradigma": TRACK_NOME[track], "T (registros)": cfg.window,
                       "Duração da janela (s)": cfg.window / 8, "Dropout": cfg.dropout_rate,
                       "Parâmetros": tr.contar_parametros(modelo, track)})
    salvar_tabela(linhas, tab / "configuracao.csv", f"{nome}: configuração")

    linhas = [{"Modelo": nome, **colunas_media_dp(final)}]
    if base_final:
        linhas.append({"Modelo": f"{clf} (baseline)", **colunas_media_dp(base_final)})
    salvar_tabela(linhas, tab / "resultado.csv",
                  f"{nome}: acurácia balanceada no teste comum, média e DP entre os 5 participantes (acaso = 33,3%)")

    salvar_tabela([{"Participante": u, **{f"{rot} (%)": pct(final[(p, t)].get(u)) for p, t, rot in CENARIOS}}
                   for u in USERS], tab / "por_participante.csv", f"{nome}: por participante")

    salvar_tabela(linhas_por_classe(runs), tab / "metricas_por_classe.csv",
                  f"{nome}: precisão, sensibilidade e F1 por classe no teste comum, média e DP entre os 5 participantes")

    linhas = []
    for t in TRACKS:
        todos = {**{("h",) + k: r for k, r in runs[("holdout", t)].items()},
                 **{("c",) + k: r for k, r in runs[("cv", t)].items()}}
        if not todos:
            continue
        linhas.append({
            "Paradigma": TRACK_NOME[t],
            "Validação hold-out (%)": pct(media_dp(por_participante(runs[("holdout", t)],
                                                                    lambda r: r.val_scores()))[0]),
            "Melhor época (mediana)": float(np.median([r.meta["best_epoch"] for r in todos.values()])),
            "Épocas treinadas (mediana)": float(np.median([r.meta["epochs_trained"] for r in todos.values()])),
            "Treino − teste (pp)": pct(gap_treino_teste({k[1:]: r for k, r in todos.items()})),
        })
    salvar_tabela(linhas, tab / "treino.csv", f"{nome}: validação, épocas e diferença treino − teste")

    linhas = []
    for t in TRACKS:
        if perm[t]:
            m, sd = media_dp(perm[t])
            linhas.append({"Paradigma": TRACK_NOME[t],
                           "Rótulos verdadeiros (%)": pct(media_dp(final[("holdout", t)])[0]),
                           "Rótulos permutados (%)": pct(m), "Permutados DP (pp)": pct(sd),
                           **{f"{u} permutados (%)": pct(v) for u, v in perm[t].items()}})
    salvar_tabela(linhas, tab / "permutacao.csv",
                  f"{nome}: controle de permutação de rótulos (hold-out, semente 42; sem vazamento, perto do acaso)")

    curvas, linhas = {}, []
    for t in TRACKS:
        if runs[("cv", t)]:
            curvas[t] = calibracao_de(runs[("cv", t)])
            linhas.append({"Paradigma": TRACK_NOME[t], "Confiança média (%)": pct(curvas[t]["mean_confidence"]),
                           "Acurácia (%)": pct(curvas[t]["accuracy_overall"]), "ECE": round(curvas[t]["ece"], 3)})
    salvar_tabela(linhas, tab / "calibracao.csv", f"{nome}: calibração (validação cruzada, rodadas concatenadas)")

    # Figuras
    fig_dir = raiz / "figuras"
    completo = all(runs[(p, t)] for p in ("holdout", "cv") for t in TRACKS)
    if completo:
        fg.fig_resultado_modelo(nome, final, None, fig_dir / "01_resultado.png")  # baseline só na tabela
        fg.fig_matriz_confusao(nome, {t: matriz_confusao(runs[("holdout", t)]) for t in TRACKS},
                               fig_dir / "02_matriz_confusao.png")
        fg.fig_curvas(nome, {t: list(runs[("holdout", t)].values()) + list(runs[("cv", t)].values())
                             for t in TRACKS}, fig_dir / "03_curvas_treino.png")
        if all(perm[t] for t in TRACKS):
            fg.fig_permutacao_modelo(nome, {t: final[("holdout", t)] for t in TRACKS}, perm,
                                     fig_dir / "04_permutacao.png")
        fg.fig_calibracao_modelo(nome, curvas, fig_dir / "05_calibracao.png")

    if xai and completo:
        explicabilidade(modelo, raiz, fig_dir)
    print(f"\nTabelas em {tab} e figuras em {fig_dir}")


def _janela_de_entrada(modelo: str, ig: np.ndarray) -> np.ndarray:
    """IG no formato do modelo → (N, T, 25)."""
    if modelo == "cnn":                       # (N, 5, 5, T)
        n, _, _, window = ig.shape
        return ig.transpose(0, 3, 1, 2).reshape(n, window, 25)
    if modelo == "cnn_transformer":           # (N, 25, T, 1)
        return ig[..., 0].transpose(0, 2, 1)
    return ig                                 # (N, T, 25)


def explicabilidade(modelo: str, raiz: Path, fig_dir: Path,
                    por_classe: int = 400, por_classe_atencao: int = 1000) -> None:
    """IG (eletrodo × banda e posição na janela) e atenção do [CLS], nos modelos da semente 42 do hold-out."""
    from tensorflow import keras

    from src.avaliacao import figuras as fg
    from src.avaliacao.explicabilidade import (amostra_estratificada, cls_attention_over_tokens, load_model,
                                               window_integrated_gradients)
    from src.avaliacao.resultados import salvar_tabela
    from src.config import HOLDOUT_FOLD, PURGE, subjects
    from src.constantes import EEG_BANDS, EEG_CHANNELS
    from src.dados.preparacao import janelas, preparar
    from src.modelos import MODELOS, NOMES
    from src.modelos.cnn_transformer import token_offsets_s
    from src.modelos.treino import pasta_treino

    nome = NOMES[modelo]
    reshape = MODELOS[modelo][2]
    com_atencao = modelo in ("transformer", "cnn_transformer")
    print(f"\nExplicabilidade — {nome} (modelos da semente 42 do hold-out)")
    grades, tempos, atencao, offsets = {}, {}, {}, {}
    for track in TRACKS:
        cfg = tr.config_modelo(modelo, track, 42)
        g_list, t_list, a_list = [], [], []
        for subject in subjects(track):
            path = pasta_treino(raiz, "modelos", "holdout", track, subject, "semente_42") / "modelo.keras"
            if not path.exists():
                continue
            _, prep, _ = preparar("holdout", track, subject, HOLDOUT_FOLD, verbose=False)
            w = janelas(prep, cfg.window, PURGE, verbose=False)
            X_in, y = reshape(w.X_test), w.y_test
            model = load_model(path)
            idx = amostra_estratificada(y, por_classe)
            ig = _janela_de_entrada(modelo, window_integrated_gradients(model, X_in[idx], y[idx]))
            yi = y[idx]
            g_list.append(np.stack([ig[yi == k].sum(axis=1).mean(axis=0).reshape(5, 5) for k in range(3)]))
            t_list.append(np.stack([np.abs(ig[yi == k]).sum(axis=2).mean(axis=0) for k in range(3)]))
            if com_atencao:
                ia = amostra_estratificada(y, por_classe_atencao)
                a_list.append(cls_attention_over_tokens(model, X_in[ia], y[ia])[-1])
            keras.backend.clear_session()
        if not g_list:
            continue
        grades[track], tempos[track] = np.mean(g_list, axis=0), np.mean(t_list, axis=0)
        if com_atencao:
            atencao[track] = np.mean(a_list, axis=0)
            offsets[track] = (token_offsets_s(cfg) if modelo == "cnn_transformer"
                              else (np.arange(cfg.window) - (cfg.window - 1)) / 8)
        print(f"  {TRACK_NOME[track]}: {len(g_list)} modelo(s)")

    if len(grades) < len(TRACKS):
        return
    fg.fig_xai_grades(nome, grades, fig_dir / "06_xai_eletrodo_banda.png")
    fg.fig_xai_tempo(nome, tempos, fig_dir / "07_xai_tempo.png")
    if com_atencao:
        fg.fig_atencao(nome, atencao, offsets, fig_dir / "08_atencao.png")

    def top3(grid):
        order = np.argsort(np.abs(grid).ravel())[::-1][:3]
        return [f"{EEG_CHANNELS[i // 5]}_{EEG_BANDS[i % 5]}" for i in order]

    linhas = []
    for t in TRACKS:
        for k, classe in enumerate(["Esquerda", "Direita", "Neutro"]):
            maiores = top3(grades[t][k])
            linhas.append({"Paradigma": TRACK_NOME[t], "Classe": classe,
                           "1º eletrodo × banda": maiores[0], "2º": maiores[1], "3º": maiores[2],
                           "IG no registro mais recente (%)": round(100 * float(tempos[t][k][-1] / tempos[t][k].sum()), 1),
                           "Uniforme (%)": round(100 / len(tempos[t][k]), 1)})
    tab = raiz / "tabelas"
    salvar_tabela(linhas, tab / "explicabilidade.csv",
                  f"{nome}: Integrated Gradients ({por_classe} janelas de teste por classe; "
                  "no especialista, média dos 5 modelos)")
    if com_atencao:
        salvar_tabela([{"Paradigma": TRACK_NOME[t],
                        "Atenção do [CLS] no token mais recente (%)": round(100 * float(atencao[t][:, -1].mean()), 1),
                        "Uniforme (%)": round(100 / atencao[t].shape[1], 1)} for t in TRACKS],
                      tab / "atencao.csv", f"{nome}: atenção do [CLS]")
