"""Dados do simulador do dashboard (a versão no navegador). O Pygame não usa este módulo.

Entrega o mesmo que o ModelController calcula para o Pygame, mas sem recalcular as
predições: as probabilidades vêm de predicoes.npz, gravado no treino (hold-out por
blocos, semente 42), que são as mesmas que o modelo produz ao ser recarregado.

    proba      probabilidade das 3 classes em cada janela de teste do participante
    y          classe verdadeira
    blocos     bloco de cada janela (para o texto de progresso)
    registros  as 25 features do registro mais recente da janela (grade EEG)
    atencao    atenção do [CLS] por token de tempo (só nos dois Transformers)

A grade EEG vem do mesmo pré-processamento do treino (só pandas/numpy). O TensorFlow
só é carregado para a atenção dos Transformers.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from src.config import CONFIG_FINAL, HOLDOUT_FOLD, PURGE, RESULTADOS, TRACK_DIR, USERS

TRACK = {"generalista": "pooled", "especialista": "personalized"}
COM_ATENCAO = ("transformer", "cnn_transformer")
NOMES = {"cnn": "CNN", "bilstm": "Bi-LSTM", "bigru": "Bi-GRU", "transformer": "Transformer",
         "cnn_transformer": "CNN + Transformer"}
SEMENTE = 42


class SemResultados(Exception):
    """O modelo pedido ainda não foi treinado."""


def _pasta(parte: str, modelo: str, track: str, subject: str | None):
    pasta = RESULTADOS / modelo / parte / "holdout" / TRACK_DIR[track]
    return (pasta / subject if subject else pasta) / f"semente_{SEMENTE}"


def _arredondar(a: np.ndarray, casas: int = 4) -> list:
    return np.round(np.asarray(a, dtype=float), casas).tolist()


@lru_cache(maxsize=32)
def dados_simulacao(modelo: str, paradigma: str, participante: str) -> dict:
    if modelo not in CONFIG_FINAL or paradigma not in TRACK or participante not in USERS:
        raise ValueError(f"Opção inválida: {modelo}, {paradigma}, {participante}")
    track = TRACK[paradigma]
    subject = participante if track == "personalized" else None

    arquivo = _pasta("treinos", modelo, track, subject) / "predicoes.npz"
    if not arquivo.exists():
        raise SemResultados(f"Predições não encontradas: {arquivo}. Treine antes: python main.py {modelo}")
    npz = np.load(arquivo)
    sel_npz = npz["users"] == participante
    proba, y, blocos = npz["y_pred_proba"][sel_npz], npz["y_true"][sel_npz], npz["blocks"][sel_npz]

    # Mesmo pré-processamento e mesmas janelas do treino: o teste sai na mesma ordem do .npz
    from src.dados.preparacao import janelas, preparar

    T = CONFIG_FINAL[modelo][track]["window"]
    _, prep, _ = preparar("holdout", track, subject, HOLDOUT_FOLD, verbose=False)
    w = janelas(prep, T, PURGE, verbose=False)
    sel = w.users_test == participante
    if not (np.array_equal(w.y_test[sel], y) and np.array_equal(w.blocks_test[sel], blocos)):
        raise RuntimeError("O teste reconstruído não bate com o predicoes.npz gravado (ordem ou rótulos).")
    X = w.X_test[sel]

    atencao = None
    if modelo in COM_ATENCAO:
        caminho = _pasta("modelos", modelo, track, subject) / "modelo.keras"
        if not caminho.exists():
            raise SemResultados(f"Modelo não encontrado: {caminho}. Treine antes: python main.py {modelo}")
        # Imports tardios: o TensorFlow só é inicializado aqui
        from src.avaliacao.explicabilidade import cls_attention_por_amostra, load_model
        from src.modelos import MODELOS
        from src.modelos.treino import configurar_tensorflow

        configurar_tensorflow()
        reshape = MODELOS[modelo][2]
        atencao = _arredondar(cls_attention_por_amostra(load_model(caminho), reshape(X)))

    return {
        "modelo": modelo,
        "nome": NOMES[modelo],
        "paradigma": paradigma,
        "participante": participante,
        "semente": SEMENTE,
        "T": int(T),
        "n": int(len(y)),
        "acerto_total": float((proba.argmax(axis=1) == y).mean()),
        "proba": _arredondar(proba),
        "y": y.astype(int).tolist(),
        "blocos": blocos.astype(int).tolist(),
        "registros": _arredondar(X[:, -1, :], 3),
        "atencao": atencao,
    }
