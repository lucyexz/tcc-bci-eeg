"""Configuração final do trabalho: dados, protocolos de avaliação e modelos.

Tudo que define os resultados está aqui, com valores fixos.

Paradigmas
    generalista   um modelo para os 5 participantes juntos. Features log10 e
                  normalização por participante (StandardScaler ajustado no
                  treino de cada um).
    especialista  um modelo por participante. Features log10 e normalização
                  global (dentro de um só participante as duas são iguais).

Protocolos (sempre por blocos: um bloco é um trecho contínuo da mesma classe)
    holdout            os 36 blocos de cada participante em 6 grupos de 6;
                       teste = grupo 6 (o fim da sessão), validação = grupo 5,
                       treino = grupos 1–4. Sementes 42, 43 e 44.
    cv                 validação cruzada por blocos: cada grupo é teste uma
                       vez (6 folds), com o grupo seguinte como validação.
    holdout_permutado  controle: o hold-out com os rótulos embaralhados entre
                       blocos no treino e na validação (teste intacto),
                       semente 42. Sem vazamento, a acurácia deve ficar perto
                       do acaso.
Em todos, 40 registros de purge separam as partições, e o teste usa só os
registros com pelo menos 63 anteriores no mesmo bloco (teste comum: todos os
modelos e janelas são avaliados nos mesmos registros).

Modelos: entrada = janela causal dos T registros mais recentes. T e dropout
foram escolhidos pela acurácia de validação do hold-out (nunca pelo teste).
"""

from pathlib import Path

from src.dados.preprocessamento import BasePreprocessingConfig

RAIZ = Path(__file__).resolve().parents[1]
RESULTADOS = RAIZ / "resultados"   # tudo que as etapas geram (fora do git)

USERS = ["user_a", "user_b", "user_c", "user_d", "user_e"]
TRACKS = ["pooled", "personalized"]
TRACK_DIR = {"pooled": "generalista", "personalized": "especialista"}
TRACK_NOME = {"pooled": "Generalista", "personalized": "Especialista"}

PROTOCOLOS = ["holdout", "cv", "holdout_permutado"]
SEMENTES = [42, 43, 44]
SEMENTES_PERMUTACAO = [42]
N_FOLDS = 6
HOLDOUT_FOLD = 5          # o último grupo de blocos de cada participante
PURGE = 40
EVAL_HISTORY = 64         # teste comum: registros com >= 63 anteriores no bloco
THREADS = 4               # threads do TensorFlow (o treino é determinístico)

NORMALIZACAO = {"pooled": "per_subject", "personalized": "global"}

# Configuração final de cada modelo e paradigma (o resto vem da classe *Config).
CONFIG_FINAL = {
    "cnn": {"pooled": {"window": 64}, "personalized": {"window": 32}},
    "bilstm": {"pooled": {"window": 32}, "personalized": {"window": 32}},
    "bigru": {"pooled": {"window": 32}, "personalized": {"window": 32}},
    "transformer": {"pooled": {"window": 64, "dropout_rate": 0.1},
                    "personalized": {"window": 32, "dropout_rate": 0.2}},
    "cnn_transformer": {"pooled": {"window": 64, "dropout_rate": 0.1},
                        "personalized": {"window": 16, "dropout_rate": 0.1}},
}
MODELOS = list(CONFIG_FINAL)


def preprocessamento(protocolo: str, track: str, subject: str | None, fold: int) -> BasePreprocessingConfig:
    """Configuração do pré-processamento de uma unidade de treino.

    No hold-out o teste é o grupo HOLDOUT_FOLD e a validação o grupo anterior;
    na CV a validação é o grupo seguinte (circular).
    """
    holdout = protocolo in ("holdout", "holdout_permutado")
    return BasePreprocessingConfig(
        subject=subject,
        normalization=NORMALIZACAO[track],
        cv_n_folds=N_FOLDS,
        cv_fold=HOLDOUT_FOLD if holdout else fold,
        cv_val_position="previous" if holdout else "next",
        purge_samples=PURGE,
    )


def sementes_de(protocolo: str) -> list[int]:
    return SEMENTES_PERMUTACAO if protocolo == "holdout_permutado" else SEMENTES


def unidades(protocolo: str, tracks=TRACKS, users=USERS, sementes=None, folds=range(N_FOLDS)):
    """(track, sujeito, fold, sementes) de um protocolo, na ordem de execução."""
    sementes = sementes_de(protocolo) if sementes is None else sementes
    for track in tracks:
        for subject in ([None] if track == "pooled" else users):
            if protocolo == "cv":
                for fold in folds:
                    yield track, subject, fold, [42]
            else:
                yield track, subject, HOLDOUT_FOLD, list(sementes)


def subjects(track: str):
    return [None] if track == "pooled" else USERS


def folhas(protocolo: str) -> list[str]:
    """Nome das pastas de cada treino de uma unidade."""
    if protocolo == "cv":
        return [f"fold_{f}" for f in range(N_FOLDS)]
    return [f"semente_{s}" for s in sementes_de(protocolo)]
