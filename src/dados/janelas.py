"""Janelas temporais causais sobre o resultado de `prepare_base_data`.

Cada amostra passa a ser a sequência dos T registros mais recentes de um
participante, terminando no registro que se quer classificar:

    janela(t) = registros [t − T + 1, …, t]      rótulo = rótulo de t

O janelamento acontece DEPOIS do pré-processamento, então imputação e scaler
continuam ajustados por registro e só no treino. Ele também acontece dentro
de cada partição (treino, validação e teste separadamente), de modo que
nenhuma janela mistura partições.

Uma janela só existe quando os T registros são do mesmo participante, do
mesmo bloco (trial) e têm posições consecutivas:

- **bloco:** o ciclo de classes é fixo (E → D → N). Uma janela que pegasse o
  fim do bloco anterior entregaria a classe atual de graça. Por isso os
  T − 1 primeiros registros de cada bloco ficam sem janela;
- **posições consecutivas:** o purge da CV por blocos e do hold-out remove
  registros perto das fronteiras; uma janela nunca salta esse buraco.

No teste, `eval_history` fixa um subconjunto comum: só ficam os registros com
pelo menos `eval_history − 1` registros anteriores no mesmo bloco. Com
`eval_history = 64`, todos os modelos e todos os T (até 64) são avaliados
exatamente nos mesmos registros.
"""

from dataclasses import dataclass

import numpy as np


def window_end_indices(
    users: np.ndarray,
    blocks: np.ndarray,
    positions: np.ndarray,
    window: int,
) -> np.ndarray:
    """Índices i em que a janela [i − window + 1, i] é válida.

    Pressupõe a ordem produzida por `prepare_base_data`: dentro de cada
    partição, as linhas de um participante aparecem juntas e em ordem
    crescente de posição. Assim, posições estritamente crescentes com
    diferença total de `window − 1` implicam registros consecutivos.
    """
    n = len(users)
    if window < 1:
        raise ValueError(f"`window` deve ser >= 1, recebido: {window}")
    if n < window:
        return np.array([], dtype=int)

    end = np.arange(window - 1, n)
    start = end - (window - 1)
    valid = (
        (users[start] == users[end])
        & (blocks[start] == blocks[end])
        & (positions[end] - positions[start] == window - 1)
    )
    return end[valid]


def make_windows(X: np.ndarray, end_idx: np.ndarray, window: int) -> np.ndarray:
    """(N, F) → (len(end_idx), window, F) em float32, do mais antigo ao atual."""
    offsets = np.arange(-(window - 1), 1)
    return X[end_idx[:, None] + offsets].astype(np.float32)


@dataclass
class WindowedData:
    """Mesma interface do `BasePreprocessingResult`, com janelas no lugar de registros.

    `X_*` têm forma (N, T, 25); y/users/blocks/positions são os do último
    registro de cada janela. `source_*` guarda o índice desse registro no
    array por registro correspondente do `prep` original (auditoria).
    """

    window: int
    X_train: np.ndarray
    y_train: np.ndarray
    users_train: np.ndarray
    blocks_train: np.ndarray
    positions_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    users_test: np.ndarray
    blocks_test: np.ndarray
    positions_test: np.ndarray
    feature_columns: list[str]
    preprocessing_report: dict
    source_train: np.ndarray
    source_test: np.ndarray
    X_val: np.ndarray | None = None
    y_val: np.ndarray | None = None
    users_val: np.ndarray | None = None
    blocks_val: np.ndarray | None = None
    positions_val: np.ndarray | None = None
    source_val: np.ndarray | None = None


def window_prep(prep, window: int, eval_history: int | None = None) -> WindowedData:
    """Janela treino, validação e teste de um `BasePreprocessingResult`.

    Args:
        window: T, o número de registros de cada janela.
        eval_history: se informado, o teste mantém só os registros em que uma
            janela de `eval_history` registros também seria válida (teste
            comum entre os T). Precisa ser >= `window`. None usa `window`.
    """
    if eval_history is None:
        eval_history = window
    if eval_history < window:
        raise ValueError(
            f"`eval_history` ({eval_history}) não pode ser menor que a janela ({window})"
        )

    def split(name: str, history: int):
        X = getattr(prep, f"X_{name}")
        if X is None:
            return None
        users = getattr(prep, f"users_{name}")
        blocks = getattr(prep, f"blocks_{name}")
        positions = getattr(prep, f"positions_{name}")
        end_idx = window_end_indices(users, blocks, positions, history)
        return {
            "X": make_windows(X, end_idx, window),
            "y": getattr(prep, f"y_{name}")[end_idx],
            "users": users[end_idx],
            "blocks": blocks[end_idx],
            "positions": positions[end_idx],
            "source": end_idx,
            "n_records": len(X),
        }

    train = split("train", window)
    val = split("val", window)
    test = split("test", eval_history)

    report = dict(prep.preprocessing_report)
    report["window_T"] = window
    report["window_eval_history"] = eval_history
    for name, part in (("train", train), ("val", val), ("test", test)):
        if part is None:
            continue
        report[f"n_windows_{name}"] = int(len(part["y"]))
        report[f"dropped_{name}"] = int(part["n_records"] - len(part["y"]))
        report[f"class_distribution_{name}_windows"] = {
            int(c): int((part["y"] == c).sum()) for c in sorted(np.unique(part["y"]))
        }

    kwargs = {}
    if val is not None:
        kwargs = {
            "X_val": val["X"], "y_val": val["y"], "users_val": val["users"],
            "blocks_val": val["blocks"], "positions_val": val["positions"],
            "source_val": val["source"],
        }

    return WindowedData(
        window=window,
        X_train=train["X"], y_train=train["y"], users_train=train["users"],
        blocks_train=train["blocks"], positions_train=train["positions"],
        X_test=test["X"], y_test=test["y"], users_test=test["users"],
        blocks_test=test["blocks"], positions_test=test["positions"],
        feature_columns=prep.feature_columns,
        preprocessing_report=report,
        source_train=train["source"], source_test=test["source"],
        **kwargs,
    )


def permute_block_labels(prep, seed: int) -> dict:
    """Controle de permutação: embaralha os rótulos entre os blocos (no lugar).

    Para cada participante, os rótulos dos blocos de treino e validação são
    permutados entre si (um bloco inteiro recebe o rótulo de outro bloco), e
    o teste fica com os rótulos verdadeiros. A estrutura temporal e a
    proporção de classes são preservadas; a única coisa destruída é a relação
    real entre sinal e classe. Um modelo honesto deve cair para ~33% no teste.

    Returns:
        Mapa participante → {bloco: (rótulo original, rótulo novo)}.
    """
    rng = np.random.default_rng(seed)
    mapping: dict[str, dict[int, tuple[int, int]]] = {}
    # Os rótulos do prep podem ser views somente leitura: troca por cópias.
    prep.y_train = np.array(prep.y_train, copy=True)
    parts = [("train", prep.y_train, prep.users_train, prep.blocks_train)]
    if prep.y_val is not None:
        prep.y_val = np.array(prep.y_val, copy=True)
        parts.append(("val", prep.y_val, prep.users_val, prep.blocks_val))

    users = np.unique(np.concatenate([p[2] for p in parts]))
    for user in users:
        block_label: dict[int, int] = {}
        for _, y, u, b in parts:
            mask = u == user
            for block in np.unique(b[mask]):
                labels = np.unique(y[mask & (b == block)])
                assert len(labels) == 1, f"Bloco {block} de {user} com mais de um rótulo"
                block_label[int(block)] = int(labels[0])

        blocks = sorted(block_label)
        shuffled = rng.permutation([block_label[b] for b in blocks])
        new_label = dict(zip(blocks, shuffled.tolist()))
        mapping[str(user)] = {b: (block_label[b], new_label[b]) for b in blocks}

        for _, y, u, b in parts:
            mask = u == user
            for block in np.unique(b[mask]):
                y[mask & (b == block)] = new_label[int(block)]

    return mapping
