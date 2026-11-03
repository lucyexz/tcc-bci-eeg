"""Dados de uma unidade de treino: pré-processamento, janelas e checagens anti-vazamento."""

import numpy as np

from src.config import EVAL_HISTORY, preprocessamento
from src.dados.preprocessamento import prepare_base_data
from src.dados.janelas import permute_block_labels, window_prep


def preparar(protocolo: str, track: str, subject: str | None, fold: int,
             semente_permutacao: int | None = None, verbose: bool = True):
    """Pré-processa uma unidade e confere as partições.

    Com `semente_permutacao`, embaralha os rótulos entre blocos no treino e na
    validação (controle de permutação); o teste fica com os rótulos verdadeiros.
    Devolve (config, prep, permutação ou None).
    """
    config = preprocessamento(protocolo, track, subject, fold)
    prep = prepare_base_data(config)
    checar_particoes(prep, config, verbose=verbose)
    permutacao = None
    if semente_permutacao is not None:
        permutacao = permute_block_labels(prep, seed=semente_permutacao)
        if verbose:
            trocados = sum(o != n for m in permutacao.values() for o, n in m.values())
            total = sum(len(m) for m in permutacao.values())
            print(f"  [permutação] rótulos trocados em {trocados}/{total} blocos (treino + validação)")
    return config, prep, permutacao


def janelas(prep, window: int, purge: int, verbose: bool = True):
    """Janelas causais de T registros (teste comum de EVAL_HISTORY) e checagem."""
    wprep = window_prep(prep, window, eval_history=EVAL_HISTORY)
    checar_janelas(wprep, prep, purge, verbose=verbose)
    return wprep


# ---------------------------------------------------------------------------
# Checagens (levantam AssertionError se alguma garantia for quebrada)
# ---------------------------------------------------------------------------

def checar_particoes(prep, config, verbose: bool = True) -> None:
    """Treino, validação e teste formam uma partição exata, sem sobreposição.

    Também confere: um só participante no especialista; blocos de teste fora do
    treino e purge > `purge_samples` registros; as 3 classes em cada partição;
    StandardScaler ajustado só no treino (de cada participante, no generalista).
    """
    partitions = {"treino": prep.users_train, "validação": prep.users_val, "teste": prep.users_test}

    if config.subject is not None:
        for name, users in partitions.items():
            found = sorted(np.unique(users).tolist())
            assert found == [config.subject], f"Outro participante na partição {name}: {found}"

    total = int(prep.preprocessing_report["shape_after_cleaning"][0])
    n = len(prep.y_train) + len(prep.y_val) + len(prep.y_test) + int(prep.preprocessing_report.get("purged_rows", 0))
    assert n == total, f"As partições somam {n} registros, o dataset tem {total}"

    def keys(users, positions):
        return set(zip(users.tolist(), positions.tolist()))

    train_k = keys(prep.users_train, prep.positions_train)
    val_k = keys(prep.users_val, prep.positions_val)
    test_k = keys(prep.users_test, prep.positions_test)
    assert not (train_k & test_k), "Registros em treino e teste"
    assert not (val_k & test_k), "Registros em validação e teste"
    assert not (train_k & val_k), "Registros em treino e validação"

    for user in np.unique(prep.users_test):
        test_pos = prep.positions_test[prep.users_test == user]
        fit_pos = np.concatenate([prep.positions_train[prep.users_train == user],
                                  prep.positions_val[prep.users_val == user]])
        gap = np.maximum(0, np.maximum(test_pos.min() - fit_pos, fit_pos - test_pos.max())).min()
        assert gap > config.purge_samples, f"Purge violado em {user}: distância {gap}"
        test_blocks = set(prep.blocks_test[prep.users_test == user].tolist())
        fit_blocks = set(prep.blocks_train[prep.users_train == user].tolist())
        assert not (test_blocks & fit_blocks), f"Bloco de teste também no treino em {user}"

    for name, y in (("treino", prep.y_train), ("validação", prep.y_val), ("teste", prep.y_test)):
        assert sorted(np.unique(y).tolist()) == [0, 1, 2], f"Partição {name} sem as 3 classes"

    train_mean = float(np.abs(prep.X_train.mean(axis=0)).max())
    test_mean = float(np.abs(prep.X_test.mean(axis=0)).max())
    assert train_mean < 1e-6, f"Treino não centrado (desvio {train_mean:.2e})"
    assert test_mean > 1e-6, "Média do teste exatamente 0: o scaler viu o teste"
    if config.normalization == "per_subject":
        for user in np.unique(prep.users_train):
            m = float(np.abs(prep.X_train[prep.users_train == user].mean(axis=0)).max())
            assert m < 1e-6, f"Treino de {user} não centrado (desvio {m:.2e})"

    if verbose:
        print(f"  [ok] partições exatas e disjuntas, purge > {config.purge_samples}, scaler só no treino | "
              f"treino {len(prep.y_train)} | validação {len(prep.y_val)} | teste {len(prep.y_test)} | "
              f"purge {prep.preprocessing_report.get('purged_rows', 0)}", flush=True)


def checar_janelas(wprep, prep, purge: int, verbose: bool = True) -> None:
    """Cada janela tem T registros consecutivos do mesmo participante, bloco e partição,
    termina no registro rotulado, e nenhuma janela de treino/validação toca o teste."""
    offsets = np.arange(-(wprep.window - 1), 1)
    covered = {}
    for name in ("train", "val", "test"):
        source = getattr(wprep, f"source_{name}")
        rows = source[:, None] + offsets
        assert rows.min() >= 0, f"Janela de {name} começa antes da partição"
        users = getattr(prep, f"users_{name}")[rows]
        blocks = getattr(prep, f"blocks_{name}")[rows]
        positions = getattr(prep, f"positions_{name}")[rows]
        assert (users == users[:, -1:]).all(), f"Janela de {name} mistura participantes"
        assert (blocks == blocks[:, -1:]).all(), f"Janela de {name} cruza bloco"
        assert (np.diff(positions, axis=1) == 1).all(), f"Janela de {name} com registros não consecutivos"
        assert np.array_equal(getattr(wprep, f"y_{name}"), getattr(prep, f"y_{name}")[source]), \
            f"Rótulo de {name} não é o do último registro"
        assert np.array_equal(getattr(wprep, f"X_{name}"), getattr(prep, f"X_{name}")[rows].astype(np.float32)), \
            f"Conteúdo das janelas de {name} difere dos registros"
        covered[name] = set(zip(users.ravel().tolist(), positions.ravel().tolist()))

    test_keys = set(zip(prep.users_test.tolist(), prep.positions_test.tolist()))
    for name in ("train", "val"):
        assert not (covered[name] & test_keys), f"Registro de teste dentro de uma janela de {name}"

    for user in np.unique(prep.users_test):
        pos = {n: getattr(prep, f"positions_{n}")[getattr(prep, f"users_{n}") == user]
               for n in ("train", "val", "test")}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            if len(pos[a]) and len(pos[b]):
                gap = _min_gap(pos[a], pos[b])
                assert gap > purge, f"Purge {a}-{b} violado em {user}: distância {gap}"

    if verbose:
        print(f"  [ok] janelas T={wprep.window}: consecutivas, dentro do bloco e da partição | "
              f"treino {len(wprep.y_train)} | validação {len(wprep.y_val)} | teste {len(wprep.y_test)}",
              flush=True)


def _min_gap(a: np.ndarray, b: np.ndarray) -> int:
    b = np.sort(b)
    if len(b) == 1:
        return int(np.abs(a - b[0]).min())
    idx = np.clip(np.searchsorted(b, a), 1, len(b) - 1)
    return int(np.minimum(np.abs(a - b[idx - 1]), np.abs(a - b[idx])).min())
