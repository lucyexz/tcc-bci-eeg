"""Pré-processamento por registro, antes do janelamento.

Etapas, nesta ordem:

1. carrega os 5 participantes (ou só um, no paradigma especialista);
2. limpeza: descarta linhas sem rótulo/participante e linhas duplicadas;
3. numera os blocos (trechos contínuos da mesma classe) e a posição de cada
   registro na gravação do participante;
4. troca a potência pelo log10 da potência (por linha e sem estado, então
   pode vir antes da divisão sem risco de vazamento);
5. divide por blocos em treino, validação e teste, com purge nas fronteiras;
6. imputa valores ausentes pela mediana do treino;
7. normaliza com StandardScaler ajustado só no treino (um único, ou um por
   participante).

Os passos 6 e 7 são ajustados só no treino e depois aplicados à validação e ao
teste, então nada do teste entra no ajuste.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.constantes import (BLOCK_COLUMN, EEG_BANDS, EEG_CHANNELS, LABEL_COLUMN, POSITION_COLUMN,
                            USER_COLUMN)
from src.dados.carregamento import load_all_users

Normalization = Literal["global", "per_subject"]
NORMALIZATIONS = {"global", "per_subject"}

# Potências exatamente 0 existem no dataset (4 valores) e não têm log. O piso
# fica bem abaixo da menor potência positiva observada, então só afeta esses
# casos degenerados.
POWER_FLOOR = 1e-3


def raw_columns() -> list[str]:
    """As 25 colunas de potência, `POW.<canal>.<banda>`, em ordem canal-major."""
    return [f"POW.{ch}.{band}" for ch in EEG_CHANNELS for band in EEG_BANDS]


def log_power_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Substitui as 25 colunas de potência pelo log10 delas (prefixo `LOG.`).

    Mantém a ordem canal-major (AF3, T7, Pz, T8, AF4) × (Theta, Alpha, BetaL,
    BetaH, Gamma), da qual os `reshape` dos modelos dependem. As demais colunas
    são preservadas, e valores ausentes continuam ausentes (a imputação vem
    depois da divisão, ajustada só no treino).

    Returns:
        (DataFrame transformado, lista ordenada das colunas de feature)
    """
    columns = raw_columns()
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas de potência ausentes: {missing}")

    power = df[columns].to_numpy(dtype=float)
    names = [f"LOG.{ch}.{band}" for ch in EEG_CHANNELS for band in EEG_BANDS]
    features = pd.DataFrame(np.log10(np.maximum(power, POWER_FLOOR)), columns=names, index=df.index)
    return pd.concat([features, df.drop(columns=columns)], axis=1), names


@dataclass
class BasePreprocessingConfig:
    # None = os 5 participantes (generalista); um nome = só ele (especialista).
    subject: str | None = None

    # "global": um StandardScaler ajustado no treino inteiro.
    # "per_subject": um StandardScaler por participante, ajustado no treino
    # dele e aplicado só aos registros dele (equivale a uma calibração por
    # usuário).
    normalization: Normalization = "global"

    # Divisão por blocos: os blocos de cada participante são agrupados em
    # `cv_n_folds` grupos contíguos; `cv_fold` é o grupo de teste e o grupo
    # seguinte ("next") ou anterior ("previous"), circular, é a validação.
    cv_n_folds: int = 6
    cv_fold: int = 0
    cv_val_position: str = "next"

    # Registros descartados do treino/validação em cada fronteira com o teste
    # (e do treino na fronteira com a validação): os vizinhos temporais são
    # quase cópias dos registros do outro lado.
    purge_samples: int = 40

    def __post_init__(self) -> None:
        if self.normalization not in NORMALIZATIONS:
            raise ValueError(
                f"`normalization` inválida: '{self.normalization}'. "
                f"Opções válidas: {sorted(NORMALIZATIONS)}"
            )
        if self.cv_n_folds < 3:
            raise ValueError(
                f"`cv_n_folds` deve ser >= 3 (treino, validação e teste), "
                f"recebido: {self.cv_n_folds}"
            )
        if not (0 <= self.cv_fold < self.cv_n_folds):
            raise ValueError(
                f"`cv_fold` deve estar em [0, {self.cv_n_folds}), "
                f"recebido: {self.cv_fold}"
            )
        if self.cv_val_position not in {"next", "previous"}:
            raise ValueError(
                f"`cv_val_position` deve ser 'next' ou 'previous', "
                f"recebido: {self.cv_val_position}"
            )
        if self.purge_samples < 0:
            raise ValueError(
                f"`purge_samples` deve ser >= 0, recebido: {self.purge_samples}"
            )


@dataclass
class BasePreprocessingResult:
    """Registros de cada partição, já imputados e normalizados.

    Para cada partição (train, val, test): X (N, 25), rótulo, participante,
    bloco (0..35 dentro do participante) e posição na gravação.
    """

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    users_train: np.ndarray
    users_val: np.ndarray
    users_test: np.ndarray
    blocks_train: np.ndarray
    blocks_val: np.ndarray
    blocks_test: np.ndarray
    positions_train: np.ndarray
    positions_val: np.ndarray
    positions_test: np.ndarray
    feature_columns: list[str]
    preprocessing_report: dict


def validate_labels(df: pd.DataFrame) -> None:
    """Rótulos precisam ser inteiros exatos em {0, 1, 2}."""
    labels = df[LABEL_COLUMN]
    fractional = labels[labels.apply(lambda v: float(v) != int(float(v)))].unique().tolist()
    if fractional:
        raise ValueError(f"Rótulos com parte fracionária: {fractional}")
    invalid = set(labels.astype(int).unique()) - {0, 1, 2}
    if invalid:
        raise ValueError(f"Rótulos inválidos: {invalid}. Esperados: {{0, 1, 2}}")


def assign_trial_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """Numera os blocos contíguos de rótulo e a posição temporal por participante.

    Cada participante foi gravado como uma sequência de blocos, e cada bloco é
    um trecho contíguo com o mesmo rótulo (o ciclo Esquerda → Direita → Neutro
    repetido 12×, totalizando 36 blocos). Um novo bloco começa sempre que o
    rótulo muda. A posição é o índice do registro dentro do participante,
    depois da limpeza.
    """
    df = df.copy()
    labels = df[LABEL_COLUMN]
    users = df[USER_COLUMN]

    new_block = (labels != labels.shift()) | (users != users.shift())
    df[BLOCK_COLUMN] = new_block.groupby(users).cumsum().astype(int) - 1
    df[POSITION_COLUMN] = df.groupby(USER_COLUMN).cumcount().astype(int)

    return df


def _distance_to_range(positions: np.ndarray, start: int, end: int) -> np.ndarray:
    """Distância (em registros) de cada posição ao intervalo fechado [start, end]."""
    return np.maximum(0, np.maximum(start - positions, positions - end))


def blocked_split(
    df: pd.DataFrame,
    n_folds: int,
    fold: int,
    purge_samples: int,
    val_position: str = "next",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Divisão por blocos, participante a participante.

    Os blocos de cada participante são distribuídos em `n_folds` grupos
    contíguos (com 36 blocos e 6 grupos: 6 blocos = 2 ciclos completos E→D→N
    por grupo, o que mantém as classes balanceadas). O grupo `fold` vira
    teste, o seguinte ou o anterior (circular) vira validação e o resto é
    treino.

    O purge descarta os registros de treino/validação a até `purge_samples`
    posições do teste, e os de treino a até `purge_samples` da validação.

    Returns:
        (índices de treino, índices de validação, índices de teste, info)
    """
    users = df[USER_COLUMN].to_numpy()
    blocks = df[BLOCK_COLUMN].to_numpy()
    positions = df[POSITION_COLUMN].to_numpy()

    val_fold = (fold + (1 if val_position == "next" else -1)) % n_folds

    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    purged = 0
    test_blocks: dict[str, list[int]] = {}

    for user in sorted(np.unique(users)):
        user_idx = np.flatnonzero(users == user)
        user_blocks = blocks[user_idx]
        user_positions = positions[user_idx]

        n_blocks = int(user_blocks.max()) + 1
        if n_blocks < n_folds:
            raise ValueError(
                f"Participante '{user}' tem {n_blocks} blocos, menos que "
                f"{n_folds} grupos."
            )

        block_fold = user_blocks * n_folds // n_blocks

        is_test = block_fold == fold
        is_val = block_fold == val_fold
        is_train = ~is_test & ~is_val

        test_pos = user_positions[is_test]
        dist_test = _distance_to_range(user_positions, test_pos.min(), test_pos.max())
        val_pos = user_positions[is_val]
        dist_val = _distance_to_range(user_positions, val_pos.min(), val_pos.max())

        keep = np.ones(len(user_idx), dtype=bool)
        keep &= ~((is_train | is_val) & (dist_test <= purge_samples))
        keep &= ~(is_train & (dist_val <= purge_samples))

        purged += int((~keep).sum())

        train_parts.append(user_idx[is_train & keep])
        val_parts.append(user_idx[is_val & keep])
        test_parts.append(user_idx[is_test])
        test_blocks[str(user)] = sorted(np.unique(user_blocks[is_test]).tolist())

    info = {
        "cv_n_folds": n_folds,
        "cv_fold": fold,
        "cv_val_fold": val_fold,
        "purge_samples": purge_samples,
        "purged_rows": purged,
        "test_blocks_per_user": test_blocks,
    }

    return (
        np.concatenate(train_parts),
        np.concatenate(val_parts),
        np.concatenate(test_parts),
        info,
    )


def _per_subject(
    fit_fn,
    X_train: np.ndarray,
    users_train: np.ndarray,
    others: list[tuple[np.ndarray, np.ndarray]],
):
    """Aplica `fit_fn` participante a participante: ajusta no treino, transforma o resto.

    `fit_fn(X_user_train)` devolve uma função `transform(X)`. Retorna os
    arrays transformados (treino primeiro, depois `others` na mesma ordem).
    Todo participante avaliado precisa ter dados de treino.
    """
    X_train_out = np.empty_like(X_train, dtype=float)
    others_out = [np.empty_like(X, dtype=float) for X, _ in others]

    for user in sorted(np.unique(users_train)):
        mask = users_train == user
        transform = fit_fn(X_train[mask])
        X_train_out[mask] = transform(X_train[mask])

        for (X, users), out in zip(others, others_out):
            other_mask = users == user
            if other_mask.any():
                out[other_mask] = transform(X[other_mask])

    for _, users in others:
        unseen = set(np.unique(users)) - set(np.unique(users_train))
        if unseen:
            raise ValueError(
                "normalization='per_subject' exige que todo participante avaliado "
                f"tenha dados de treino; sem treino: {sorted(unseen)}"
            )

    return X_train_out, others_out


def prepare_base_data(config: BasePreprocessingConfig) -> BasePreprocessingResult:
    df = load_all_users()

    # O filtro por participante acontece antes de qualquer divisão, para que o
    # especialista nunca enxergue dados dos outros participantes.
    if config.subject is not None:
        available = sorted(df[USER_COLUMN].unique())
        if config.subject not in available:
            raise ValueError(
                f"Participante '{config.subject}' não encontrado. Disponíveis: {available}"
            )
        df = df[df[USER_COLUMN] == config.subject].reset_index(drop=True)

    report = {
        "subject": config.subject,
        "initial_shape": df.shape,
        "initial_missing_values": int(df.isnull().sum().sum()),
    }

    # Limpeza
    before = len(df)
    df = df.dropna(subset=[LABEL_COLUMN, USER_COLUMN]).copy()
    report["removed_rows_without_label_or_user"] = before - len(df)

    validate_labels(df)
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)
    df[USER_COLUMN] = df[USER_COLUMN].astype(str)

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    report["removed_duplicates"] = before - len(df)
    report["shape_after_cleaning"] = df.shape

    # Blocos e posição são calculados depois da limpeza, sobre a ordem
    # cronológica original de cada participante.
    df = assign_trial_blocks(df)
    report["blocks_per_user"] = {
        user: int(n)
        for user, n in df.groupby(USER_COLUMN)[BLOCK_COLUMN].nunique().items()
    }

    # log10 da potência: por linha e sem estado, então é seguro antes da
    # divisão (nada é ajustado em dados de teste).
    df, feature_columns = log_power_features(df)
    report["n_features"] = len(feature_columns)
    report["normalization"] = config.normalization

    # Divisão por índice posicional sobre `df`, o que permite carregar bloco e
    # posição junto com cada registro.
    train_idx, val_idx, test_idx, split_info = blocked_split(
        df,
        n_folds=config.cv_n_folds,
        fold=config.cv_fold,
        purge_samples=config.purge_samples,
        val_position=config.cv_val_position,
    )
    report.update(split_info)

    def take(idx: np.ndarray, column: str | list[str]) -> np.ndarray:
        return df.iloc[idx][column].to_numpy()

    parts = {}
    for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        parts[name] = {
            "X": take(idx, feature_columns).astype(float),
            "y": take(idx, LABEL_COLUMN),
            "users": take(idx, USER_COLUMN),
            "blocks": take(idx, BLOCK_COLUMN),
            "positions": take(idx, POSITION_COLUMN),
        }
    X_train, X_val, X_test = parts["train"]["X"], parts["val"]["X"], parts["test"]["X"]
    users_train, users_val, users_test = (parts[n]["users"] for n in ("train", "val", "test"))

    # Imputação pela mediana do treino
    report["missing_values_before_imputation_train"] = int(np.isnan(X_train).sum())
    report["missing_values_before_imputation_test"] = int(np.isnan(X_test).sum())
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    X_val = imputer.transform(X_val)

    for name in ("train", "val", "test"):
        y, users = parts[name]["y"], parts[name]["users"]
        report[f"class_distribution_{name}"] = {
            int(label): int((y == label).sum()) for label in sorted(np.unique(y))
        }
        report[f"samples_per_user_{name}"] = {
            user: int((users == user).sum()) for user in sorted(np.unique(users))
        }

    # Normalização ajustada só no treino
    if config.normalization == "global":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_val = scaler.transform(X_val)
    else:
        def fit_scaler(X_user):
            return StandardScaler().fit(X_user).transform

        X_train, (X_test, X_val) = _per_subject(
            fit_scaler, X_train, users_train, [(X_test, users_test), (X_val, users_val)]
        )

    report["final_train_shape"] = X_train.shape
    report["final_val_shape"] = X_val.shape
    report["final_test_shape"] = X_test.shape

    return BasePreprocessingResult(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=parts["train"]["y"],
        y_val=parts["val"]["y"],
        y_test=parts["test"]["y"],
        users_train=users_train,
        users_val=users_val,
        users_test=users_test,
        blocks_train=parts["train"]["blocks"],
        blocks_val=parts["val"]["blocks"],
        blocks_test=parts["test"]["blocks"],
        positions_train=parts["train"]["positions"],
        positions_val=parts["val"]["positions"],
        positions_test=parts["test"]["positions"],
        feature_columns=feature_columns,
        preprocessing_report=report,
    )
