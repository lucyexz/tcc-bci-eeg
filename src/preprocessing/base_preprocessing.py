from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from src.constants import BLOCK_COLUMN, LABEL_COLUMN, POSITION_COLUMN, USER_COLUMN
from src.data.loader import load_all_users
from src.features.spectral_features import FEATURE_SETS, transform_features

OutlierStrategy = Literal["none", "clip_iqr"]
SplitStrategy = Literal["chronological", "random", "blocked_cv"]
Normalization = Literal["global", "per_subject"]

SPLIT_STRATEGIES = {"chronological", "random", "blocked_cv"}
NORMALIZATIONS = {"global", "per_subject"}


@dataclass
class BasePreprocessingConfig:
    test_size: float = 0.2
    random_state: int = 42
    remove_duplicates: bool = True
    handle_missing: bool = True
    imputation_strategy: str = "median"
    outlier_strategy: OutlierStrategy = "clip_iqr"
    iqr_multiplier: float = 1.5
    scale_data: bool = True
    split_strategy: SplitStrategy = "random"
    subject: str | None = None
    validation_size: float = 0.0

    # Representação das features (ver src/features/spectral_features.py).
    # "raw" reproduz exatamente a v1.
    feature_set: str = "raw"

    # "global": um único IQR/StandardScaler ajustado no treino inteiro (v1).
    # "per_subject": limites de IQR e StandardScaler ajustados separadamente
    # no treino de cada sujeito e aplicados só às amostras daquele sujeito.
    # Equivale a uma sessão de calibração por usuário.
    normalization: Normalization = "global"

    # Validação cruzada por blocos de trials (split_strategy="blocked_cv").
    # Os blocos de cada sujeito são agrupados em `cv_n_folds` folds
    # contíguos; `cv_fold` é o fold de teste e o fold seguinte vira validação.
    # `test_size` é ignorado nesse modo.
    cv_n_folds: int = 6
    cv_fold: int = 0
    # Amostras descartadas do treino/validação em cada fronteira com o teste
    # (e do treino na fronteira com a validação). Remove as vizinhas
    # temporais que são quase cópias das amostras de teste.
    purge_samples: int = 40
    # Qual fold vira validação: "next" (fold seguinte, circular — padrão) ou
    # "previous" (fold anterior, circular). Usado no diagnóstico sobre a
    # proximidade temporal entre validação e teste.
    cv_val_position: str = "next"

    def __post_init__(self) -> None:
        if not (0.0 <= self.validation_size < 1.0):
            raise ValueError(
                f"`validation_size` deve estar em [0, 1) "
                f"(0 desativa a validação explícita), "
                f"recebido: {self.validation_size}"
            )
        if not (0.0 < self.test_size < 1.0):
            raise ValueError(
                f"`test_size` deve estar entre 0 e 1 (exclusivo), "
                f"recebido: {self.test_size}"
            )
        if self.iqr_multiplier < 0:
            raise ValueError(
                f"`iqr_multiplier` deve ser >= 0, "
                f"recebido: {self.iqr_multiplier}"
            )
        valid_strategies = {"mean", "median", "most_frequent", "constant"}
        if self.imputation_strategy not in valid_strategies:
            raise ValueError(
                f"`imputation_strategy` inválida: '{self.imputation_strategy}'. "
                f"Opções válidas: {valid_strategies}"
            )
        if self.split_strategy not in SPLIT_STRATEGIES:
            raise ValueError(
                f"`split_strategy` inválida: '{self.split_strategy}'. "
                f"Opções válidas: {sorted(SPLIT_STRATEGIES)}"
            )
        if self.feature_set not in FEATURE_SETS:
            raise ValueError(
                f"`feature_set` inválido: '{self.feature_set}'. "
                f"Opções válidas: {FEATURE_SETS}"
            )
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
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    users_train: np.ndarray
    users_test: np.ndarray

    feature_columns: list[str]

    # StandardScaler único (normalization="global") ou dict sujeito → scaler
    # (normalization="per_subject").
    scaler: StandardScaler | dict[str, StandardScaler] | None
    imputer: SimpleImputer | None

    processed_train_dataframe: pd.DataFrame
    processed_test_dataframe: pd.DataFrame

    preprocessing_report: dict

    # Conjunto de validação explícito. Só é preenchido quando
    # `validation_size > 0`; caso contrário permanece None e o Keras volta a
    # fatiar a cauda do treino via `validation_split`.
    X_val: np.ndarray | None = None
    y_val: np.ndarray | None = None
    users_val: np.ndarray | None = None

    # Bloco de trial (0..35 por sujeito) e posição temporal de cada amostra
    # dentro do sujeito. Servem para a acurácia por trial, para as timelines
    # de predição e para auditar a CV por blocos.
    blocks_train: np.ndarray | None = None
    blocks_test: np.ndarray | None = None
    blocks_val: np.ndarray | None = None
    positions_train: np.ndarray | None = None
    positions_test: np.ndarray | None = None
    positions_val: np.ndarray | None = None

    @property
    def n_feature_maps(self) -> int:
        return len(self.feature_columns) // 25


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    ignored_columns = {LABEL_COLUMN, USER_COLUMN, BLOCK_COLUMN, POSITION_COLUMN}

    return [
        column
        for column in df.columns
        if column not in ignored_columns
    ]


def validate_required_columns(df: pd.DataFrame) -> None:
    required_columns = [LABEL_COLUMN, USER_COLUMN]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Colunas obrigatórias não encontradas: {missing_columns}"
        )


def validate_numeric_features(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    non_numeric_columns = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(df[column])
    ]

    if non_numeric_columns:
        raise TypeError(
            "As seguintes features não são numéricas: "
            f"{non_numeric_columns}"
        )


def validate_labels_are_whole_numbers(df: pd.DataFrame) -> None:
    label_series = df[LABEL_COLUMN]
    fractional_mask = label_series.apply(lambda v: float(v) != int(float(v)))
    bad_values = label_series[fractional_mask].unique().tolist()
    if bad_values:
        raise ValueError(
            f"Labels com parte fracionária encontradas: {bad_values}. "
            "Labels devem ser inteiros exatos (0, 1 ou 2)."
        )


def validate_labels(df: pd.DataFrame) -> None:
    valid_labels = {0, 1, 2}

    labels = set(df[LABEL_COLUMN].unique())

    invalid_labels = labels - valid_labels

    if invalid_labels:
        raise ValueError(
            f"Labels inválidas encontradas: {invalid_labels}. "
            f"Labels esperadas: {valid_labels}"
        )


def remove_rows_without_label_or_user(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before_rows = len(df)

    df_clean = df.dropna(subset=[LABEL_COLUMN, USER_COLUMN]).copy()

    removed_rows = before_rows - len(df_clean)

    return df_clean, removed_rows


def assign_trial_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """Numera os blocos contíguos de rótulo e a posição temporal por sujeito.

    Cada sujeito foi gravado como uma sequência de trials, e cada trial é um
    trecho contíguo com o mesmo rótulo (o ciclo Esquerda → Direita → Neutro
    repetido 12×, totalizando 36 blocos). Um novo bloco começa sempre que o
    rótulo muda. A posição é o índice da amostra dentro do sujeito, depois da
    limpeza.
    """
    df = df.copy()
    labels = df[LABEL_COLUMN]
    users = df[USER_COLUMN]

    new_block = (labels != labels.shift()) | (users != users.shift())
    df[BLOCK_COLUMN] = new_block.groupby(users).cumsum().astype(int) - 1
    df[POSITION_COLUMN] = df.groupby(USER_COLUMN).cumcount().astype(int)

    return df


def _index_chronological_split(
    df: pd.DataFrame,
    test_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Índices posicionais de treino/teste do corte cronológico por sujeito."""
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    users = df[USER_COLUMN].to_numpy()

    for user in sorted(np.unique(users)):
        user_idx = np.flatnonzero(users == user)
        n_train = int(len(user_idx) * (1 - test_size))
        train_parts.append(user_idx[:n_train])
        test_parts.append(user_idx[n_train:])

    return np.concatenate(train_parts), np.concatenate(test_parts)


def random_stratified_split(
    df: pd.DataFrame,
    feature_columns: list[str],
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = df[feature_columns].to_numpy()
    y = df[LABEL_COLUMN].to_numpy()
    users = df[USER_COLUMN].to_numpy()

    X_train, X_test, y_train, y_test, users_train, users_test = train_test_split(
        X, y, users,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test, users_train, users_test


def chronological_split_by_user(
    df: pd.DataFrame,
    feature_columns: list[str],
    test_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_idx, test_idx = _index_chronological_split(df, test_size)

    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    return (
        train_df[feature_columns].to_numpy(),
        test_df[feature_columns].to_numpy(),
        train_df[LABEL_COLUMN].to_numpy(),
        test_df[LABEL_COLUMN].to_numpy(),
        train_df[USER_COLUMN].to_numpy(),
        test_df[USER_COLUMN].to_numpy(),
    )


def _chronological_validation_indices(
    users_train: np.ndarray,
    validation_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    train_indices: list[np.ndarray] = []
    val_indices: list[np.ndarray] = []

    for user in sorted(np.unique(users_train)):
        user_indices = np.flatnonzero(users_train == user)
        n_val = int(len(user_indices) * validation_size)

        if n_val == 0:
            train_indices.append(user_indices)
            continue

        train_indices.append(user_indices[:-n_val])
        val_indices.append(user_indices[-n_val:])

    return (
        np.concatenate(train_indices),
        np.concatenate(val_indices) if val_indices else np.array([], dtype=int),
    )


def _random_validation_indices(
    y_train: np.ndarray,
    validation_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    train_idx, val_idx = train_test_split(
        np.arange(len(y_train)),
        test_size=validation_size,
        random_state=random_state,
        stratify=y_train,
    )
    return np.sort(train_idx), np.sort(val_idx)


def carve_chronological_validation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    users_train: np.ndarray,
    validation_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Separa a cauda temporal do treino de CADA sujeito como validação.

    `chronological_split_by_user` preserva a ordem cronológica dentro de cada
    sujeito, então as últimas linhas do bloco de um sujeito são as mais
    recentes dele. Fatiar por sujeito evita o problema do `validation_split` do
    Keras, que pega a cauda do array concatenado e acabaria montando uma
    validação composta por um único sujeito.

    A validação sai de dentro do treino — o conjunto de teste não é tocado.
    """
    train_idx, val_idx = _chronological_validation_indices(
        users_train, validation_size
    )
    return _apply_validation_indices(
        X_train, y_train, users_train, train_idx=train_idx, val_idx=val_idx
    )


def carve_random_validation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    users_train: np.ndarray,
    validation_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Separa uma fração aleatória e estratificada do treino como validação."""
    train_idx, val_idx = _random_validation_indices(
        y_train, validation_size, random_state
    )
    return _apply_validation_indices(
        X_train, y_train, users_train, train_idx=train_idx, val_idx=val_idx
    )


def _apply_validation_indices(
    X_train: np.ndarray,
    y_train: np.ndarray,
    users_train: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        X_train[train_idx],
        y_train[train_idx],
        users_train[train_idx],
        X_train[val_idx],
        y_train[val_idx],
        users_train[val_idx],
    )


def _distance_to_range(positions: np.ndarray, start: int, end: int) -> np.ndarray:
    """Distância (em amostras) de cada posição ao intervalo fechado [start, end]."""
    return np.maximum(0, np.maximum(start - positions, positions - end))


def blocked_cv_split(
    df: pd.DataFrame,
    n_folds: int,
    fold: int,
    use_validation: bool,
    purge_samples: int,
    val_position: str = "next",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Validação cruzada por blocos de trials, sujeito a sujeito.

    Os blocos de cada sujeito são distribuídos em `n_folds` grupos contíguos
    (com 36 blocos e 6 folds: 6 blocos = 2 ciclos completos E→D→N por fold, o
    que mantém as classes balanceadas). O fold `fold` vira teste, o seguinte
    (circular) vira validação e o resto é treino.

    O purge descarta as amostras de treino/validação a até `purge_samples`
    posições do teste, e as de treino a até `purge_samples` da validação.

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
                f"Sujeito '{user}' tem {n_blocks} blocos, menos que "
                f"{n_folds} folds."
            )

        block_fold = user_blocks * n_folds // n_blocks

        is_test = block_fold == fold
        is_val = (block_fold == val_fold) if use_validation else np.zeros_like(is_test)
        is_train = ~is_test & ~is_val

        test_pos = user_positions[is_test]
        dist_test = _distance_to_range(user_positions, test_pos.min(), test_pos.max())

        keep = np.ones(len(user_idx), dtype=bool)
        keep &= ~((is_train | is_val) & (dist_test <= purge_samples))

        if use_validation:
            val_pos = user_positions[is_val]
            dist_val = _distance_to_range(user_positions, val_pos.min(), val_pos.max())
            keep &= ~(is_train & (dist_val <= purge_samples))

        purged += int((~keep).sum())

        train_parts.append(user_idx[is_train & keep])
        val_parts.append(user_idx[is_val & keep])
        test_parts.append(user_idx[is_test])
        test_blocks[str(user)] = sorted(np.unique(user_blocks[is_test]).tolist())

    info = {
        "cv_n_folds": n_folds,
        "cv_fold": fold,
        "cv_val_fold": val_fold if use_validation else None,
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


def remove_duplicate_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before_rows = len(df)

    df_clean = df.drop_duplicates().reset_index(drop=True)

    removed_duplicates = before_rows - len(df_clean)

    return df_clean, removed_duplicates


def calculate_iqr_bounds(
    X_train: np.ndarray,
    iqr_multiplier: float,
) -> tuple[np.ndarray, np.ndarray]:
    q1 = np.percentile(X_train, 25, axis=0)
    q3 = np.percentile(X_train, 75, axis=0)

    iqr = q3 - q1

    lower_bound = q1 - iqr_multiplier * iqr
    upper_bound = q3 + iqr_multiplier * iqr

    return lower_bound, upper_bound


def clip_outliers(
    X: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
) -> np.ndarray:
    return np.clip(X, lower_bound, upper_bound)


def build_dataframe(
    X: np.ndarray,
    y: np.ndarray,
    users: np.ndarray,
    feature_columns: list[str],
) -> pd.DataFrame:
    df = pd.DataFrame(X, columns=feature_columns)
    df[LABEL_COLUMN] = y
    df[USER_COLUMN] = users

    return df


def _per_subject(
    fit_fn,
    X_train: np.ndarray,
    users_train: np.ndarray,
    others: list[tuple[np.ndarray, np.ndarray]],
):
    """Aplica `fit_fn` sujeito a sujeito: ajusta no treino, transforma o resto.

    `fit_fn(X_user_train)` devolve uma função `transform(X)`. Retorna os
    arrays transformados (treino primeiro, depois `others` na mesma ordem) e
    o dict sujeito → objeto ajustado. Todo sujeito avaliado precisa ter
    dados de treino.
    """
    X_train_out = np.empty_like(X_train, dtype=float)
    others_out = [np.empty_like(X, dtype=float) for X, _ in others]
    fitted = {}

    for user in sorted(np.unique(users_train)):
        mask = users_train == user
        transform, state = fit_fn(X_train[mask])
        fitted[str(user)] = state
        X_train_out[mask] = transform(X_train[mask])

        for (X, users), out in zip(others, others_out):
            other_mask = users == user
            if other_mask.any():
                out[other_mask] = transform(X[other_mask])

    for _, users in others:
        unseen = set(np.unique(users)) - set(np.unique(users_train))
        if unseen:
            raise ValueError(
                "normalization='per_subject' exige que todo sujeito avaliado "
                f"tenha dados de treino; sem treino: {sorted(unseen)}"
            )

    return X_train_out, others_out, fitted


def prepare_base_data(
    config: BasePreprocessingConfig | None = None,
) -> BasePreprocessingResult:
    if config is None:
        config = BasePreprocessingConfig()

    df = load_all_users()

    # O filtro por sujeito acontece antes de qualquer split, para que o track
    # personalizado nunca enxergue dados dos outros sujeitos.
    if config.subject is not None:
        available_subjects = sorted(df[USER_COLUMN].unique())

        if config.subject not in available_subjects:
            raise ValueError(
                f"Sujeito '{config.subject}' não encontrado. "
                f"Disponíveis: {available_subjects}"
            )

        df = df[df[USER_COLUMN] == config.subject].reset_index(drop=True)

    report = {
        "subject": config.subject,
        "initial_shape": df.shape,
        "initial_missing_values": int(df.isnull().sum().sum()),
    }

    validate_required_columns(df)

    df, removed_rows_without_label_or_user = remove_rows_without_label_or_user(df)

    report["removed_rows_without_label_or_user"] = removed_rows_without_label_or_user

    validate_labels_are_whole_numbers(df)

    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)
    df[USER_COLUMN] = df[USER_COLUMN].astype(str)

    validate_labels(df)

    feature_columns = get_feature_columns(df)

    validate_numeric_features(df, feature_columns)

    removed_duplicates = 0

    if config.remove_duplicates:
        df, removed_duplicates = remove_duplicate_rows(df)

    report["removed_duplicates"] = removed_duplicates
    report["shape_after_cleaning"] = df.shape
    report["split_strategy"] = config.split_strategy
    report["test_size"] = config.test_size

    # Blocos e posição são calculados depois da limpeza, sobre a ordem
    # cronológica original de cada sujeito.
    df = assign_trial_blocks(df)
    report["blocks_per_user"] = {
        user: int(n)
        for user, n in df.groupby(USER_COLUMN)[BLOCK_COLUMN].nunique().items()
    }

    # Transformação de features: por linha e sem estado, então é segura antes
    # do split (nada é ajustado em dados de teste).
    df, feature_columns = transform_features(df, config.feature_set)
    report["feature_set"] = config.feature_set
    report["n_features"] = len(feature_columns)
    report["normalization"] = config.normalization

    # Todos os splits são feitos por índice posicional sobre `df`, o que
    # permite carregar bloco e posição junto com cada amostra.
    use_validation = config.validation_size > 0.0
    val_idx = None

    if config.split_strategy == "random":
        train_idx, test_idx = train_test_split(
            np.arange(len(df)),
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=df[LABEL_COLUMN].to_numpy(),
        )
        if use_validation:
            inner_train, inner_val = _random_validation_indices(
                df[LABEL_COLUMN].to_numpy()[train_idx],
                config.validation_size,
                config.random_state,
            )
            train_idx, val_idx = train_idx[inner_train], train_idx[inner_val]

    elif config.split_strategy == "chronological":
        train_idx, test_idx = _index_chronological_split(df, config.test_size)
        if use_validation:
            inner_train, inner_val = _chronological_validation_indices(
                df[USER_COLUMN].to_numpy()[train_idx],
                config.validation_size,
            )
            train_idx, val_idx = train_idx[inner_train], train_idx[inner_val]

    else:
        train_idx, val_idx, test_idx, cv_info = blocked_cv_split(
            df,
            n_folds=config.cv_n_folds,
            fold=config.cv_fold,
            use_validation=use_validation,
            purge_samples=config.purge_samples,
            val_position=config.cv_val_position,
        )
        report.update(cv_info)
        if not use_validation:
            val_idx = None

    report.setdefault("purged_rows", 0)

    def take(idx: np.ndarray, column: str | list[str]) -> np.ndarray:
        return df.iloc[idx][column].to_numpy()

    X_train = take(train_idx, feature_columns).astype(float)
    X_test = take(test_idx, feature_columns).astype(float)
    y_train = take(train_idx, LABEL_COLUMN)
    y_test = take(test_idx, LABEL_COLUMN)
    users_train = take(train_idx, USER_COLUMN)
    users_test = take(test_idx, USER_COLUMN)
    blocks_train = take(train_idx, BLOCK_COLUMN)
    blocks_test = take(test_idx, BLOCK_COLUMN)
    positions_train = take(train_idx, POSITION_COLUMN)
    positions_test = take(test_idx, POSITION_COLUMN)

    X_val = y_val = users_val = blocks_val = positions_val = None

    if val_idx is not None:
        X_val = take(val_idx, feature_columns).astype(float)
        y_val = take(val_idx, LABEL_COLUMN)
        users_val = take(val_idx, USER_COLUMN)
        blocks_val = take(val_idx, BLOCK_COLUMN)
        positions_val = take(val_idx, POSITION_COLUMN)

    report["validation_size"] = config.validation_size

    report["missing_values_before_imputation_train"] = int(np.isnan(X_train).sum())
    report["missing_values_before_imputation_test"] = int(np.isnan(X_test).sum())

    imputer = None

    if config.handle_missing:
        imputer = SimpleImputer(strategy=config.imputation_strategy)

        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        if X_val is not None:
            X_val = imputer.transform(X_val)

    else:
        missing_train = int(np.isnan(X_train).sum())
        missing_test = int(np.isnan(X_test).sum())

        if X_val is not None:
            missing_train += int(np.isnan(X_val).sum())

        if missing_train > 0 or missing_test > 0:
            raise ValueError(
                "Foram encontrados valores ausentes, mas "
                "`handle_missing=False` foi configurado."
            )

    report["missing_values_after_imputation_train"] = int(np.isnan(X_train).sum())
    report["missing_values_after_imputation_test"] = int(np.isnan(X_test).sum())

    report["class_distribution_train"] = {
        int(label): int((y_train == label).sum())
        for label in sorted(np.unique(y_train))
    }
    report["class_distribution_test"] = {
        int(label): int((y_test == label).sum())
        for label in sorted(np.unique(y_test))
    }
    report["samples_per_user_train"] = {
        user: int((users_train == user).sum())
        for user in sorted(np.unique(users_train))
    }
    report["samples_per_user_test"] = {
        user: int((users_test == user).sum())
        for user in sorted(np.unique(users_test))
    }

    if y_val is not None:
        report["class_distribution_val"] = {
            int(label): int((y_val == label).sum())
            for label in sorted(np.unique(y_val))
        }
        report["samples_per_user_val"] = {
            user: int((users_val == user).sum())
            for user in sorted(np.unique(users_val))
        }

    others = [(X_test, users_test)]
    if X_val is not None:
        others.append((X_val, users_val))

    def unpack(others_out):
        test = others_out[0]
        val = others_out[1] if X_val is not None else None
        return test, val

    if config.outlier_strategy == "clip_iqr":
        if config.normalization == "global":
            lower_bound, upper_bound = calculate_iqr_bounds(
                X_train=X_train,
                iqr_multiplier=config.iqr_multiplier,
            )

            X_train = clip_outliers(X_train, lower_bound, upper_bound)
            X_test = clip_outliers(X_test, lower_bound, upper_bound)

            if X_val is not None:
                X_val = clip_outliers(X_val, lower_bound, upper_bound)

        else:
            # Limites por sujeito: com sujeitos em escalas diferentes, um
            # limite global cortaria um sujeito inteiro em vez de só a cauda.
            def fit_iqr(X_user):
                lower, upper = calculate_iqr_bounds(X_user, config.iqr_multiplier)
                return (lambda X: clip_outliers(X, lower, upper)), (lower, upper)

            X_train, others_out, _ = _per_subject(
                fit_iqr, X_train, users_train, others
            )
            X_test, X_val = unpack(others_out)
            others = [(X_test, users_test)]
            if X_val is not None:
                others.append((X_val, users_val))

        report["outlier_strategy"] = "clip_iqr"
        report["iqr_multiplier"] = config.iqr_multiplier

    elif config.outlier_strategy == "none":
        report["outlier_strategy"] = "none"

    else:
        raise ValueError(
            f"Estratégia de outlier inválida: {config.outlier_strategy}"
        )

    scaler = None

    if config.scale_data:
        if config.normalization == "global":
            scaler = StandardScaler()

            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            if X_val is not None:
                X_val = scaler.transform(X_val)

        else:
            def fit_scaler(X_user):
                user_scaler = StandardScaler().fit(X_user)
                return user_scaler.transform, user_scaler

            X_train, others_out, scaler = _per_subject(
                fit_scaler, X_train, users_train, others
            )
            X_test, X_val = unpack(others_out)

    report["scale_data"] = config.scale_data
    report["final_train_shape"] = X_train.shape
    report["final_test_shape"] = X_test.shape

    if X_val is not None:
        report["final_val_shape"] = X_val.shape

    processed_train_dataframe = build_dataframe(
        X=X_train,
        y=y_train,
        users=users_train,
        feature_columns=feature_columns,
    )

    processed_test_dataframe = build_dataframe(
        X=X_test,
        y=y_test,
        users=users_test,
        feature_columns=feature_columns,
    )

    return BasePreprocessingResult(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        users_train=users_train,
        users_test=users_test,
        feature_columns=feature_columns,
        scaler=scaler,
        imputer=imputer,
        processed_train_dataframe=processed_train_dataframe,
        processed_test_dataframe=processed_test_dataframe,
        preprocessing_report=report,
        X_val=X_val,
        y_val=y_val,
        users_val=users_val,
        blocks_train=blocks_train,
        blocks_test=blocks_test,
        blocks_val=blocks_val,
        positions_train=positions_train,
        positions_test=positions_test,
        positions_val=positions_val,
    )
