from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from src.data.loader import load_all_users


LABEL_COLUMN = "Values"
USER_COLUMN = "user"

OutlierStrategy = Literal["none", "clip_iqr"]


@dataclass
class BasePreprocessingConfig:
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True
    remove_duplicates: bool = True
    handle_missing: bool = True
    imputation_strategy: str = "median"
    outlier_strategy: OutlierStrategy = "clip_iqr"
    iqr_multiplier: float = 1.5
    scale_data: bool = True


@dataclass
class BasePreprocessingResult:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    users_train: np.ndarray
    users_test: np.ndarray

    feature_columns: list[str]

    scaler: StandardScaler | None
    imputer: SimpleImputer | None

    train_dataframe: pd.DataFrame
    test_dataframe: pd.DataFrame

    preprocessing_report: dict


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    ignored_columns = {LABEL_COLUMN, USER_COLUMN}

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


def validate_labels(df: pd.DataFrame) -> None:
    valid_labels = {0, 1, 2}

    labels = set(df[LABEL_COLUMN].dropna().astype(int).unique())

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


def prepare_base_data(
    config: BasePreprocessingConfig | None = None,
) -> BasePreprocessingResult:
    if config is None:
        config = BasePreprocessingConfig()

    df = load_all_users()

    report = {
        "initial_shape": df.shape,
        "initial_missing_values": int(df.isnull().sum().sum()),
    }

    validate_required_columns(df)

    df, removed_rows_without_label_or_user = remove_rows_without_label_or_user(df)

    report["removed_rows_without_label_or_user"] = removed_rows_without_label_or_user

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

    X = df[feature_columns].to_numpy()
    y = df[LABEL_COLUMN].to_numpy()
    users = df[USER_COLUMN].to_numpy()

    stratify_values = y if config.stratify else None

    X_train, X_test, y_train, y_test, users_train, users_test = train_test_split(
        X,
        y,
        users,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=stratify_values,
    )

    imputer = None

    if config.handle_missing:
        imputer = SimpleImputer(strategy=config.imputation_strategy)

        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

    else:
        missing_train = int(np.isnan(X_train).sum())
        missing_test = int(np.isnan(X_test).sum())

        if missing_train > 0 or missing_test > 0:
            raise ValueError(
                "Foram encontrados valores ausentes, mas "
                "`handle_missing=False` foi configurado."
            )

    report["missing_values_after_imputation_train"] = int(np.isnan(X_train).sum())
    report["missing_values_after_imputation_test"] = int(np.isnan(X_test).sum())

    if config.outlier_strategy == "clip_iqr":
        lower_bound, upper_bound = calculate_iqr_bounds(
            X_train=X_train,
            iqr_multiplier=config.iqr_multiplier,
        )

        X_train = clip_outliers(X_train, lower_bound, upper_bound)
        X_test = clip_outliers(X_test, lower_bound, upper_bound)

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
        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    report["scale_data"] = config.scale_data
    report["final_train_shape"] = X_train.shape
    report["final_test_shape"] = X_test.shape

    train_dataframe = build_dataframe(
        X=X_train,
        y=y_train,
        users=users_train,
        feature_columns=feature_columns,
    )

    test_dataframe = build_dataframe(
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
        train_dataframe=train_dataframe,
        test_dataframe=test_dataframe,
        preprocessing_report=report,
    )