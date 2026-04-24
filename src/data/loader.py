from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def load_user(user_name: str) -> pd.DataFrame:
    """
    Carrega o arquivo CSV de um usuário específico.
    Exemplo: user_a, user_b, user_c, user_d, user_e.
    """

    matches = list(DATA_DIR.rglob(f"{user_name}.csv"))

    if not matches:
        raise FileNotFoundError(
            f"Arquivo {user_name}.csv não encontrado dentro de {DATA_DIR}"
        )

    file_path = matches[0]

    df = pd.read_csv(file_path)
    df["user"] = user_name

    return df


def load_all_users() -> pd.DataFrame:
    """
    Carrega todos os usuários do dataset e junta em um único DataFrame.
    """

    users = ["user_a", "user_b", "user_c", "user_d", "user_e"]

    dataframes = []

    for user in users:
        df_user = load_user(user)
        dataframes.append(df_user)

    df = pd.concat(dataframes, ignore_index=True)

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Retorna apenas as colunas de entrada dos modelos,
    removendo label e identificação do usuário.
    """

    ignored_columns = ["Values", "user"]

    return [col for col in df.columns if col not in ignored_columns]


def get_features_and_labels(df: pd.DataFrame):
    """
    Separa X e y.
    X = features EEG
    y = classe/label
    """

    feature_columns = get_feature_columns(df)

    X = df[feature_columns]
    y = df["Values"]

    return X, y