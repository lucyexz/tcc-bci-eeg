from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
USERS = ["user_a", "user_b", "user_c", "user_d", "user_e"]


def load_user(user_name: str) -> pd.DataFrame:
    """Carrega o CSV de um participante (user_a ... user_e) de qualquer subpasta de data/raw."""
    matches = list(DATA_DIR.rglob(f"{user_name}.csv"))
    if not matches:
        raise FileNotFoundError(f"Arquivo {user_name}.csv não encontrado dentro de {DATA_DIR}")

    df = pd.read_csv(matches[0])
    df["user"] = user_name
    return df


def load_all_users() -> pd.DataFrame:
    """Carrega os 5 participantes em um único DataFrame, na ordem de gravação."""
    return pd.concat([load_user(user) for user in USERS], ignore_index=True)
