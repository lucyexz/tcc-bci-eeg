"""Representações alternativas das features de potência espectral.

As 25 colunas brutas são `POW.<canal>.<banda>` em ordem canal-major
(AF3, T7, Pz, T8, AF4) × (Theta, Alpha, BetaL, BetaH, Gamma).

Todas as transformações aqui são **por linha e sem estado**: cada amostra é
transformada olhando só para ela mesma. Por isso podem ser aplicadas antes do
split sem nenhum risco de vazamento — nada é ajustado em dados de teste.

Feature sets disponíveis:

- ``raw``     — potência bruta (v1). 25 colunas, 1 mapa 5×5.
- ``log``     — log10 da potência. 25 colunas, 1 mapa 5×5.
- ``log_rel`` — log10 da potência + log10 da potência relativa (banda ÷ soma
  das 5 bandas do mesmo canal). 50 colunas, 2 mapas 5×5.

Os mapas são concatenados em ordem mapa-major: as 25 colunas do mapa 0, depois
as 25 do mapa 1, cada bloco em ordem canal-major. Os `reshape_for_*` dos
modelos dependem dessa convenção.
"""

from typing import Literal

import numpy as np
import pandas as pd

from src.constants import EEG_BANDS, EEG_CHANNELS

FeatureSet = Literal["raw", "log", "log_rel"]
FEATURE_SETS: tuple[str, ...] = ("raw", "log", "log_rel")

N_CHANNELS = len(EEG_CHANNELS)
N_BANDS = len(EEG_BANDS)
GRID_SIZE = N_CHANNELS * N_BANDS

# Potências exatamente 0 existem no dataset (4 valores) e não têm log. O piso
# fica bem abaixo da menor potência positiva observada, então só afeta esses
# casos degenerados.
POWER_FLOOR = 1e-3

# Pares de eletrodos simétricos (hemisfério esquerdo, hemisfério direito).
ASYMMETRY_PAIRS: tuple[tuple[str, str], ...] = (("AF3", "AF4"), ("T7", "T8"))


def raw_column(channel: str, band: str) -> str:
    return f"POW.{channel}.{band}"


def raw_columns() -> list[str]:
    return [raw_column(ch, band) for ch in EEG_CHANNELS for band in EEG_BANDS]


def n_feature_maps(feature_set: str) -> int:
    return 2 if feature_set == "log_rel" else 1


def _log_power(power: np.ndarray) -> np.ndarray:
    return np.log10(np.maximum(power, POWER_FLOOR))


def _log_relative_power(power: np.ndarray) -> np.ndarray:
    """log10(banda / soma das bandas do canal), calculado canal a canal."""
    grid = np.maximum(power, POWER_FLOOR).reshape(-1, N_CHANNELS, N_BANDS)
    total = grid.sum(axis=2, keepdims=True)
    return np.log10(grid / total).reshape(-1, GRID_SIZE)


def transform_features(
    df: pd.DataFrame,
    feature_set: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Substitui as 25 colunas brutas pelas do feature set pedido.

    As demais colunas (rótulo, sujeito, bloco...) são preservadas. Valores
    ausentes continuam ausentes: a imputação acontece depois do split, ajustada
    só no treino.

    Returns:
        (DataFrame transformado, lista ordenada das colunas de feature)
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"feature_set inválido: '{feature_set}'. Opções: {FEATURE_SETS}"
        )

    columns = raw_columns()
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas de potência ausentes: {missing}")

    if feature_set == "raw":
        return df, columns

    power = df[columns].to_numpy(dtype=float)
    other = df.drop(columns=columns)

    blocks = [(_log_power(power), "LOG")]
    if feature_set == "log_rel":
        blocks.append((_log_relative_power(power), "REL"))

    feature_frames = []
    feature_columns: list[str] = []

    for values, prefix in blocks:
        names = [f"{prefix}.{ch}.{band}" for ch in EEG_CHANNELS for band in EEG_BANDS]
        feature_frames.append(pd.DataFrame(values, columns=names, index=df.index))
        feature_columns.extend(names)

    out = pd.concat([*feature_frames, other], axis=1)
    return out, feature_columns


def asymmetry_features(df: pd.DataFrame) -> pd.DataFrame:
    """Assimetria hemisférica log10(esquerdo) − log10(direito) por banda.

    Usada na análise de dados e de explicabilidade. Não entra como entrada das
    redes: no espaço log ela é combinação linear das features ``log``, então as
    redes já conseguem aprendê-la a partir da grade 5×5.
    """
    out = {}
    for left, right in ASYMMETRY_PAIRS:
        for band in EEG_BANDS:
            left_power = df[raw_column(left, band)].to_numpy(dtype=float)
            right_power = df[raw_column(right, band)].to_numpy(dtype=float)
            out[f"ASY.{left}-{right}.{band}"] = (
                _log_power(left_power) - _log_power(right_power)
            )
    return pd.DataFrame(out, index=df.index)
