"""Coleta os resultados da v3 (LOSO e LOTO do agrupado) numa tabela longa.

Mesmo formato de `_v2_results`: uma linha por (iteração, protocolo, variante,
modelo, sujeito, repetição). A repetição é a semente no LOSO e o fold no LOTO.

- LOSO: cada treino testa um único sujeito, então o JSON já é a métrica
  daquele sujeito.
- LOTO: cada treino testa 1 ciclo de todos os sujeitos; as métricas por
  sujeito saem do .npz de predições, como no agrupado da v2.

`collect_all()` junta o agrupado da v2 (hold-out cronológico e CV de 6 folds)
para a comparação entre protocolos.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from _v2_results import MODELS, PROJECT_ROOT, USERS, _subject_metrics, collect_v2

V3_ROOT = PROJECT_ROOT / "outputs" / "iterations" / "v3" / "metrics"

LOTO_N_FOLDS = 12


def collect_v3(root: Path = V3_ROOT) -> pd.DataFrame:
    rows = []
    for json_path in sorted(root.rglob("*.json")):
        if json_path.name.startswith("_"):
            continue
        data = json.loads(json_path.read_text(encoding="utf-8"))
        if data.get("iteration") != "v3":
            continue

        base = dict(
            iteration="v3", protocol=data["protocol"], variant=data["variant"], track="pooled",
            model=data["model_name"], best_epoch=data.get("best_epoch"),
            epochs_trained=data.get("epochs_trained"), train_seconds=data.get("train_seconds"),
        )
        overall = {
            "accuracy": data["test_accuracy"],
            "balanced_accuracy": data["balanced_accuracy"],
            "f1_macro": data["f1_macro"],
            "majority_baseline": data["majority_baseline"],
            "cohen_kappa": data.get("cohen_kappa"),
        }
        if "trial_level" in data:
            overall["trial_accuracy"] = data["trial_level"]["trial_accuracy_mean_proba"]
            overall["trial_accuracy_vote"] = data["trial_level"]["trial_accuracy_vote"]
            overall["n_trials"] = data["trial_level"]["n_trials"]

        if data["protocol"] == "loso":
            rows.append({**base, "rep": f"seed_{data['seed']}",
                         "subject": data["test_subject"], **overall})
            continue

        rep = f"fold_{data['fold']}"
        rows.append({**base, "rep": rep, "subject": "ALL", **overall})
        npz_path = json_path.with_name(f"{data['model_name']}_predictions.npz")
        if npz_path.exists():
            npz = np.load(npz_path)
            users = npz["users"]
            for user in np.unique(users):
                mask = users == user
                rows.append({**base, "rep": rep, "subject": str(user), **_subject_metrics(
                    npz["y_true"][mask], npz["y_pred"][mask], npz["y_pred_proba"][mask],
                    users[mask], npz["blocks"][mask],
                )})
    return pd.DataFrame(rows)


def collect_all() -> pd.DataFrame:
    """v3 + agrupado da v2, para comparar os quatro protocolos."""
    v2 = collect_v2()
    if not v2.empty:
        v2 = v2[v2.track == "pooled"]
    frames = [f for f in (v2, collect_v3()) if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_predictions(protocol: str, variant: str, model: str, subject: str,
                     seed: int = 42) -> dict | None:
    """Predições de um sujeito: o teste inteiro (LOSO) ou os 12 folds concatenados (LOTO)."""
    base = V3_ROOT / protocol / variant / "pooled"

    if protocol == "loso":
        paths = [base / f"test_{subject}" / f"seed_{seed}" / f"{model}_predictions.npz"]
    else:
        paths = [base / f"fold_{k}" / f"seed_{seed}" / f"{model}_predictions.npz"
                 for k in range(LOTO_N_FOLDS)]

    parts = []
    for path in paths:
        if not path.exists():
            continue
        npz = np.load(path)
        mask = npz["users"] == subject
        parts.append({k: npz[k][mask] for k in npz.files})

    if not parts:
        return None
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


__all__ = ["MODELS", "USERS", "V3_ROOT", "collect_v3", "collect_all", "load_predictions"]
