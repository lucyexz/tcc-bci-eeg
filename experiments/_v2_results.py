"""Coleta os resultados da v1 (snapshot) e da v2 numa tabela longa.

Unidade: uma linha por (iteração, protocolo, variante, track, modelo, sujeito,
repetição). A repetição é a semente (cronológico) ou o fold (CV).

As métricas por sujeito da v2 são recalculadas a partir dos .npz de
predição, o que dá acurácia balanceada e acurácia por trial também para o
modelo agrupado. Da v1 só existem os JSONs: o agregado sai da matriz de
confusão e, no agrupado, por sujeito só há a acurácia simples.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.export_metrics import trial_level_accuracy  # noqa: E402

V1_ROOT = PROJECT_ROOT / "outputs" / "iterations" / "v1" / "metrics"
V2_ROOT = PROJECT_ROOT / "outputs" / "iterations" / "v2" / "metrics"

MODELS = ["cnn", "lstm", "gru", "transformer"]
USERS = ["user_a", "user_b", "user_c", "user_d", "user_e"]


def _metrics_from_cm(cm: np.ndarray) -> dict:
    cm = np.asarray(cm, dtype=float)
    support = cm.sum(axis=1)
    recall = np.divide(np.diag(cm), support, out=np.zeros(3), where=support > 0)
    precision = np.divide(np.diag(cm), cm.sum(axis=0), out=np.zeros(3), where=cm.sum(axis=0) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(3), where=(precision + recall) > 0)
    return {
        "accuracy": float(np.trace(cm) / cm.sum()),
        "balanced_accuracy": float(recall.mean()),
        "f1_macro": float(f1.mean()),
        "majority_baseline": float(support.max() / support.sum()),
    }


def _subject_metrics(y_true, y_pred, proba, users, blocks) -> dict:
    counts = np.bincount(y_true, minlength=3)
    out = {
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0)),
        "majority_baseline": float(counts.max() / counts.sum()),
        "n_test": int(len(y_true)),
    }
    if blocks is not None:
        trial = trial_level_accuracy(y_true, proba, users, blocks)
        out["trial_accuracy"] = trial["trial_accuracy_mean_proba"]
        out["trial_accuracy_vote"] = trial["trial_accuracy_vote"]
        out["n_trials"] = trial["n_trials"]
    return out


def collect_v1() -> pd.DataFrame:
    rows = []
    for model in MODELS:
        path = V1_ROOT / "chronological" / f"{model}.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            base = dict(iteration="v1", protocol="chronological", variant="raw", track="pooled",
                        model=model, rep="seed_42", best_epoch=None)
            rows.append({**base, "subject": "ALL", **_metrics_from_cm(data["confusion_matrix"])})
            for user, acc in data.get("per_user_accuracy", {}).items():
                rows.append({**base, "subject": user, "accuracy": acc})

        for user in USERS:
            path = V1_ROOT / "personalized" / user / f"{model}.json"
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                history = data.get("history", {})
                rows.append(dict(
                    iteration="v1", protocol="chronological", variant="raw", track="personalized",
                    model=model, rep="seed_42", subject=user,
                    best_epoch=int(np.argmin(history["val_loss"])) + 1 if history.get("val_loss") else None,
                    **_metrics_from_cm(data["confusion_matrix"]),
                ))
    return pd.DataFrame(rows)


def collect_v2(root: Path = V2_ROOT) -> pd.DataFrame:
    rows = []
    for json_path in sorted(root.rglob("*.json")):
        if json_path.name.startswith("_"):
            continue
        data = json.loads(json_path.read_text(encoding="utf-8"))
        if data.get("iteration") != "v2":
            continue
        npz_path = json_path.with_name(f"{data['model_name']}_predictions.npz")
        rep = (f"fold_{data['fold']}" if data["protocol"] == "blocked_cv" else f"seed_{data['seed']}")
        base = dict(
            iteration="v2", protocol=data["protocol"], variant=data["variant"], track=data["track"],
            model=data["model_name"], rep=rep, best_epoch=data.get("best_epoch"),
            train_seconds=data.get("train_seconds"),
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

        if data["track"] == "pooled":
            rows.append({**base, "subject": "ALL", **overall})
            if npz_path.exists():
                npz = np.load(npz_path)
                users = npz["users"]
                for user in np.unique(users):
                    mask = users == user
                    rows.append({**base, "subject": str(user), **_subject_metrics(
                        npz["y_true"][mask], npz["y_pred"][mask], npz["y_pred_proba"][mask],
                        users[mask], npz["blocks"][mask] if "blocks" in npz else None,
                    )})
        else:
            rows.append({**base, "subject": data["subject"], **overall})
    return pd.DataFrame(rows)


def collect_all() -> pd.DataFrame:
    frames = [collect_v1(), collect_v2()]
    return pd.concat([f for f in frames if not f.empty], ignore_index=True)


def load_predictions(protocol: str, variant: str, track: str, model: str,
                     subject: str | None = None, rep: str = "seed_42") -> dict | None:
    path = V2_ROOT / protocol / variant / track
    if subject is not None and track == "personalized":
        path = path / subject
    path = path / rep / f"{model}_predictions.npz"
    if not path.exists():
        return None
    npz = np.load(path)
    data = {k: npz[k] for k in npz.files}
    if subject is not None and track == "pooled":
        mask = data["users"] == subject
        data = {k: v[mask] for k, v in data.items()}
    return data


def load_cv_predictions(variant: str, track: str, model: str, subject: str) -> dict | None:
    """Concatena as predições dos 6 folds: cobre os 36 trials do sujeito."""
    parts = [load_predictions("blocked_cv", variant, track, model, subject, f"fold_{k}") for k in range(6)]
    parts = [p for p in parts if p is not None]
    if not parts:
        return None
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
