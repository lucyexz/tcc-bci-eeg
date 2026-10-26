import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

CLASS_NAMES = ["Esquerda", "Direita", "Neutro"]
_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "metrics"

_DISPLAY_NAMES = {
    "cnn": "CNN",
    "lstm": "Bi-LSTM",
    "gru": "Bi-GRU",
    "transformer": "Transformer",
}


def _to_python(obj: Any) -> Any:
    """Recursively convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_python(v) for v in obj]
    return obj


def _align_tail(values: np.ndarray | None, n: int) -> np.ndarray | None:
    """Alinha um array ao y_test do resultado (a CNN com janelas corta o início)."""
    if values is None:
        return None
    return np.asarray(values)[len(values) - n:]


def trial_level_accuracy(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    users: np.ndarray,
    blocks: np.ndarray,
) -> dict[str, float | int]:
    """Acurácia por trial: uma decisão por bloco contíguo de rótulo.

    Em uso real a interface decide por trial, não por amostra isolada. Duas
    regras de agregação são reportadas: voto majoritário das predições e
    média das probabilidades.
    """
    keys = np.array([f"{u}#{b}" for u, b in zip(users, blocks)])
    y_pred = y_pred_proba.argmax(axis=1)

    vote_hits = []
    mean_hits = []
    for key in np.unique(keys):
        mask = keys == key
        true_label = np.bincount(y_true[mask]).argmax()
        vote = np.bincount(y_pred[mask], minlength=3).argmax()
        mean = y_pred_proba[mask].mean(axis=0).argmax()
        vote_hits.append(vote == true_label)
        mean_hits.append(mean == true_label)

    return {
        "n_trials": int(len(vote_hits)),
        "trial_accuracy_vote": float(np.mean(vote_hits)),
        "trial_accuracy_mean_proba": float(np.mean(mean_hits)),
    }


def export_metrics(
    result: Any,
    model_name: str,
    display_name: str | None = None,
    users_test: np.ndarray | None = None,
    output_dir: Path | None = None,
    blocks_test: np.ndarray | None = None,
    positions_test: np.ndarray | None = None,
    metadata: dict | None = None,
    save_predictions: bool = False,
) -> Path:
    """Export model metrics to outputs/metrics/{model_name}.json.

    Args:
        result: Any of CNNResult, LSTMResult, GRUResult, TransformerResult.
        model_name: Lowercase key, e.g. "cnn", "lstm", "gru", "transformer".
        display_name: Human-readable label shown in the dashboard. Defaults to
            a built-in mapping based on model_name.
        users_test: Array of user identifiers aligned with the original y_test.
            Used to compute per-user accuracy.
        output_dir: Override output directory (defaults to outputs/metrics/).
        blocks_test: Trial block id of each test sample (enables trial-level
            accuracy).
        positions_test: Temporal position of each test sample within its
            subject (saved with the predictions, for timeline plots).
        metadata: Extra fields merged into the JSON (feature set, protocol,
            fold, seed...).
        save_predictions: Also write `<model_name>_predictions.npz` with
            y_true, y_pred, probabilities, users, blocks and positions.

    The fields written by the original version are unchanged, so the
    dashboard keeps reading these JSONs as before.

    Returns:
        Path to the written JSON file.
    """
    if display_name is None:
        display_name = _DISPLAY_NAMES.get(model_name, model_name.upper())

    out_dir = Path(output_dir) if output_dir is not None else _OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(result.y_test, result.y_pred)
    report = classification_report(
        result.y_test,
        result.y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
    )

    per_user_accuracy: dict[str, float] = {}
    if users_test is not None:
        # Align users_test with result.y_test in case CNN windowing trimmed the front
        n = len(result.y_test)
        offset = len(users_test) - n
        users_aligned = users_test[offset:]
        for user in np.unique(users_aligned):
            mask = users_aligned == user
            per_user_accuracy[str(user)] = float(
                (result.y_pred[mask] == result.y_test[mask]).mean()
            )

    y_true = np.asarray(result.y_test).astype(int)
    y_pred = np.asarray(result.y_pred).astype(int)
    n = len(y_true)

    users_aligned_all = _align_tail(users_test, n)
    blocks_aligned = _align_tail(blocks_test, n)
    positions_aligned = _align_tail(positions_test, n)

    counts = np.bincount(y_true, minlength=3)
    history = result.history or {}
    best_epoch = (
        int(np.argmin(history["val_loss"])) + 1
        if history.get("val_loss")
        else None
    )

    extended = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "majority_baseline": float(counts.max() / counts.sum()),
        "best_epoch": best_epoch,
        "epochs_trained": len(history.get("loss", [])),
    }

    if users_aligned_all is not None and blocks_aligned is not None:
        extended["trial_level"] = trial_level_accuracy(
            y_true, np.asarray(result.y_pred_proba), users_aligned_all, blocks_aligned
        )

    payload = {
        "model_name": model_name,
        "display_name": display_name,
        "test_accuracy": float(result.test_accuracy),
        "test_loss": float(result.test_loss),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "config": _to_python(asdict(result.config)),
        "history": _to_python(result.history),
        "confusion_matrix": _to_python(cm),
        "classification_report": _to_python(report),
        "per_user_accuracy": per_user_accuracy,
        **extended,
        **_to_python(metadata or {}),
    }

    output_path = out_dir / f"{model_name}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[export_metrics] Salvo: {output_path}")

    if save_predictions:
        npz_path = out_dir / f"{model_name}_predictions.npz"
        arrays = {
            "y_true": y_true,
            "y_pred": y_pred,
            "y_pred_proba": np.asarray(result.y_pred_proba, dtype=np.float32),
        }
        if users_aligned_all is not None:
            arrays["users"] = users_aligned_all.astype(str)
        if blocks_aligned is not None:
            arrays["blocks"] = blocks_aligned.astype(int)
        if positions_aligned is not None:
            arrays["positions"] = positions_aligned.astype(int)
        np.savez_compressed(npz_path, **arrays)

    return output_path
