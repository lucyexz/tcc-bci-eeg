import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

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


def export_metrics(
    result: Any,
    model_name: str,
    display_name: str | None = None,
    users_test: np.ndarray | None = None,
    output_dir: Path | None = None,
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
    }

    output_path = out_dir / f"{model_name}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[export_metrics] Salvo: {output_path}")
    return output_path
