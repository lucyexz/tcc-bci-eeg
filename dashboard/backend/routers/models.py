import json
from pathlib import Path
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/models", tags=["models"])

METRICS_DIR = Path(__file__).parent.parent.parent.parent / "outputs" / "metrics"
MODEL_NAMES = ["cnn", "lstm", "gru", "transformer"]


def _load(name: str) -> dict:
    path = METRICS_DIR / f"{name}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Model '{name}' not found")
    return json.loads(path.read_text(encoding="utf-8"))


@router.get(
    "",
    summary="Lista todos os modelos",
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": [
                        {
                            "name": "string",
                            "display_name": "string",
                            "test_accuracy": 0.0,
                            "test_loss": 0.0,
                            "trained_at": "string",
                        }
                    ]
                }
            }
        }
    },
)
def list_models():
    """Retorna `name`, `display_name`, `test_accuracy`, `test_loss` e `trained_at` de cada modelo disponível."""
    result = []
    for name in MODEL_NAMES:
        try:
            data = _load(name)
            result.append({
                "name": data["model_name"],
                "display_name": data["display_name"],
                "test_accuracy": data["test_accuracy"],
                "test_loss": data.get("test_loss"),
                "trained_at": data.get("trained_at"),
            })
        except HTTPException:
            pass
    return result


@router.get(
    "/comparison",
    summary="Comparação completa entre modelos",
    responses={
        200: {
            "content": {
                "application/json": {
                    "example": {
                        "models": [
                            {
                                "name": "string",
                                "display_name": "string",
                                "test_accuracy": 0.0,
                                "test_loss": 0.0,
                                "per_class_f1": {
                                    "Esquerda": 0.0,
                                    "Direita": 0.0,
                                    "Neutro": 0.0,
                                },
                                "macro_f1": 0.0,
                                "per_user_accuracy": {
                                    "user_a": 0.0,
                                    "user_b": 0.0,
                                    "user_c": 0.0,
                                    "user_d": 0.0,
                                    "user_e": 0.0,
                                },
                            }
                        ]
                    }
                }
            }
        }
    },
)
def comparison():
    """Retorna F1 por classe, F1 macro e acurácia por usuário de todos os modelos disponíveis."""
    models = []
    for name in MODEL_NAMES:
        try:
            data = _load(name)
            report = data.get("classification_report", {})
            per_class_f1 = {
                cls: report[cls]["f1-score"]
                for cls in report
                if cls not in ("accuracy", "macro avg", "weighted avg")
            }
            models.append({
                "name": data["model_name"],
                "display_name": data["display_name"],
                "test_accuracy": data["test_accuracy"],
                "test_loss": data.get("test_loss"),
                "per_class_f1": per_class_f1,
                "macro_f1": report.get("macro avg", {}).get("f1-score"),
                "per_user_accuracy": data.get("per_user_accuracy", {}),
            })
        except HTTPException:
            pass
    return {"models": models}


@router.get("/{name}", summary="Métricas detalhadas de um modelo")
def get_model(name: str):
    """
    Retorna o JSON completo do modelo especificado (`cnn`, `lstm`, `gru` ou `transformer`):
    histórico de treinamento por época, confusion matrix 3×3, classification report por classe
    e acurácia por usuário.
    """
    return _load(name)
