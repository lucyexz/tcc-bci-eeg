"""Diagnóstico: a posição temporal da validação muda o resultado?

No fold 5 da CV por blocos (os 6 últimos trials, praticamente o mesmo trecho
de teste do hold-out cronológico), a validação padrão é o fold "seguinte"
circular — o fold 0, ou seja, os primeiros trials da gravação, longe do teste.
No hold-out cronológico, a validação fica imediatamente antes do teste.

Este script refaz só o fold 5 com a validação no fold ANTERIOR (adjacente ao
teste) e compara com a CV padrão. Se o resultado mudar muito, a seleção de
época pelo EarlyStopping depende da proximidade temporal entre validação e
teste — uma consequência direta da deriva entre trials.

Saída: outputs/iterations/v2/diagnostics/val_previous/...

Uso:
    python experiments/diagnostico_validacao.py
"""

from dataclasses import replace

from _common import assert_no_leakage, banner, train_and_export
from run_v2 import V2_ROOT, VARIANTS, build_config

from src.preprocessing.base_preprocessing import prepare_base_data

FOLD = 5
JOBS = [
    ("pooled", "raw", None),
    ("pooled", "log_subjnorm", None),
    *[("personalized", "log", user) for user in ["user_a", "user_b", "user_c", "user_d", "user_e"]],
]
MODELS = ["cnn", "lstm", "gru", "transformer"]
ROOT = V2_ROOT / "diagnostics" / "val_previous" / "metrics"


def main() -> None:
    for track, variant_name, subject in JOBS:
        label = f"{variant_name}/{track}" + (f"/{subject}" if subject else "")
        out = ROOT / "blocked_cv" / variant_name / track
        if subject:
            out = out / subject
        out = out / f"fold_{FOLD}"

        if all((out / f"{m}.json").exists() for m in MODELS):
            print(f"--- {label}: já existe")
            continue

        banner(f"DIAGNÓSTICO validação anterior — {label}")
        config = replace(build_config("blocked_cv", VARIANTS[variant_name], subject, FOLD),
                         cv_val_position="previous")
        prep = prepare_base_data(config)
        print(f"  validação = fold {prep.preprocessing_report['cv_val_fold']}")
        assert_no_leakage(prep, config)

        for model in MODELS:
            if (out / f"{model}.json").exists():
                continue
            train_and_export(
                model, prep, metrics_dir=out, models_dir=out, random_state=42, verbose=2,
                metadata={"iteration": "v2-diagnostic", "protocol": "blocked_cv",
                          "variant": variant_name, "track": track, "subject": subject,
                          "fold": FOLD, "seed": 42, "cv_val_position": "previous"},
                save_model=False, save_predictions=True,
            )


if __name__ == "__main__":
    main()
