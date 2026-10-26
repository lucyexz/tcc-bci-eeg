"""PARTE 1 — modelo agrupado com split cronológico por sujeito.

Re-roda as 4 arquiteturas trocando o split aleatório estratificado pelo split
cronológico, para medir quanto da acurácia atual era vazamento temporal.

As saídas vão para pastas próprias; os artefatos do split aleatório
(outputs/metrics/*.json, outputs/models/*.keras, data/processed/*.npy)
permanecem intactos como baseline "antes".

Uso:
    python experiments/run_pooled.py                  # 4 modelos
    python experiments/run_pooled.py --models cnn     # só a CNN
    python experiments/run_pooled.py --smoke          # 2 épocas, valida o pipeline
"""

import argparse
from dataclasses import replace

from _common import (
    DATA_DIR,
    METRICS_DIR,
    MODELS_DIR,
    assert_no_leakage,
    banner,
    format_duration,
    resolve_models,
    save_arrays,
    save_report,
    train_and_export,
)

from src.preprocessing.base_preprocessing import (
    BasePreprocessingConfig,
    prepare_base_data,
)

# iqr_multiplier=1.5 replica o pré-processamento do notebook 02, que gerou o
# baseline aleatório. Assim a única variável que muda na comparação
# "antes vs depois" é a estratégia de split.
POOLED_CONFIG = BasePreprocessingConfig(
    test_size=0.2,
    random_state=42,
    remove_duplicates=True,
    handle_missing=True,
    imputation_strategy="median",
    outlier_strategy="clip_iqr",
    iqr_multiplier=1.5,
    scale_data=True,
    split_strategy="chronological",
    validation_size=0.2,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        help="Subconjunto de modelos (cnn lstm gru transformer). Padrão: todos.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Pula modelos que já têm JSON de métricas gravado.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Roda 2 épocas e grava em pastas descartáveis, só para validar o pipeline.",
    )
    parser.add_argument(
        "--split",
        choices=["chronological", "random"],
        default="chronological",
        help=(
            "Estratégia de split. 'random' roda o EXPERIMENTO DE CONTROLE: o mesmo "
            "caminho de código, o mesmo pré-processamento e a mesma validação "
            "explícita, mudando apenas o split. Serve para confirmar que a queda de "
            "acurácia vem do split e não de um erro de implementação."
        ),
    )
    args = parser.parse_args()

    models = resolve_models(args.models)
    config = replace(POOLED_CONFIG, split_strategy=args.split)

    folder = "chronological" if args.split == "chronological" else "control_random"

    if args.smoke:
        metrics_dir = METRICS_DIR / "_smoke" / folder
        models_dir = MODELS_DIR / "_smoke" / folder
        data_dir = DATA_DIR / "_smoke" / folder
        epochs = 2
    else:
        metrics_dir = METRICS_DIR / folder
        models_dir = MODELS_DIR / folder
        data_dir = DATA_DIR / folder
        epochs = None

    if args.split == "chronological":
        banner("PARTE 1 — AGRUPADO, SPLIT CRONOLÓGICO POR SUJEITO")
    else:
        banner("CONTROLE — AGRUPADO, SPLIT ALEATÓRIO PELO MESMO CAMINHO DE CÓDIGO")

    print(f"Modelos  : {', '.join(models)}")
    print(f"Métricas : {metrics_dir}")
    print(f"Modelos  : {models_dir}")
    if args.smoke:
        print("MODO SMOKE: 2 épocas, resultados descartáveis")

    print("\n[1/2] Preparando os dados...", flush=True)
    prep = prepare_base_data(config)

    report = prep.preprocessing_report
    print(f"  treino     : {report['final_train_shape']}")
    print(f"  validação  : {report['final_val_shape']}")
    print(f"  teste      : {report['final_test_shape']}")
    print(f"  val/sujeito: {report['samples_per_user_val']}")

    assert_no_leakage(prep, config)
    save_arrays(prep, data_dir)
    save_report(prep, config, metrics_dir)

    print(f"\n[2/2] Treinando {len(models)} modelo(s)...", flush=True)
    results: dict[str, tuple[float, float]] = {}

    for position, model_name in enumerate(models, start=1):
        if args.skip_existing and (metrics_dir / f"{model_name}.json").exists():
            print(f"\n--- {model_name} ({position}/{len(models)}) — pulado, já existe")
            continue

        banner(f"{model_name.upper()} ({position}/{len(models)})")
        accuracy, elapsed = train_and_export(
            model_name,
            prep,
            metrics_dir=metrics_dir,
            models_dir=models_dir,
            epochs=epochs,
        )
        results[model_name] = (accuracy, elapsed)
        print(
            f"[resumo] {model_name}: acurácia {accuracy:.4f} "
            f"em {format_duration(elapsed)}",
            flush=True,
        )

    banner(f"RESUMO — AGRUPADO ({args.split})")
    for model_name, (accuracy, elapsed) in results.items():
        print(f"  {model_name:<12} {accuracy:.4f}   {format_duration(elapsed)}")

    if results:
        total = sum(elapsed for _, elapsed in results.values())
        print(f"\n  tempo total: {format_duration(total)}")


if __name__ == "__main__":
    main()
