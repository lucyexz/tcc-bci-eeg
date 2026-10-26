"""PARTE 2 — track personalizado: um modelo por sujeito.

Para cada sujeito, filtra o dataset antes de qualquer split, aplica o mesmo
split cronológico do track agrupado e treina as 4 arquiteturas só com os dados
daquele sujeito. Isso permite comparar o paradigma generalista (um modelo para
todos) contra o personalizado (um modelo por pessoa).

O split e o pré-processamento são idênticos aos do run_pooled.py — só muda
quais sujeitos entram nos dados. É isso que torna a comparação legítima.

Uso:
    python experiments/run_personalized.py --users user_c   # prova de conceito
    python experiments/run_personalized.py                  # os 5 sujeitos
    python experiments/run_personalized.py --smoke --users user_c
"""

import argparse

from _common import (
    DATA_DIR,
    METRICS_DIR,
    MODELS_DIR,
    assert_no_leakage,
    banner,
    format_duration,
    resolve_models,
    resolve_users,
    save_arrays,
    save_report,
    train_and_export,
)

from src.preprocessing.base_preprocessing import (
    BasePreprocessingConfig,
    prepare_base_data,
)


def build_config(subject: str) -> BasePreprocessingConfig:
    """Mesma config do track agrupado, com o filtro de sujeito ligado."""
    return BasePreprocessingConfig(
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
        subject=subject,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--users",
        nargs="+",
        help="Sujeitos a treinar (user_a ... user_e). Padrão: todos.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Subconjunto de modelos (cnn lstm gru transformer). Padrão: todos.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Pula combinações sujeito/modelo que já têm JSON gravado.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Roda 2 épocas e grava em pastas descartáveis, só para validar o pipeline.",
    )
    args = parser.parse_args()

    users = resolve_users(args.users)
    models = resolve_models(args.models)

    if args.smoke:
        metrics_root = METRICS_DIR / "_smoke" / "personalized"
        models_root = MODELS_DIR / "_smoke" / "personalized"
        data_root = DATA_DIR / "_smoke" / "personalized"
        epochs = 2
    else:
        metrics_root = METRICS_DIR / "personalized"
        models_root = MODELS_DIR / "personalized"
        data_root = DATA_DIR / "personalized"
        epochs = None

    banner("PARTE 2 — PERSONALIZADO, UM MODELO POR SUJEITO")
    print(f"Sujeitos: {', '.join(users)}")
    print(f"Modelos : {', '.join(models)}")
    print(f"Treinos : {len(users) * len(models)}")
    print(f"Métricas: {metrics_root}/<sujeito>/")
    if args.smoke:
        print("MODO SMOKE: 2 épocas, resultados descartáveis")

    results: dict[str, dict[str, tuple[float, float]]] = {}

    for user in users:
        banner(f"SUJEITO {user}")

        metrics_dir = metrics_root / user
        models_dir = models_root / user

        config = build_config(user)
        prep = prepare_base_data(config)

        report = prep.preprocessing_report
        print(f"  amostras   : {report['shape_after_cleaning'][0]}")
        print(f"  treino     : {report['final_train_shape']}")
        print(f"  validação  : {report['final_val_shape']}")
        print(f"  teste      : {report['final_test_shape']}")
        print(f"  classes/teste: {report['class_distribution_test']}")

        assert_no_leakage(prep, config)
        save_arrays(prep, data_root / user)
        save_report(prep, config, metrics_dir)

        results[user] = {}

        for position, model_name in enumerate(models, start=1):
            if args.skip_existing and (metrics_dir / f"{model_name}.json").exists():
                print(f"\n--- {user}/{model_name} — pulado, já existe")
                continue

            print(f"\n--- {user} / {model_name} ({position}/{len(models)})", flush=True)
            accuracy, elapsed = train_and_export(
                model_name,
                prep,
                metrics_dir=metrics_dir,
                models_dir=models_dir,
                epochs=epochs,
            )
            results[user][model_name] = (accuracy, elapsed)
            print(
                f"[resumo] {user}/{model_name}: acurácia {accuracy:.4f} "
                f"em {format_duration(elapsed)}",
                flush=True,
            )

    banner("RESUMO — PERSONALIZADO")
    total = 0.0

    for user, per_model in results.items():
        for model_name, (accuracy, elapsed) in per_model.items():
            print(f"  {user:<10} {model_name:<12} {accuracy:.4f}   {format_duration(elapsed)}")
            total += elapsed

    if total:
        print(f"\n  tempo total: {format_duration(total)}")


if __name__ == "__main__":
    main()
