"""Iteração v2 — efeito das novas representações de features.

Percorre a grade

    protocolo × variante de features × track × [sujeito] × [fold] × semente × modelo

mantendo arquiteturas e hiperparâmetros da v1. Só muda a representação dos
dados (ver src/features/spectral_features.py) e, nas variantes `*_subjnorm`, a
normalização por sujeito.

Protocolos:
    chronological — hold-out cronológico 80/20, idêntico ao da v1. Roda com
                    várias sementes para medir a variância de inicialização.
    blocked_cv    — validação cruzada por blocos de trials (6 folds de 6
                    blocos, purge de 40 amostras nas fronteiras). Todo trial
                    é testado exatamente uma vez.

Variantes de features:
    raw               — potência bruta (v1; serve de controle de equivalência)
    log               — log10 da potência
    log_subjnorm      — log10 + IQR/StandardScaler por sujeito (só agrupado)
    log_rel           — log10 + potência relativa (2 mapas; só personalizado)
    log_rel_subjnorm  — log10 + potência relativa + norm. por sujeito (só agrupado)

No track personalizado, normalizar "por sujeito" é o mesmo que normalizar
globalmente (só existe um sujeito), por isso as variantes `*_subjnorm` não
são rodadas lá.

Saídas (nada da v1 é sobrescrito):
    outputs/iterations/v2/metrics/<protocolo>/<variante>/<track>/[<sujeito>/]<seed_S|fold_K>/<modelo>.json
    (+ <modelo>_predictions.npz e _preprocessing.json ao lado)
    outputs/iterations/v2/models/... (só cronológico, semente 42)

Uso:
    python experiments/run_v2.py --smoke                                # valida a grade inteira, 2 épocas
    python experiments/run_v2.py --protocols chronological --seeds 42   # etapa 1
    python experiments/run_v2.py --protocols chronological --seeds 43 44
    python experiments/run_v2.py --protocols blocked_cv                 # etapa 3 (longa)
    (sempre retomável com --skip-existing)
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

from _common import (
    MODEL_NAMES,
    PROJECT_ROOT,
    assert_no_leakage,
    banner,
    format_duration,
    resolve_models,
    resolve_users,
    save_report,
    train_and_export,
)

from src.preprocessing.base_preprocessing import (
    BasePreprocessingConfig,
    prepare_base_data,
)

V2_ROOT = PROJECT_ROOT / "outputs" / "iterations" / "v2"

PROTOCOLS = ["chronological", "blocked_cv"]
TRACKS = ["pooled", "personalized"]


@dataclass(frozen=True)
class Variant:
    feature_set: str
    normalization: str
    tracks: tuple[str, ...]


VARIANTS: dict[str, Variant] = {
    "raw": Variant("raw", "global", ("pooled", "personalized")),
    "log": Variant("log", "global", ("pooled", "personalized")),
    "log_subjnorm": Variant("log", "per_subject", ("pooled",)),
    "log_rel": Variant("log_rel", "global", ("personalized",)),
    "log_rel_subjnorm": Variant("log_rel", "per_subject", ("pooled",)),
}

DEFAULT_SEEDS = [42]
N_FOLDS = 6
PURGE_SAMPLES = 40


def build_config(
    protocol: str,
    variant: Variant,
    subject: str | None,
    fold: int,
) -> BasePreprocessingConfig:
    """Mesmo pré-processamento da v1 (iqr 1.5, validação 20%), mais as
    opções de features/normalização e, na CV, o fold."""
    return BasePreprocessingConfig(
        test_size=0.2,
        random_state=42,
        remove_duplicates=True,
        handle_missing=True,
        imputation_strategy="median",
        outlier_strategy="clip_iqr",
        iqr_multiplier=1.5,
        scale_data=True,
        split_strategy=protocol,
        validation_size=0.2,
        subject=subject,
        feature_set=variant.feature_set,
        normalization=variant.normalization,
        cv_n_folds=N_FOLDS,
        cv_fold=fold,
        purge_samples=PURGE_SAMPLES,
    )


def iter_jobs(args):
    """Gera (protocolo, variante, track, sujeito, fold, sementes) na ordem de execução."""
    for protocol in args.protocols:
        folds = args.folds if protocol == "blocked_cv" else [None]
        # Na CV a variância vem dos folds; várias sementes só no cronológico.
        seeds = args.seeds if protocol == "chronological" else [42]

        for variant_name in args.variants:
            variant = VARIANTS[variant_name]
            for track in args.tracks:
                if track not in variant.tracks:
                    continue
                subjects = [None] if track == "pooled" else args.users
                for subject in subjects:
                    for fold in folds:
                        yield protocol, variant_name, track, subject, fold, seeds


def run_dir(root: Path, protocol, variant_name, track, subject, leaf) -> Path:
    path = root / protocol / variant_name / track
    if subject is not None:
        path = path / subject
    return path / leaf


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--protocols", nargs="+", choices=PROTOCOLS, default=PROTOCOLS)
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--tracks", nargs="+", choices=TRACKS, default=TRACKS)
    parser.add_argument("--users", nargs="+", help="Sujeitos do track personalizado. Padrão: todos.")
    parser.add_argument("--models", nargs="+", help=f"Subconjunto de {MODEL_NAMES}. Padrão: todos.")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                        help="Sementes do protocolo cronológico. Padrão: 42.")
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(N_FOLDS)),
                        help="Folds da CV a rodar. Padrão: todos.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Pula combinações que já têm JSON gravado (retoma runs).")
    parser.add_argument("--smoke", action="store_true",
                        help="2 épocas, pastas descartáveis: valida a grade inteira.")
    args = parser.parse_args()

    args.users = resolve_users(args.users)
    models = resolve_models(args.models)

    root = V2_ROOT / "_smoke" if args.smoke else V2_ROOT
    metrics_root = root / "metrics"
    models_root = root / "models"
    epochs = 2 if args.smoke else None

    jobs = list(iter_jobs(args))
    n_trainings = sum(len(seeds) for *_, seeds in jobs) * len(models)

    banner("ITERAÇÃO v2 — FEATURES NOVAS")
    print(f"Protocolos : {', '.join(args.protocols)}")
    print(f"Variantes  : {', '.join(args.variants)}")
    print(f"Tracks     : {', '.join(args.tracks)}")
    print(f"Modelos    : {', '.join(models)}")
    print(f"Sementes   : {args.seeds} (cronológico)  |  folds: {args.folds} (CV)")
    print(f"Treinos    : {n_trainings}")
    print(f"Saída      : {metrics_root}")
    if args.smoke:
        print("MODO SMOKE: 2 épocas, resultados descartáveis")

    started = time.perf_counter()
    done = 0
    summary: list[tuple[str, float, float]] = []

    for protocol, variant_name, track, subject, fold, seeds in jobs:
        variant = VARIANTS[variant_name]
        leaves = (
            [f"fold_{fold}"] if protocol == "blocked_cv"
            else [f"seed_{seed}" for seed in seeds]
        )
        pending = [
            (leaf, seed)
            for leaf, seed in zip(leaves, seeds)
            if not (
                args.skip_existing
                and all(
                    (run_dir(metrics_root, protocol, variant_name, track, subject, leaf)
                     / f"{m}.json").exists()
                    for m in models
                )
            )
        ]

        label = f"{protocol}/{variant_name}/{track}" + (f"/{subject}" if subject else "")
        if fold is not None:
            label += f"/fold_{fold}"

        if not pending:
            print(f"\n--- {label}: já existe, pulado")
            done += len(leaves) * len(models)
            continue

        banner(label)
        config = build_config(protocol, variant, subject, fold if fold is not None else 0)
        prep = prepare_base_data(config)
        report = prep.preprocessing_report
        print(f"  features   : {variant.feature_set} ({report['n_features']} colunas), "
              f"normalização {variant.normalization}")
        print(f"  treino/val/teste: {len(prep.y_train)}/{len(prep.y_val)}/{len(prep.y_test)}"
              f"  purge: {report['purged_rows']}")
        print(f"  classes/teste: {report['class_distribution_test']}")
        assert_no_leakage(prep, config)

        for leaf, seed in pending:
            metrics_dir = run_dir(metrics_root, protocol, variant_name, track, subject, leaf)
            models_dir = run_dir(models_root, protocol, variant_name, track, subject, leaf)
            save_report(prep, config, metrics_dir)

            for model_name in models:
                if args.skip_existing and (metrics_dir / f"{model_name}.json").exists():
                    done += 1
                    continue

                done += 1
                print(f"\n--- [{done}/{n_trainings}] {label} / {leaf} / {model_name}", flush=True)
                accuracy, elapsed = train_and_export(
                    model_name,
                    prep,
                    metrics_dir=metrics_dir,
                    models_dir=models_dir,
                    epochs=epochs,
                    random_state=seed,
                    verbose=2,
                    metadata={
                        "iteration": "v2",
                        "protocol": protocol,
                        "variant": variant_name,
                        "feature_set": variant.feature_set,
                        "normalization": variant.normalization,
                        "track": track,
                        "subject": subject,
                        "fold": fold,
                        "seed": seed,
                    },
                    save_model=(protocol == "chronological" and seed == 42),
                    save_predictions=True,
                )
                summary.append((f"{label}/{leaf}/{model_name}", accuracy, elapsed))
                total = time.perf_counter() - started
                print(f"[resumo] acurácia {accuracy:.4f} em {format_duration(elapsed)} "
                      f"(decorrido {format_duration(total)})", flush=True)

    banner("RESUMO — v2")
    for name, accuracy, elapsed in summary:
        print(f"  {name:<70} {accuracy:.4f}  {format_duration(elapsed)}")
    print(f"\n  tempo total: {format_duration(time.perf_counter() - started)}")


if __name__ == "__main__":
    main()
