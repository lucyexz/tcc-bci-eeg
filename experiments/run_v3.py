"""Iteração v3 — LOSO e LOTO para o modelo agrupado (generalista).

A v3 não muda features, arquiteturas nem hiperparâmetros: muda só o
protocolo de avaliação do modelo agrupado. Os modelos personalizados ficam de
fora (com um único sujeito não existe "sujeito deixado de fora").

Percorre a grade

    protocolo × variante de features × fold × semente × modelo

Protocolos:
    loso — leave-one-subject-out. Um sujeito inteiro é o teste e nunca
           aparece no treino; o sujeito seguinte (circular) é a validação do
           EarlyStopping; os 3 restantes são treino. Responde "funciona para
           uma pessoa nova?". 5 folds.
    loto — leave-one-trial-out. Um ciclo Esquerda → Direita → Neutro (1 trial
           de cada classe) de cada sujeito é o teste; o ciclo seguinte é a
           validação; purge de 40 amostras nas fronteiras. É a CV por blocos
           da v2 com 12 folds em vez de 6. 12 folds.

Variantes (as do agrupado na v2): raw, log, log_subjnorm, log_rel_subjnorm.

No LOSO, as variantes `*_subjnorm` ajustam IQR/StandardScaler do sujeito de
teste (e do de validação) nas próprias features dele, SEM rótulos: é o cenário
"usuário novo com dados não rotulados". As variantes `raw` e `log` são o
cenário "usuário novo sem calibração nenhuma".

Saídas (v1 e v2 não são tocadas):
    outputs/iterations/v3/metrics/<protocolo>/<variante>/pooled/<fold>/seed_S/<modelo>.json
    (+ <modelo>_predictions.npz e _preprocessing.json ao lado)
    outputs/iterations/v3/models/... (só LOSO, semente 42)

Uso:
    python experiments/run_v3.py --estimate                 # só projeta o tempo, não treina
    python experiments/run_v3.py --smoke --folds 0 1        # valida a grade, 2 épocas
    python experiments/run_v3.py --protocols loso
    python experiments/run_v3.py --protocols loto
    (sempre retomável com --skip-existing)
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from _common import (
    MODEL_NAMES,
    PROJECT_ROOT,
    USERS,
    assert_no_leakage,
    banner,
    format_duration,
    resolve_models,
    save_report,
    train_and_export,
)
from run_v2 import PURGE_SAMPLES, V2_ROOT, VARIANTS, Variant

from src.preprocessing.base_preprocessing import (
    BasePreprocessingConfig,
    prepare_base_data,
)

V3_ROOT = PROJECT_ROOT / "outputs" / "iterations" / "v3"

PROTOCOLS = ["loso", "loto"]
POOLED_VARIANTS = [name for name, v in VARIANTS.items() if "pooled" in v.tracks]

LOTO_N_FOLDS = 12
N_FOLDS = {"loso": len(USERS), "loto": LOTO_N_FOLDS}
DEFAULT_SEEDS = [42]


def fold_name(protocol: str, fold: int) -> str:
    return f"test_{USERS[fold]}" if protocol == "loso" else f"fold_{fold}"


def build_config(protocol: str, variant: Variant, fold: int) -> BasePreprocessingConfig:
    """Mesmo pré-processamento da v1/v2 (iqr 1.5); só o split muda."""
    common = dict(
        random_state=42,
        remove_duplicates=True,
        handle_missing=True,
        imputation_strategy="median",
        outlier_strategy="clip_iqr",
        iqr_multiplier=1.5,
        scale_data=True,
        validation_size=0.2,
        feature_set=variant.feature_set,
        normalization=variant.normalization,
    )

    if protocol == "loso":
        return BasePreprocessingConfig(
            split_strategy="loso",
            loso_test_subject=USERS[fold],
            **common,
        )

    return BasePreprocessingConfig(
        split_strategy="blocked_cv",
        cv_n_folds=LOTO_N_FOLDS,
        cv_fold=fold,
        purge_samples=PURGE_SAMPLES,
        **common,
    )


def assert_loto_fold(prep) -> None:
    """Cada fold do LOTO testa exatamente 1 trial de cada classe por sujeito."""
    for user in np.unique(prep.users_test):
        mask = prep.users_test == user
        n_blocks = len(np.unique(prep.blocks_test[mask]))
        classes = sorted(np.unique(prep.y_test[mask]).tolist())
        assert n_blocks == 3 and classes == [0, 1, 2], (
            f"LOTO: {user} deveria ter 3 trials (1 por classe) no teste, "
            f"encontrado {n_blocks} trials e classes {classes}"
        )
    print("  [ok] LOTO: 1 trial de cada classe por sujeito no teste")


def iter_jobs(args):
    """Gera (protocolo, variante, fold, sementes) na ordem de execução."""
    for protocol in args.protocols:
        folds = [f for f in args.folds if f < N_FOLDS[protocol]]
        # No LOTO a variância vem dos 12 folds; várias sementes só no LOSO,
        # que tem apenas 5.
        seeds = args.seeds if protocol == "loso" else [42]

        for variant_name in args.variants:
            for fold in folds:
                yield protocol, variant_name, fold, seeds


def run_dir(root: Path, protocol: str, variant_name: str, fold: int, seed: int) -> Path:
    return root / protocol / variant_name / "pooled" / fold_name(protocol, fold) / f"seed_{seed}"


def v2_seconds_per_row() -> dict[tuple[str, str], float]:
    """Segundos de treino por linha de treino, por (variante, modelo).

    Medido nos treinos do agrupado da v2, que usam as mesmas redes e a mesma
    máquina. O EarlyStopping para quase sempre no mínimo (paciência + 1
    épocas), então o tempo é aproximadamente proporcional ao nº de linhas.
    """
    seconds = defaultdict(list)
    rows = defaultdict(list)

    for json_path in (V2_ROOT / "metrics").rglob("*.json"):
        if json_path.name.startswith("_"):
            continue
        data = json.loads(json_path.read_text(encoding="utf-8"))
        if data.get("track") != "pooled" or not data.get("train_seconds"):
            continue
        report_path = json_path.with_name("_preprocessing.json")
        if not report_path.exists():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        key = (data["variant"], data["model_name"])
        seconds[key].append(data["train_seconds"])
        rows[key].append(report["preprocessing_report"]["final_train_shape"][0])

    return {key: sum(seconds[key]) / sum(rows[key]) for key in seconds}


def estimate(args, jobs, models) -> None:
    rates = v2_seconds_per_row()
    if not rates:
        raise SystemExit(
            f"Sem treinos da v2 em {V2_ROOT / 'metrics'} para servir de base."
        )

    banner("ESTIMATIVA DE TEMPO — v3 (nenhum treino é feito)")

    # O nº de linhas de treino não depende da variante: um preparo por fold.
    train_rows: dict[tuple[str, int], int] = {}
    for protocol, _, fold, _ in jobs:
        if (protocol, fold) in train_rows:
            continue
        prep = prepare_base_data(build_config(protocol, VARIANTS["raw"], fold))
        train_rows[(protocol, fold)] = len(prep.y_train)

    totals = defaultdict(float)
    counts = defaultdict(int)
    for protocol, variant_name, fold, seeds in jobs:
        for model_name in models:
            rate = rates.get((variant_name, model_name))
            if rate is None:
                raise SystemExit(f"Sem base da v2 para {variant_name}/{model_name}.")
            totals[protocol] += rate * train_rows[(protocol, fold)] * len(seeds)
            counts[protocol] += len(seeds)

    for protocol in args.protocols:
        rows = [n for (p, _), n in train_rows.items() if p == protocol]
        print(f"\n  {protocol.upper()}: {counts[protocol]} treinos, "
              f"{min(rows)}–{max(rows)} linhas de treino por fold")
        print(f"    projeção: {format_duration(totals[protocol])} "
              f"(faixa {format_duration(totals[protocol] * 0.8)} a "
              f"{format_duration(totals[protocol] * 1.4)})")

    total = sum(totals.values())
    print(f"\n  TOTAL: {sum(counts.values())} treinos — {format_duration(total)} "
          f"(faixa {format_duration(total * 0.8)} a {format_duration(total * 1.4)})")
    print("\n  Base: tempo por linha de treino dos modelos agrupados da v2. A faixa")
    print("  superior cobre treinos que passem do mínimo de épocas do EarlyStopping")
    print("  e o preparo dos dados de cada fold.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--protocols", nargs="+", choices=PROTOCOLS, default=PROTOCOLS)
    parser.add_argument("--variants", nargs="+", choices=POOLED_VARIANTS, default=POOLED_VARIANTS)
    parser.add_argument("--models", nargs="+", help=f"Subconjunto de {MODEL_NAMES}. Padrão: todos.")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                        help="Sementes do LOSO. Padrão: 42.")
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(LOTO_N_FOLDS)),
                        help="Folds a rodar (LOSO: 0–4, na ordem dos sujeitos; LOTO: 0–11). "
                             "Padrão: todos.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Pula combinações que já têm JSON gravado (retoma runs).")
    parser.add_argument("--smoke", action="store_true",
                        help="2 épocas, pastas descartáveis: valida a grade.")
    parser.add_argument("--estimate", action="store_true",
                        help="Só projeta o tempo de execução a partir dos treinos da v2.")
    args = parser.parse_args()

    models = resolve_models(args.models)
    jobs = list(iter_jobs(args))

    if args.estimate:
        estimate(args, jobs, models)
        return

    root = V3_ROOT / "_smoke" if args.smoke else V3_ROOT
    metrics_root = root / "metrics"
    models_root = root / "models"
    epochs = 2 if args.smoke else None

    n_trainings = sum(len(seeds) for *_, seeds in jobs) * len(models)

    banner("ITERAÇÃO v3 — LOSO E LOTO (AGRUPADO)")
    print(f"Protocolos : {', '.join(args.protocols)}")
    print(f"Variantes  : {', '.join(args.variants)}")
    print(f"Modelos    : {', '.join(models)}")
    print(f"Sementes   : {args.seeds} (LOSO)  |  folds: {args.folds}")
    print(f"Treinos    : {n_trainings}")
    print(f"Saída      : {metrics_root}")
    if args.smoke:
        print("MODO SMOKE: 2 épocas, resultados descartáveis")

    started = time.perf_counter()
    done = 0
    summary: list[tuple[str, float, float]] = []

    for protocol, variant_name, fold, seeds in jobs:
        variant = VARIANTS[variant_name]
        pending = [
            seed
            for seed in seeds
            if not (
                args.skip_existing
                and all(
                    (run_dir(metrics_root, protocol, variant_name, fold, seed)
                     / f"{m}.json").exists()
                    for m in models
                )
            )
        ]

        label = f"{protocol}/{variant_name}/{fold_name(protocol, fold)}"

        if not pending:
            print(f"\n--- {label}: já existe, pulado")
            done += len(seeds) * len(models)
            continue

        banner(label)
        config = build_config(protocol, variant, fold)
        prep = prepare_base_data(config)
        report = prep.preprocessing_report
        print(f"  features   : {variant.feature_set} ({report['n_features']} colunas), "
              f"normalização {variant.normalization}")
        print(f"  treino/val/teste: {len(prep.y_train)}/{len(prep.y_val)}/{len(prep.y_test)}"
              f"  purge: {report['purged_rows']}")
        print(f"  classes/teste: {report['class_distribution_test']}")
        assert_no_leakage(prep, config)
        if protocol == "loto":
            assert_loto_fold(prep)

        for seed in seeds:
            metrics_dir = run_dir(metrics_root, protocol, variant_name, fold, seed)
            models_dir = run_dir(models_root, protocol, variant_name, fold, seed)

            if seed not in pending:
                done += len(models)
                continue

            save_report(prep, config, metrics_dir)

            for model_name in models:
                done += 1
                if args.skip_existing and (metrics_dir / f"{model_name}.json").exists():
                    continue

                print(f"\n--- [{done}/{n_trainings}] {label} / seed_{seed} / {model_name}", flush=True)
                accuracy, elapsed = train_and_export(
                    model_name,
                    prep,
                    metrics_dir=metrics_dir,
                    models_dir=models_dir,
                    epochs=epochs,
                    random_state=seed,
                    verbose=2,
                    metadata={
                        "iteration": "v3",
                        "protocol": protocol,
                        "variant": variant_name,
                        "feature_set": variant.feature_set,
                        "normalization": variant.normalization,
                        "track": "pooled",
                        "subject": None,
                        "fold": fold,
                        "test_subject": report.get("loso_test_subject"),
                        "val_subject": report.get("loso_val_subject"),
                        "seed": seed,
                    },
                    save_model=(protocol == "loso" and seed == 42),
                    save_predictions=True,
                )
                summary.append((f"{label}/seed_{seed}/{model_name}", accuracy, elapsed))
                total = time.perf_counter() - started
                print(f"[resumo] acurácia {accuracy:.4f} em {format_duration(elapsed)} "
                      f"(decorrido {format_duration(total)})", flush=True)

    banner("RESUMO — v3")
    for name, accuracy, elapsed in summary:
        print(f"  {name:<70} {accuracy:.4f}  {format_duration(elapsed)}")
    print(f"\n  tempo total: {format_duration(time.perf_counter() - started)}")


if __name__ == "__main__":
    main()
