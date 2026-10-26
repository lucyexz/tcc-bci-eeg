"""Infraestrutura compartilhada pelos scripts de experimento do TCC2.

Os scripts em `experiments/` assumem a orquestração que hoje vive nos
notebooks 03/04/05: preparar os dados, treinar as 4 arquiteturas, salvar os
modelos e exportar as métricas. A diferença é que aqui tudo é reexecutável e
grava em pastas próprias, sem sobrescrever os artefatos do split aleatório.
"""

import gc
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from tensorflow import keras

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.export_metrics import export_metrics  # noqa: E402
from src.models.cnn import CNNConfig, train_cnn  # noqa: E402
from src.models.lstm import GRUConfig, LSTMConfig, train_gru, train_lstm  # noqa: E402
from src.models.transformer import TransformerConfig, train_transformer  # noqa: E402

USERS = ["user_a", "user_b", "user_c", "user_d", "user_e"]

# A ordem importa: é a ordem em que os modelos são treinados e reportados.
MODEL_SPECS = {
    "cnn": (train_cnn, CNNConfig),
    "lstm": (train_lstm, LSTMConfig),
    "gru": (train_gru, GRUConfig),
    "transformer": (train_transformer, TransformerConfig),
}

MODEL_NAMES = list(MODEL_SPECS)

DATA_DIR = PROJECT_ROOT / "data" / "processed"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"


def banner(text: str) -> None:
    print()
    print("=" * 78)
    print(text)
    print("=" * 78, flush=True)


def assert_no_leakage(prep, config) -> None:
    """Confere os guardrails anti-vazamento e imprime um resumo.

    Levanta AssertionError se alguma garantia foi quebrada. As checagens são
    baratas o bastante para rodar antes de todo treino.
    """
    partitions = {"treino": prep.users_train, "teste": prep.users_test}

    if prep.users_val is not None:
        partitions["validacao"] = prep.users_val

    # 1. O filtro por sujeito não pode deixar passar dados de outro sujeito.
    if config.subject is not None:
        for name, users in partitions.items():
            found = sorted(np.unique(users).tolist())
            assert found == [config.subject], (
                f"Vazamento de sujeito na partição '{name}': "
                f"esperado ['{config.subject}'], encontrado {found}"
            )

    # 1b. No LOSO nenhum sujeito pode aparecer em duas partições, e o teste
    #     tem que ser exatamente o sujeito deixado de fora.
    if config.split_strategy == "loso":
        subjects = {
            name: set(np.unique(users).tolist())
            for name, users in partitions.items()
        }
        assert subjects["teste"] == {config.loso_test_subject}, (
            f"LOSO: teste deveria conter só '{config.loso_test_subject}', "
            f"encontrado {sorted(subjects['teste'])}"
        )
        names = list(subjects)
        for i, first in enumerate(names):
            for second in names[i + 1:]:
                shared = subjects[first] & subjects[second]
                assert not shared, (
                    f"LOSO: sujeito(s) {sorted(shared)} em '{first}' E '{second}'"
                )
        print(
            "  [ok] LOSO: sujeitos disjuntos — "
            + " | ".join(f"{name} {sorted(s)}" for name, s in subjects.items())
        )

    # 2. As partições precisam formar uma partição exata das linhas limpas.
    #    Todos os splits são construídos por índice (`iloc` no corte
    #    cronológico, `flatnonzero` no carve de validação), então se as
    #    contagens fecham com o total não existe sobreposição possível.
    #
    #    Comparar o CONTEÚDO das linhas não serviria aqui: depois do clip de
    #    IQR, linhas extremas saturam no vetor de limite e ficam idênticas
    #    entre si mesmo vindo de sujeitos e momentos diferentes.
    total_rows = int(prep.preprocessing_report["shape_after_cleaning"][0])
    # Na CV por blocos, as linhas descartadas pelo purge também entram na conta.
    partition_rows = (
        len(prep.y_train)
        + len(prep.y_test)
        + int(prep.preprocessing_report.get("purged_rows", 0))
    )

    if prep.y_val is not None:
        partition_rows += len(prep.y_val)

    assert partition_rows == total_rows, (
        f"As partições somam {partition_rows} linhas, mas o dataset limpo tem "
        f"{total_rows}. Treino/validação/teste não formam uma partição exata."
    )

    # 2b. Com bloco/posição disponíveis, confere diretamente que nenhuma
    #     amostra aparece em duas partições e, na CV por blocos, que o purge
    #     deixou pelo menos `purge_samples` amostras entre treino/validação e
    #     teste dentro de cada sujeito.
    if prep.positions_test is not None:
        def keys(users, positions):
            return set(zip(users.tolist(), positions.tolist()))

        train_keys = keys(prep.users_train, prep.positions_train)
        test_keys = keys(prep.users_test, prep.positions_test)
        val_keys = (
            keys(prep.users_val, prep.positions_val)
            if prep.positions_val is not None
            else set()
        )
        assert not (train_keys & test_keys), "Amostras em treino E teste"
        assert not (val_keys & test_keys), "Amostras em validação E teste"
        assert not (train_keys & val_keys), "Amostras em treino E validação"

        if config.split_strategy == "blocked_cv":
            for user in np.unique(prep.users_test):
                test_pos = prep.positions_test[prep.users_test == user]
                fit_pos = np.concatenate([
                    prep.positions_train[prep.users_train == user],
                    prep.positions_val[prep.users_val == user]
                    if prep.positions_val is not None
                    else np.array([], dtype=int),
                ])
                gap = np.maximum(
                    0,
                    np.maximum(test_pos.min() - fit_pos, fit_pos - test_pos.max()),
                ).min()
                assert gap > config.purge_samples, (
                    f"Purge violado em {user}: distância mínima {gap} "
                    f"<= {config.purge_samples}"
                )
                test_blocks = set(prep.blocks_test[prep.users_test == user].tolist())
                fit_blocks = set(prep.blocks_train[prep.users_train == user].tolist())
                assert not (test_blocks & fit_blocks), (
                    f"Bloco de teste também no treino em {user}"
                )
            print(f"  [ok] CV por blocos: teste disjunto e purge > {config.purge_samples} amostras")

    # 3. As 3 classes precisam existir em cada partição, senão a matriz de
    #    confusão e o F1 macro ficam sem sentido.
    labels = {"treino": prep.y_train, "teste": prep.y_test}

    if prep.y_val is not None:
        labels["validacao"] = prep.y_val

    for name, y in labels.items():
        classes = sorted(np.unique(y).tolist())
        assert classes == [0, 1, 2], (
            f"Partição '{name}' não contém as 3 classes: {classes}"
        )

    # 4. O scaler tem que ter sido ajustado só no treino. Se tivesse visto o
    #    teste, a média do teste também seria ~0.
    if config.scale_data:
        train_mean = float(np.abs(prep.X_train.mean(axis=0)).max())
        test_mean = float(np.abs(prep.X_test.mean(axis=0)).max())

        assert train_mean < 1e-6, (
            f"O treino deveria estar centrado em 0, mas o desvio máximo "
            f"da média é {train_mean:.2e}"
        )
        # Exceção declarada: no LOSO com normalização por sujeito, o scaler do
        # sujeito de teste é ajustado nas features dele (sem rótulos), então a
        # média do teste é 0 por construção.
        transductive = (
            config.split_strategy == "loso"
            and getattr(config, "normalization", "global") == "per_subject"
        )

        if transductive:
            print(
                "  [aviso] LOSO + normalização por sujeito: scaler do sujeito "
                "de teste ajustado nas próprias features, SEM rótulos "
                "(calibração não supervisionada)"
            )
        else:
            assert test_mean > 1e-6, (
                "A média do teste é exatamente 0, o que indica que o scaler foi "
                "ajustado incluindo o teste (vazamento)"
            )

        if getattr(config, "normalization", "global") == "per_subject":
            for user in np.unique(prep.users_train):
                user_mean = float(
                    np.abs(prep.X_train[prep.users_train == user].mean(axis=0)).max()
                )
                assert user_mean < 1e-6, (
                    f"Normalização por sujeito: treino de {user} não está "
                    f"centrado (desvio {user_mean:.2e})"
                )
            print("  [ok] scaler por sujeito ajustado só no treino de cada sujeito")

        print(
            f"  [ok] scaler ajustado só no treino "
            f"(desvio da média: treino {train_mean:.2e}, teste {test_mean:.2e})"
        )

    sizes = " | ".join(
        f"{name} {len(users)}" for name, users in partitions.items()
    )
    print(f"  [ok] partição exata das {total_rows} linhas: {sizes}")

    # Informativo: quantas linhas saturaram no clip de IQR. Não é vazamento,
    # mas é bom saber quanta informação o clipping está achatando.
    if config.outlier_strategy == "clip_iqr":
        distinct = len(set(map(bytes, np.ascontiguousarray(prep.X_train))))
        saturated = len(prep.X_train) - distinct
        print(
            f"  [info] {saturated} linha(s) de treino ficaram idênticas a "
            f"outra após o clip de IQR ({saturated / len(prep.X_train):.2%})",
            flush=True,
        )


def save_arrays(prep, out_dir: Path) -> None:
    """Salva os arrays em .npy, na mesma convenção de nomes de data/processed."""
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "X_train_base.npy", prep.X_train)
    np.save(out_dir / "X_test_base.npy", prep.X_test)
    np.save(out_dir / "y_train.npy", prep.y_train)
    np.save(out_dir / "y_test.npy", prep.y_test)
    np.save(out_dir / "users_train.npy", prep.users_train)
    np.save(out_dir / "users_test.npy", prep.users_test)

    if prep.X_val is not None:
        np.save(out_dir / "X_val_base.npy", prep.X_val)
        np.save(out_dir / "y_val.npy", prep.y_val)
        np.save(out_dir / "users_val.npy", prep.users_val)

    print(f"  [ok] arrays salvos em {out_dir}", flush=True)


def save_report(prep, config, out_dir: Path) -> None:
    """Grava a config e o relatório de pré-processamento ao lado das métricas.

    É isso que torna o experimento auditável: o JSON registra o split, o
    iqr_multiplier e de onde veio a validação, informação que não cabe no
    `config` do modelo exportado por `export_metrics`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if prep.X_val is None:
        validation_source = "keras_validation_split"
    elif config.split_strategy == "loso":
        validation_source = "sujeito_de_treino_inteiro"
    elif config.split_strategy == "blocked_cv":
        validation_source = "fold_vizinho"
    else:
        validation_source = "carve_cronologico_por_sujeito"

    payload = {
        "preprocessing_config": asdict(config),
        "preprocessing_report": prep.preprocessing_report,
        "validation_source": validation_source,
    }

    path = out_dir / "_preprocessing.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print(f"  [ok] relatório salvo em {path}", flush=True)


def train_and_export(
    model_name: str,
    prep,
    metrics_dir: Path,
    models_dir: Path,
    epochs: int | None = None,
    random_state: int | None = None,
    verbose: int | None = None,
    metadata: dict | None = None,
    save_model: bool = True,
    save_predictions: bool = False,
) -> tuple[float, float]:
    """Treina uma arquitetura, salva o .keras e exporta o JSON de métricas.

    Args:
        epochs: Sobrescreve `config.epochs`. Usado só pelo modo --smoke;
            nos runs de verdade fica None e valem os hiperparâmetros padrão.
        random_state: Sobrescreve a semente do modelo (runs com várias
            sementes). None mantém o padrão (42).
        metadata: Campos extras gravados no JSON (feature set, protocolo...).
        save_model: Grava o .keras. Desligado na CV, onde seriam dezenas de
            modelos que ninguém vai reabrir.
        save_predictions: Grava o .npz com as predições do teste.

    Só a largura da entrada se adapta ao feature set (`n_feature_maps`);
    camadas e hiperparâmetros continuam os padrões de cada `*Config`.

    Returns:
        (acurácia de teste, tempo de treino em segundos)
    """
    train_fn, config_cls = MODEL_SPECS[model_name]

    overrides = {"n_feature_maps": prep.n_feature_maps}
    if epochs is not None:
        overrides["epochs"] = epochs
    if random_state is not None:
        overrides["random_state"] = random_state
    if verbose is not None:
        overrides["verbose"] = verbose

    config = config_cls(**overrides)

    validation_data = None
    if prep.X_val is not None:
        validation_data = (prep.X_val, prep.y_val)

    started = time.perf_counter()
    result = train_fn(
        prep.X_train,
        prep.y_train,
        prep.X_test,
        prep.y_test,
        config,
        validation_data=validation_data,
    )
    elapsed = time.perf_counter() - started

    if save_model:
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / f"{model_name}_model.keras"
        result.model.save(model_path)
        print(f"[modelo] Salvo: {model_path}")

    export_metrics(
        result,
        model_name=model_name,
        users_test=prep.users_test,
        output_dir=metrics_dir,
        blocks_test=prep.blocks_test,
        positions_test=prep.positions_test,
        metadata={**(metadata or {}), "train_seconds": round(elapsed, 1)},
        save_predictions=save_predictions,
    )

    accuracy = float(result.test_accuracy)

    # Em runs longos (centenas de treinos no mesmo processo) o Keras acumula
    # grafos e tensores de modelos antigos. Liberar a sessão a cada treino
    # mantém o uso de memória estável.
    del result
    keras.backend.clear_session()
    gc.collect()

    return accuracy, elapsed


def resolve_models(requested: list[str] | None) -> list[str]:
    if not requested:
        return MODEL_NAMES

    unknown = [name for name in requested if name not in MODEL_SPECS]
    if unknown:
        raise SystemExit(
            f"Modelo(s) desconhecido(s): {unknown}. "
            f"Válidos: {MODEL_NAMES}"
        )

    return [name for name in MODEL_NAMES if name in requested]


def resolve_users(requested: list[str] | None) -> list[str]:
    if not requested or requested == ["all"]:
        return USERS

    unknown = [name for name in requested if name not in USERS]
    if unknown:
        raise SystemExit(
            f"Sujeito(s) desconhecido(s): {unknown}. Válidos: {USERS}"
        )

    return [name for name in USERS if name in requested]


def format_duration(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m{secs:02d}s"
