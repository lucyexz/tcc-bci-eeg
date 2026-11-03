"""Treino de um modelo em todas as unidades de um protocolo, gravando métricas, predições e modelo.

Estrutura de saída de cada treino:
    resultados/<modelo>/treinos/<protocolo>/<generalista|especialista>/[<sujeito>/]<semente_S|fold_K>/
        metricas.json  predicoes.npz  preprocessamento.json
    resultados/<modelo>/modelos/holdout/.../modelo.keras   (só hold-out)

Treinos já gravados são pulados, então uma rodada interrompida pode ser retomada.
"""

import gc
import json
import time
import warnings
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (balanced_accuracy_score, classification_report, cohen_kappa_score,
                             confusion_matrix, f1_score)
from tensorflow import keras

from src.config import (CONFIG_FINAL, EVAL_HISTORY, PURGE, RESULTADOS, THREADS, TRACK_DIR, TRACK_NOME,
                        sementes_de, unidades)
from src.dados.preparacao import janelas, preparar
from src.modelos import MODELOS, NOMES
from src.modelos.comum import treinar

CLASSES = ["Esquerda", "Direita", "Neutro"]

# Segundos por treino medidos nesta máquina (CPU, 4 threads), só para a estimativa
# antes de qualquer resultado existir. Depois valem os tempos gravados.
SEGUNDOS_POR_TREINO = {
    "cnn": {"pooled": 15, "personalized": 4.5},
    "bilstm": {"pooled": 250, "personalized": 57},
    "bigru": {"pooled": 230, "personalized": 50},
    "transformer": {"pooled": 60, "personalized": 9},
    "cnn_transformer": {"pooled": 43, "personalized": 7.6},
}


def configurar_tensorflow(threads: int = THREADS) -> None:
    """Threads fixas e operações determinísticas: o mesmo treino dá o mesmo resultado."""
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(min(2, threads))
    tf.config.experimental.enable_op_determinism()


def raiz_modelo(modelo: str, smoke: bool = False) -> Path:
    return (RESULTADOS / "_smoke" if smoke else RESULTADOS) / modelo


def pasta_treino(raiz: Path, parte: str, protocolo: str, track: str, subject, folha: str) -> Path:
    path = raiz / parte / protocolo / TRACK_DIR[track]
    return (path / subject if subject else path) / folha


def config_modelo(modelo: str, track: str, semente: int, epochs: int | None = None):
    config_cls = MODELOS[modelo][0]
    extra = {"epochs": epochs} if epochs else {}
    return config_cls(**CONFIG_FINAL[modelo][track], random_state=semente, verbose=0, **extra)


def contar_parametros(modelo: str, track: str) -> int:
    _, build, _ = MODELOS[modelo]
    model = build(config_modelo(modelo, track, 42))
    n = int(model.count_params())
    keras.backend.clear_session()
    return n


# ---------------------------------------------------------------------------
# Estimativa
# ---------------------------------------------------------------------------

def estimar(modelo: str, protocolos: list[str], raiz: Path) -> float:
    """Imprime e devolve a estimativa de tempo (s) dos treinos que ainda faltam."""
    total = 0.0
    for protocolo in protocolos:
        for track in TRACK_DIR:
            feitos, faltam, tempos = 0, 0, []
            for _, subject, fold, sementes in unidades(protocolo, tracks=[track]):
                for folha in _folhas(protocolo, fold, sementes):
                    path = pasta_treino(raiz, "treinos", protocolo, track, subject, folha) / "metricas.json"
                    if path.exists():
                        feitos += 1
                        tempos.append(json.loads(path.read_text(encoding="utf-8"))["train_seconds"])
                    else:
                        faltam += 1
            por_treino = np.mean(tempos) if tempos else SEGUNDOS_POR_TREINO[modelo][track]
            total += por_treino * faltam
            print(f"  {protocolo:<18} {TRACK_NOME[track]:<13} {feitos:>3} feitos, {faltam:>3} a fazer "
                  f"× {por_treino:5.0f} s = {formatar_duracao(por_treino * faltam)}")
    print(f"  total previsto: {formatar_duracao(total)}", flush=True)
    return total


def _folhas(protocolo: str, fold: int, sementes: list[int]) -> list[str]:
    return [f"fold_{fold}"] if protocolo == "cv" else [f"semente_{s}" for s in sementes]


def formatar_duracao(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f"{h}h{m:02d}min" if h else (f"{m}min{s:02d}s" if m else f"{s}s")


# ---------------------------------------------------------------------------
# Treino
# ---------------------------------------------------------------------------

def treinar_protocolo(modelo: str, protocolo: str, raiz: Path, epochs: int | None = None,
                      tracks=None, users=None, sementes=None) -> None:
    """Treina `modelo` em todas as unidades de `protocolo` que ainda não foram gravadas."""
    _, build, reshape = MODELOS[modelo]
    kwargs = {k: v for k, v in (("tracks", tracks), ("users", users), ("sementes", sementes)) if v}
    lista = list(unidades(protocolo, **kwargs))
    n_total = sum(len(_folhas(protocolo, fold, s)) for *_, fold, s in lista)
    feito = 0
    inicio = time.perf_counter()
    permutar = protocolo == "holdout_permutado"

    for track, subject, fold, sementes_unidade in lista:
        window = CONFIG_FINAL[modelo][track]["window"]
        rotulo = f"{NOMES[modelo]} | {protocolo} | {TRACK_NOME[track]}" + (f" | {subject}" if subject else "")
        pendentes = []
        for semente, folha in zip(sementes_unidade if protocolo != "cv" else [42],
                                  _folhas(protocolo, fold, sementes_unidade)):
            out = pasta_treino(raiz, "treinos", protocolo, track, subject, folha)
            if (out / "metricas.json").exists():
                feito += 1
            else:
                pendentes.append((semente, folha, out))
        if not pendentes:
            continue

        print(f"\n--- {rotulo}" + (f" | fold {fold}" if protocolo == "cv" else "") + f" | T={window}", flush=True)
        wprep = None
        for semente, folha, out in pendentes:
            if wprep is None or permutar:
                config_prep, prep, permutacao = preparar(
                    protocolo, track, subject, fold, semente_permutacao=semente if permutar else None)
                wprep = janelas(prep, window, PURGE)
                if permutacao is not None:
                    wprep.preprocessing_report["label_permutation"] = {
                        u: {str(b): v for b, v in m.items()} for u, m in permutacao.items()}

            keras.utils.set_random_seed(semente)
            config = config_modelo(modelo, track, semente, epochs)
            t0 = time.perf_counter()
            result = treinar(build, reshape, wprep.X_train, wprep.y_train, wprep.X_test, wprep.y_test,
                             config, validation_data=(wprep.X_val, wprep.y_val))
            elapsed = time.perf_counter() - t0

            if protocolo == "holdout":
                model_dir = pasta_treino(raiz, "modelos", protocolo, track, subject, folha)
                model_dir.mkdir(parents=True, exist_ok=True)
                result.model.save(model_dir / "modelo.keras")

            salvar_treino(out, modelo, result, wprep, config_prep, reshape, elapsed, {
                "protocolo": protocolo, "track": track, "subject": subject,
                "fold": fold if protocolo == "cv" else None, "semente": semente,
            })
            feito += 1
            bal = balanced_accuracy_score(wprep.y_test, result.y_pred)
            print(f"  [{feito}/{n_total}] {folha}: acurácia balanceada {bal:.4f} | "
                  f"{result.history and len(result.history['loss'])} épocas | {formatar_duracao(elapsed)} "
                  f"(decorrido {formatar_duracao(time.perf_counter() - inicio)})", flush=True)

            del result
            keras.backend.clear_session()
            gc.collect()


def _balanced_per_user(model, X, y, users, reshape) -> tuple[float, dict[str, float]]:
    """Acurácia balanceada (geral e por participante) com os pesos da melhor época."""
    y_pred = model.predict(reshape(X), verbose=0, batch_size=1024).argmax(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # um participante pode não ter as 3 classes na validação
        overall = float(balanced_accuracy_score(y, y_pred))
        per_user = {str(u): float(balanced_accuracy_score(y[users == u], y_pred[users == u]))
                    for u in np.unique(users)}
    return overall, per_user


def salvar_treino(out: Path, modelo: str, result, wprep, config_prep, reshape, elapsed: float,
                  meta: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(wprep.y_test).astype(int)
    y_pred = np.asarray(result.y_pred).astype(int)
    users = wprep.users_test.astype(str)
    train_ba, train_pu = _balanced_per_user(result.model, wprep.X_train, wprep.y_train, wprep.users_train, reshape)
    val_ba, val_pu = _balanced_per_user(result.model, wprep.X_val, wprep.y_val, wprep.users_val, reshape)
    history = {k: [float(v) for v in vals] for k, vals in (result.history or {}).items()}
    config = asdict(result.config)

    payload = {
        "modelo": modelo,
        "nome": NOMES[modelo],
        **meta,
        "window_T": wprep.window,
        "eval_history": EVAL_HISTORY,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "balanced_accuracy_per_user": {u: float(balanced_accuracy_score(y_true[users == u], y_pred[users == u]))
                                       for u in np.unique(users)},
        "test_accuracy": float(result.test_accuracy),
        "test_loss": float(result.test_loss),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
        "classification_report": classification_report(y_true, y_pred, labels=[0, 1, 2], target_names=CLASSES,
                                                       output_dict=True, zero_division=0),
        "train_balanced_accuracy": train_ba,
        "train_balanced_accuracy_per_user": train_pu,
        "val_balanced_accuracy": val_ba,
        "val_balanced_accuracy_per_user": val_pu,
        "best_epoch": int(np.argmin(history["val_loss"])) + 1 if history.get("val_loss") else None,
        "epochs_trained": len(history.get("loss", [])),
        "train_seconds": round(elapsed, 1),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "config": {k: list(v) if isinstance(v, tuple) else v for k, v in config.items()},
        "history": history,
    }
    (out / "metricas.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(out / "predicoes.npz", y_true=y_true, y_pred=y_pred,
                        y_pred_proba=np.asarray(result.y_pred_proba, dtype=np.float32),
                        users=users, blocks=wprep.blocks_test.astype(int),
                        positions=wprep.positions_test.astype(int))
    (out / "preprocessamento.json").write_text(json.dumps({
        "preprocessing_config": asdict(config_prep),
        "preprocessing_report": wprep.preprocessing_report,
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


__all__ = ["configurar_tensorflow", "estimar", "treinar_protocolo", "raiz_modelo", "pasta_treino",
           "contar_parametros", "config_modelo", "formatar_duracao", "RESULTADOS", "sementes_de"]
