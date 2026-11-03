"""Leitura dos treinos gravados, agregação e gravação das tabelas.

Agregação usada em todas as tabelas: primeiro a média das sementes (hold-out)
ou dos folds (CV) de cada participante; depois média e DP entre os 5
participantes. Métrica: acurácia balanceada no teste comum (acaso = 33,3%).
A tabela por classe (metricas_por_classe.csv) segue a mesma agregação, com
precisão, sensibilidade (recall) e F1 de cada classe e a média macro das três.

As tabelas são impressas no terminal e gravadas em .csv (separador ";" e
vírgula decimal, para abrir direto no Excel em português). Percentuais com uma
casa decimal; DP em pontos percentuais.
"""

import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support

from src.config import folhas, subjects
from src.modelos.treino import pasta_treino

# Os quatro cenários das tabelas principais: (protocolo, paradigma, rótulo da coluna)
CENARIOS = [("holdout", "pooled", "Hold-out generalista"), ("holdout", "personalized", "Hold-out especialista"),
            ("cv", "pooled", "CV generalista"), ("cv", "personalized", "CV especialista")]


def pct(v, casas: int = 1) -> float:
    """0,4321 → 43,2 (nan continua nan)."""
    return float("nan") if v is None or v != v else round(v * 100, casas)


def colunas_media_dp(final: dict, chave=lambda p, t: (p, t)) -> dict:
    """Média (%) e DP (pp) entre participantes nos quatro cenários, como colunas de uma linha."""
    linha = {}
    for p, t, rotulo in CENARIOS:
        m, sd = media_dp(final[chave(p, t)])
        linha[f"{rotulo} (%)"] = pct(m)
        linha[f"{rotulo} DP (pp)"] = pct(sd)
    return linha


def salvar_tabela(linhas: list[dict], caminho: Path, titulo: str) -> pd.DataFrame | None:
    """Imprime a tabela no terminal e grava em .csv (nada, se não houver linhas)."""
    if not linhas:
        print(f"\n{titulo}: sem dados (treinos incompletos)")
        return None
    df = pd.DataFrame(linhas)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(caminho, sep=";", decimal=",", index=False, encoding="utf-8-sig")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(f"\n{titulo}\n{df.to_string(index=False)}\n  → {caminho}")
    return df


class Treino:
    """Um treino gravado: metricas.json + predicoes.npz."""

    def __init__(self, pasta: Path):
        self.pasta = pasta
        self.meta = json.loads((pasta / "metricas.json").read_text(encoding="utf-8"))
        data = np.load(pasta / "predicoes.npz", allow_pickle=True)
        self.y = data["y_true"]
        self.proba = data["y_pred_proba"].astype(float)
        self.users = data["users"].astype(str)

    def scores(self) -> dict[str, float]:
        """Acurácia balanceada no teste, por participante."""
        y_pred = self.proba.argmax(axis=1)
        return {u: float(balanced_accuracy_score(self.y[self.users == u], y_pred[self.users == u]))
                for u in sorted(np.unique(self.users))}

    def por_classe(self) -> dict[str, np.ndarray]:
        """Precisão, sensibilidade e F1 de cada classe no teste, por participante: {u: array (3 métricas × 3 classes)}."""
        y_pred = self.proba.argmax(axis=1)
        out = {}
        for u in sorted(np.unique(self.users)):
            m = self.users == u
            p, r, f, _ = precision_recall_fscore_support(self.y[m], y_pred[m], labels=[0, 1, 2], zero_division=0)
            out[u] = np.vstack([p, r, f])
        return out

    def val_scores(self) -> dict[str, float]:
        return self.meta["val_balanced_accuracy_per_user"]

    def train_scores(self) -> dict[str, float]:
        return self.meta["train_balanced_accuracy_per_user"]


def carregar(raiz: Path, protocolo: str, track: str) -> dict:
    """{(sujeito, folha): Treino} dos treinos que existem."""
    runs = {}
    for subject in subjects(track):
        for folha in folhas(protocolo):
            pasta = pasta_treino(raiz, "treinos", protocolo, track, subject, folha)
            if (pasta / "predicoes.npz").exists():
                runs[(subject, folha)] = Treino(pasta)
    return runs


def esperado(protocolo: str, track: str) -> int:
    return len(folhas(protocolo)) * len(subjects(track))


def por_participante(runs: dict, score_fn=Treino.scores) -> dict[str, float]:
    """Média das sementes/folds de cada participante."""
    acc = defaultdict(list)
    for (subject, _), run in runs.items():
        for user, value in score_fn(run).items():
            if subject is None or user == subject:
                acc[user].append(value)
    return {u: statistics.mean(v) for u, v in sorted(acc.items())}


CLASSES = ["Esquerda", "Direita", "Neutro"]  # mesma ordem dos rótulos 0, 1, 2
METRICAS_CLASSE = ("Precisão", "Sensibilidade", "F1")


def linhas_por_classe(runs_por_cenario: dict, prefixo: dict | None = None) -> list[dict]:
    """Linhas da tabela por classe: {(protocolo, paradigma): runs} → cenário × (3 classes + macro).

    Mesma agregação das outras tabelas: média das sementes/folds de cada participante, depois média e DP
    entre participantes. "Macro" é a média simples das três classes (a sensibilidade macro é a acurácia
    balanceada; o F1 macro é o f1_score(average="macro")). Precisão de uma classe nunca prevista conta 0.
    """
    linhas = []
    for p, t, rotulo in CENARIOS:
        runs = runs_por_cenario.get((p, t))
        if not runs:
            continue
        acc = defaultdict(list)
        for (subject, _), run in runs.items():
            for user, valores in run.por_classe().items():
                if subject is None or user == subject:
                    acc[user].append(valores)
        por_user = np.stack([np.mean(v, axis=0) for _, v in sorted(acc.items())])     # (participantes, 3, 3)
        por_user = np.concatenate([por_user, por_user.mean(axis=2, keepdims=True)], axis=2)  # + macro
        media = por_user.mean(axis=0)
        dp = por_user.std(axis=0, ddof=1) if len(por_user) > 1 else np.zeros_like(media)
        for j, classe in enumerate([*CLASSES, "Macro"]):
            linha = {**(prefixo or {}), "Cenário": rotulo, "Classe": classe}
            for i, nome in enumerate(METRICAS_CLASSE):
                linha[f"{nome} (%)"] = pct(media[i, j])
                linha[f"{nome} DP (pp)"] = pct(dp[i, j])
            linhas.append(linha)
    return linhas


def media_dp(per_user: dict[str, float]) -> tuple[float, float]:
    values = list(per_user.values())
    if not values:
        return float("nan"), float("nan")
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def num(v, casas: int = 1) -> str:
    return f"{v:.{casas}f}".replace(".", ",").replace("-", "−")


def calibracao(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> dict:
    """Diagrama de confiabilidade e ECE (bins de confiança entre 1/3 e 1, mínimo 20 amostras)."""
    confidence = proba.max(axis=1)
    correct = proba.argmax(axis=1) == y_true
    edges = np.linspace(1 / 3, 1, n_bins + 1)
    centers, accuracy, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence >= lo) & (confidence < hi if hi < 1 else confidence <= hi)
        if mask.sum() >= 20:
            centers.append(float(confidence[mask].mean()))
            accuracy.append(float(correct[mask].mean()))
            counts.append(int(mask.sum()))
    ece = sum(c * abs(a - m) for m, a, c in zip(centers, accuracy, counts)) / max(1, sum(counts))
    return {"confidence": centers, "accuracy": accuracy, "counts": counts, "ece": float(ece),
            "mean_confidence": float(confidence.mean()), "accuracy_overall": float(correct.mean())}


def calibracao_de(runs: dict) -> dict:
    y = np.concatenate([r.y for r in runs.values()])
    proba = np.concatenate([r.proba for r in runs.values()])
    return calibracao(y, proba)


def gap_treino_teste(runs: dict) -> float:
    """Média de (acurácia de treino − acurácia de teste) por participante e treino."""
    gaps = []
    for (subject, _), r in runs.items():
        tr, te = r.train_scores(), r.scores()
        gaps += [tr[u] - te[u] for u in te if u in tr and (subject is None or u == subject)]
    return float(np.mean(gaps)) if gaps else float("nan")


def matriz_confusao(runs: dict) -> np.ndarray:
    """Matriz de confusão somada sobre os treinos, normalizada por linha (classe verdadeira)."""
    cm = np.zeros((3, 3))
    for r in runs.values():
        cm += np.array(r.meta["confusion_matrix"])
    return cm / cm.sum(axis=1, keepdims=True)
