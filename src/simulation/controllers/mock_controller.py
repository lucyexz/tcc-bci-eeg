from dataclasses import dataclass

import numpy as np


@dataclass
class PredictionResult:
    """Contrato de dados unificado entre controladores e o simulador."""
    class_id: int
    probabilities: tuple  # (P(Esquerda), P(Direita), P(Neutro))
    source: str = "mock"  # "mock" | "model"
    model_name: str = "—"


class MockController:
    """
    Gera predições falsas ciclando pelas 3 classes em um timer.

    Ciclo padrão: Neutro → Esquerda → Neutro → Direita → ...
    Começa com Neutro para que o veículo já esteja em movimento antes
    do primeiro comando direcional aparecer na demonstração.

    Modos:
        "cycle": rotação determinística — garante que todas as direções
                 apareçam na demo.
        "random": classe aleatória a cada intervalo — útil para testes visuais.
    """

    CYCLE_ORDER = (2, 0, 2, 1, 2, 0, 2, 1)

    def __init__(
        self,
        hold_ms: int = 1500,
        mode: str = "cycle",
        seed: int = 42,
    ):
        self.hold_ms = hold_ms
        self.mode = mode
        self._rng = np.random.default_rng(seed)
        self._cycle_index: int = 0
        self._last_switch_ms: int = 0
        self._current = self._build(self.CYCLE_ORDER[0])

    def update(self, ticks_ms: int) -> PredictionResult:
        if ticks_ms - self._last_switch_ms >= self.hold_ms:
            self._advance()
            self._last_switch_ms = ticks_ms
        return self._current

    def reset(self) -> None:
        self._cycle_index = 0
        self._last_switch_ms = 0
        self._current = self._build(self.CYCLE_ORDER[0])

    def _advance(self) -> None:
        if self.mode == "cycle":
            self._cycle_index = (self._cycle_index + 1) % len(self.CYCLE_ORDER)
            next_class = self.CYCLE_ORDER[self._cycle_index]
        else:
            next_class = int(self._rng.integers(0, 3))
        self._current = self._build(next_class)

    def _build(self, class_id: int) -> PredictionResult:
        confidence = float(self._rng.uniform(0.72, 0.93))
        residual = 1.0 - confidence
        split = float(self._rng.uniform(0.0, residual))
        others = [i for i in range(3) if i != class_id]
        probs = [0.0, 0.0, 0.0]
        probs[class_id] = confidence
        probs[others[0]] = split
        probs[others[1]] = residual - split
        return PredictionResult(
            class_id=class_id,
            probabilities=(probs[0], probs[1], probs[2]),
            source="mock",
            model_name="Mock",
        )
