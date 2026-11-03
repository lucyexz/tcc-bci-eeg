from __future__ import annotations

import numpy as np

from src.simulacao.config import SimulationConfig
from src.simulacao.controllers.mock_controller import PredictionResult

TRACK = {"generalista": "pooled", "especialista": "personalized"}
COM_ATENCAO = ("transformer", "cnn_transformer")


class ModelController:
    """Predições reais de um modelo treinado sobre o EEG de teste de um participante.

    Usa o teste do hold-out por blocos (o fim da sessão, que o modelo nunca viu
    no treino), na ordem em que foi gravado, e o modelo da semente 42 desse
    hold-out (resultados/<modelo>/modelos/holdout/...). Cada atualização avança
    uma janela de T registros.

    Todas as predições são calculadas de uma vez no carregamento, com o mesmo
    pré-processamento e o mesmo modelo do treino: as probabilidades são as
    mesmas gravadas em resultados/<modelo>/treinos/holdout/.../predicoes.npz.

    O carregamento acontece no primeiro update(), para a janela do Pygame abrir
    antes do TensorFlow.
    """

    def __init__(self, config: SimulationConfig, modelo: str = "cnn", paradigma: str = "especialista",
                 participante: str = "user_a"):
        self.config = config
        self.modelo = modelo
        self.paradigma = paradigma
        self.participante = participante
        self._loaded = False
        self._index = 0
        self._acertos = 0
        self._total = 0

    def carregar(self) -> None:
        # Imports deferidos: o TensorFlow só é inicializado aqui
        from src.avaliacao.explicabilidade import cls_attention_por_amostra, load_model
        from src.config import HOLDOUT_FOLD, PURGE, RESULTADOS
        from src.dados.preparacao import janelas, preparar
        from src.modelos import MODELOS
        from src.modelos.treino import config_modelo, configurar_tensorflow, pasta_treino

        configurar_tensorflow()
        track = TRACK[self.paradigma]
        subject = self.participante if track == "personalized" else None
        caminho = (pasta_treino(RESULTADOS / self.modelo, "modelos", "holdout", track, subject, "semente_42")
                   / "modelo.keras")
        if not caminho.exists():
            raise FileNotFoundError(f"Modelo não encontrado: {caminho}\nTreine antes: python main.py {self.modelo}")

        print(f"[simulador] Preparando o teste do hold-out ({self.paradigma}, {self.participante})...", flush=True)
        cfg = config_modelo(self.modelo, track, 42)
        _, prep, _ = preparar("holdout", track, subject, HOLDOUT_FOLD, verbose=False)
        w = janelas(prep, cfg.window, PURGE, verbose=False)
        reshape = MODELOS[self.modelo][2]
        modelo = load_model(caminho)

        # O teste inteiro de uma vez, como no treino (mesmas probabilidades); depois só o participante
        proba = modelo.predict(reshape(w.X_test), verbose=0)
        sel = w.users_test == self.participante
        self._proba = proba[sel]
        self._y = w.y_test[sel]
        self._blocos = w.blocks_test[sel]
        self._registros = w.X_test[sel][:, -1, :]
        self._atencao = (cls_attention_por_amostra(modelo, reshape(w.X_test[sel]))
                         if self.modelo in COM_ATENCAO else None)
        self._loaded = True
        acc = float((self._proba.argmax(axis=1) == self._y).mean())
        print(f"[simulador] {self.config.get_display_name(self.modelo)} pronto: {len(self._y)} janelas de teste "
              f"(T = {cfg.window}); acerto no trecho todo: {acc:.1%}", flush=True)

    def update(self, ticks_ms: int) -> PredictionResult:
        if not self._loaded:
            self.carregar()
        i = self._index
        proba = self._proba[i]
        class_id = int(np.argmax(proba))
        verdade = int(self._y[i])
        self._acertos += int(class_id == verdade)
        self._total += 1
        resultado = PredictionResult(
            class_id=class_id,
            probabilities=(float(proba[0]), float(proba[1]), float(proba[2])),
            source="model",
            model_name=f"{self.config.get_display_name(self.modelo)} · {self.paradigma}",
            true_class_id=verdade,
            acertos=self._acertos,
            total=self._total,
            progresso=f"{self.participante} · bloco {int(self._blocos[i]) + 1} · janela {i + 1}/{len(self._y)}",
            registro=self._registros[i],
            atencao=None if self._atencao is None else self._atencao[i],
        )
        self._index = (i + 1) % len(self._y)
        if self._index == 0:  # recomeça o trecho: zera o acerto acumulado
            self._acertos = self._total = 0
        return resultado

    def reset(self) -> None:
        self._index = 0
        self._acertos = self._total = 0
