from __future__ import annotations

import numpy as np

from src.simulation.config import SimulationConfig
from src.simulation.controllers.mock_controller import PredictionResult

_MODEL_FILES = {
    "cnn": "cnn_model.keras",
    "lstm": "lstm_model.keras",
    "gru": "gru_model.keras",
    "transformer": "transformer_model.keras",
}


class ModelController:
    """
    Serve predições reais iterando sobre X_test_base.npy com um modelo Keras treinado.

    Interface idêntica ao MockController: update(ticks_ms) → PredictionResult.

    O modelo e os dados são carregados lazily no primeiro update() para que
    a janela Pygame abra imediatamente antes da inicialização do TensorFlow (~3s).

    Integração futura:
        - Path A (atual): itera sequencialmente sobre X_test_base.npy.
        - Path B (EEG ao vivo): substitua self._X_test[self._index] por um
          buffer thread-safe alimentado pelo SDK do dispositivo EEG.
        - Path C (dashboard): consuma predições via WebSocket/fila do FastAPI.
    """

    def __init__(self, config: SimulationConfig, model_name: str = "cnn"):
        self.config = config
        self.model_name = model_name
        self._model = None
        self._X_test: np.ndarray | None = None
        self._index: int = 0
        self._loaded: bool = False
        self._current_raw_x: np.ndarray | None = None

    @property
    def current_raw_x(self) -> np.ndarray | None:
        """Array (25,) com as features brutas da predição mais recente."""
        return self._current_raw_x

    def update(self, ticks_ms: int) -> PredictionResult:
        if not self._loaded:
            self._load()
        x = self._X_test[self._index]
        self._current_raw_x = x.copy()
        x_batch = self._reshape(x)
        proba = self._model.predict(x_batch, verbose=0)[0]
        class_id = int(np.argmax(proba))
        self._index = (self._index + 1) % len(self._X_test)
        return PredictionResult(
            class_id=class_id,
            probabilities=(float(proba[0]), float(proba[1]), float(proba[2])),
            source="model",
            model_name=self.config.get_display_name(self.model_name),
        )

    def reset(self) -> None:
        self._index = 0
        self._current_raw_x = None

    @property
    def model(self):
        """Modelo Keras carregado (disponível após o primeiro update())."""
        return self._model

    def _load(self) -> None:
        # Import deferido — TF não é inicializado até o primeiro update()
        from tensorflow import keras

        model_path = self.config.models_dir / _MODEL_FILES[self.model_name]
        if not model_path.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado: {model_path}\n"
                "Treine o modelo executando o notebook correspondente primeiro."
            )

        custom_objects = None
        if self.model_name == "transformer":
            from src.models.transformer import (
                _LearnablePositionalEmbedding,
                _PrependCLSToken,
                _ExtractFirstToken,
            )
            custom_objects = {
                "_LearnablePositionalEmbedding": _LearnablePositionalEmbedding,
                "_PrependCLSToken": _PrependCLSToken,
                "_ExtractFirstToken": _ExtractFirstToken,
            }

        self._model = keras.models.load_model(
            model_path, custom_objects=custom_objects
        )
        self._X_test = np.load(self.config.data_dir / "X_test_base.npy")
        self._loaded = True
        print(
            f"[ModelController] {self.model_name} carregado — "
            f"{len(self._X_test)} amostras de teste disponíveis."
        )

    def _reshape(self, x: np.ndarray) -> np.ndarray:
        """Reshape de um único sample (25,) para o formato esperado pelo modelo."""
        if self.model_name == "cnn":
            from src.models.cnn import create_temporal_windows, reshape_for_cnn
            windowed = create_temporal_windows(x.reshape(1, -1), n_frames=1)
            return reshape_for_cnn(windowed, n_frames=1)
        elif self.model_name in ("lstm", "gru"):
            from src.models.lstm import reshape_for_lstm
            return reshape_for_lstm(x.reshape(1, -1))
        else:
            from src.models.transformer import reshape_for_transformer
            return reshape_for_transformer(x.reshape(1, -1))
