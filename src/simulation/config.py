from dataclasses import dataclass, field
from pathlib import Path

_DISPLAY_NAMES = {
    "cnn": "CNN",
    "lstm": "Bi-LSTM",
    "gru": "Bi-GRU",
    "transformer": "Transformer",
}


@dataclass
class SimulationConfig:
    # --- Janela ---
    window_width: int = 1280
    window_height: int = 720
    fps_target: int = 60
    title: str = "BCI EEG Simulation — Motor Imagery"

    # --- Painel EEG (esquerda) ---
    eeg_panel_x: int = 10
    eeg_panel_width: int = 260

    # --- Cadência de predição ---
    # Desacopla a taxa do hardware EEG (128 Hz) da taxa visual (1 Hz).
    # Para EEG real: reduza para 8 ms (125 Hz). Para demo: mantenha 1000 ms.
    prediction_interval_ms: int = 1000
    mock_class_hold_ms: int = 1500

    # --- Física do veículo ---
    vehicle_lateral_speed: float = 2.5   # px/frame ao desviar
    vehicle_turn_rate: float = 1.8       # graus/frame ao virar
    vehicle_max_angle: float = 30.0      # ângulo máximo de esterçamento

    # --- Geometria do corredor ---
    track_lane_width: int = 320
    track_dash_length: int = 40
    track_dash_gap: int = 30
    track_scroll_speed: float = 2.0      # px/frame de scroll do fundo

    # --- Paleta de cores (R, G, B) ---
    color_background: tuple = (15, 20, 35)
    color_track: tuple = (30, 38, 58)
    color_track_line: tuple = (60, 70, 95)
    color_hud_bg: tuple = (10, 15, 28)
    color_hud_border: tuple = (70, 90, 130)
    color_class_left: tuple = (74, 144, 217)     # azul — Esquerda
    color_class_right: tuple = (232, 131, 74)    # laranja — Direita
    color_class_neutral: tuple = (92, 184, 92)   # verde — Neutro
    color_text_primary: tuple = (220, 225, 240)
    color_text_secondary: tuple = (140, 155, 180)
    color_fps: tuple = (120, 200, 120)

    # --- Paths ---
    models_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2] / "outputs" / "models"
    )
    data_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2] / "data" / "processed"
    )

    def get_display_name(self, model_key: str) -> str:
        return _DISPLAY_NAMES.get(model_key, model_key.upper())
