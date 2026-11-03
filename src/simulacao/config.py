from dataclasses import dataclass

_DISPLAY_NAMES = {
    "cnn": "CNN",
    "bilstm": "Bi-LSTM",
    "bigru": "Bi-GRU",
    "transformer": "Transformer",
    "cnn_transformer": "CNN + Transformer",
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
    # Intervalo entre duas predições. No modo real, 1000 / velocidade (registros
    # por segundo); 125 ms = 8 registros/s = o ritmo em que o dataset foi gravado.
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
    # Neutras, como nas figuras do trabalho: nada de vermelho, verde, amarelo ou laranja
    # (leem como certo/errado ou alerta). Classes: azul, magenta e ocre.
    color_background: tuple = (15, 20, 35)
    color_track: tuple = (30, 38, 58)
    color_track_line: tuple = (60, 70, 95)
    color_hud_bg: tuple = (10, 15, 28)
    color_hud_border: tuple = (70, 90, 130)
    color_class_left: tuple = (91, 155, 232)     # azul — Esquerda
    color_class_right: tuple = (208, 122, 180)   # magenta — Direita
    color_class_neutral: tuple = (201, 150, 79)  # ocre — Neutro
    color_text_primary: tuple = (220, 225, 240)
    color_text_secondary: tuple = (140, 155, 180)

    def get_display_name(self, model_key: str) -> str:
        return _DISPLAY_NAMES.get(model_key, model_key.upper())
