"""
Simulação Pygame 2D para demonstração de BCI EEG.

Uso:
    python -m src.simulation.pygame_simulator
    python -m src.simulation.pygame_simulator --model lstm --no-mock
    python -m src.simulation.pygame_simulator --model transformer --no-mock

Teclas:
    ESC / fechar janela  → sair
    SPACE                → pausar / retomar
    F                    → congelar dados EEG (carro continua) / retomar
    R                    → resetar veículo ao centro
"""
from __future__ import annotations

import argparse
import sys
from collections import deque

import pygame

from src.simulation.config import SimulationConfig
from src.simulation.controllers.mock_controller import MockController, PredictionResult
from src.simulation.entities.vehicle import Vehicle
from src.simulation.rendering.environment import Environment
from src.simulation.rendering.hud import HUD, HUDData
from src.simulation.rendering.eeg_panel import EEGPanel
from src.constants import CLASS_NAMES


def run_simulation(
    model_name: str = "cnn",
    use_mock: bool = True,
    config: SimulationConfig | None = None,
    data_path: str | None = None,
) -> None:
    """
    Ponto de entrada público da simulação.

    Args:
        model_name: "cnn" | "lstm" | "gru" | "transformer"
        use_mock:   True → MockController (sem TF, abre imediatamente)
                    False → ModelController (carrega .keras + X_test_base.npy)
        config:     Override opcional de SimulationConfig.
        data_path:  Override do diretório data/processed/ (usado com use_mock=False).
    """
    if config is None:
        config = SimulationConfig()

    if data_path is not None:
        from pathlib import Path
        config.data_dir = Path(data_path)

    # --- Seleciona controlador ---
    if use_mock:
        controller = MockController(hold_ms=config.mock_class_hold_ms)
    else:
        from src.simulation.controllers.model_controller import ModelController
        controller = ModelController(config=config, model_name=model_name)

    # --- Inicialização do Pygame ---
    pygame.init()
    pygame.display.set_caption(config.title)
    screen = pygame.display.set_mode((config.window_width, config.window_height))
    clock = pygame.time.Clock()

    # --- Entidades e renderizadores ---
    env = Environment(config)
    vehicle = Vehicle(config)
    hud = HUD(config)
    hud.init_fonts()
    eeg_panel = EEGPanel(config)
    eeg_panel.init_fonts()

    # --- Estado da simulação ---
    last_pred_ms: int = 0
    current_pred = PredictionResult(
        class_id=2,
        probabilities=(0.05, 0.05, 0.90),
        source="mock" if use_mock else "model",
        model_name="Mock" if use_mock else config.get_display_name(model_name),
    )
    running = True
    paused = False
    data_frozen = False
    attn_initialized = False

    # Histórico visual de predições para a timeline: (class_id, confidence)
    pred_history: deque[tuple[int, float]] = deque(maxlen=hud.TIMELINE_MAX)

    # ── Loop principal ──────────────────────────────────────────────────────
    while running:
        clock.tick(config.fps_target)
        ticks = pygame.time.get_ticks()

        # Eventos
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_f:
                    data_frozen = not data_frozen
                elif event.key == pygame.K_r:
                    vehicle.reset()
                    env.scroll_y = 0.0
                    pred_history.clear()
                    if hasattr(controller, "reset"):
                        controller.reset()
                    last_pred_ms = ticks

        if paused:
            _draw_pause_overlay(screen, config)
            pygame.display.flip()
            continue

        # Gate de predição: dispara a cada prediction_interval_ms (inativo quando frozen)
        if not data_frozen and ticks - last_pred_ms >= config.prediction_interval_ms:
            current_pred = controller.update(ticks)
            last_pred_ms = ticks

            # Confiança = probabilidade da classe predita
            confidence = float(current_pred.probabilities[current_pred.class_id])
            pred_history.append((current_pred.class_id, confidence))

            # Inicializa extrator de atenção do Transformer na primeira predição
            if (
                not use_mock
                and not attn_initialized
                and model_name == "transformer"
                and hasattr(controller, "model")
                and controller.model is not None
            ):
                attn_initialized = eeg_panel.try_init_attention(controller.model)

        confidence = float(current_pred.probabilities[current_pred.class_id])

        # Atualização de física
        vehicle.apply_command(current_pred.class_id, confidence)
        env.update(moving=True, confidence=confidence)

        # Obtém features brutas (None em modo mock)
        raw_x = None
        if not use_mock and hasattr(controller, "current_raw_x"):
            raw_x = controller.current_raw_x

        # Renderização
        env.draw(screen)
        vehicle.draw(screen)
        eeg_panel.draw(screen, raw_x, model_name if not use_mock else "")
        hud.draw(
            screen,
            HUDData(
                predicted_class_id=current_pred.class_id,
                class_name=CLASS_NAMES[current_pred.class_id],
                probabilities=current_pred.probabilities,
                model_display_name=current_pred.model_name,
                fps=clock.get_fps(),
                use_mock=use_mock,
                history=pred_history,
                data_frozen=data_frozen,
            ),
        )
        _draw_key_legend(screen, config)
        pygame.display.flip()

    pygame.quit()
    sys.exit(0)


def _draw_key_legend(screen: pygame.Surface, config: SimulationConfig) -> None:
    try:
        font = pygame.font.SysFont("Segoe UI", 12)
    except Exception:
        font = pygame.font.Font(None, 16)
    legend = "[SPACE] Pausar   [F] Congelar Dados   [R] Resetar   [ESC] Sair"
    surf = font.render(legend, True, config.color_text_secondary)
    x = config.window_width // 2 - surf.get_width() // 2
    y = config.window_height - surf.get_height() - 6
    screen.blit(surf, (x, y))


def _draw_pause_overlay(screen: pygame.Surface, config: SimulationConfig) -> None:
    overlay = pygame.Surface(
        (config.window_width, config.window_height), pygame.SRCALPHA
    )
    overlay.fill((0, 0, 0, 130))
    screen.blit(overlay, (0, 0))
    try:
        font = pygame.font.SysFont("Segoe UI", 40, bold=True)
    except Exception:
        font = pygame.font.Font(None, 50)
    txt = font.render("PAUSADO  —  SPACE para continuar", True, (220, 225, 240))
    screen.blit(
        txt,
        (config.window_width // 2 - txt.get_width() // 2,
         config.window_height // 2 - txt.get_height() // 2),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BCI EEG Pygame Simulation")
    parser.add_argument(
        "--model", default="cnn",
        choices=["cnn", "lstm", "gru", "transformer"],
        help="Modelo treinado a usar (apenas com --no-mock)",
    )
    parser.add_argument(
        "--no-mock", action="store_true",
        help="Usa predições reais do modelo Keras (requer outputs/models/ e data/processed/)",
    )
    parser.add_argument(
        "--data-path", default=None,
        help="Caminho alternativo para o diretório data/processed/",
    )
    args = parser.parse_args()

    run_simulation(
        model_name=args.model,
        use_mock=not args.no_mock,
        data_path=args.data_path,
    )
