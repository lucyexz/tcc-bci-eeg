"""
Simulação Pygame 2D: um modelo treinado dirige o carro com o EEG de teste.

O carro segue a classe prevista (Esquerda, Direita, Neutro) a cada janela do
teste do hold-out de um participante, na ordem em que foi gravado. O painel
mostra também a classe verdadeira e o acerto acumulado.

Uso:
    python main.py simulador                                   # CNN, especialista, user_a
    python main.py simulador --modelo transformer --participante user_c
    python main.py simulador --paradigma generalista --velocidade 16
    python main.py simulador --mock                            # sem modelo (predições sorteadas)

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

from src.simulacao.config import SimulationConfig
from src.simulacao.controllers.mock_controller import MockController, PredictionResult
from src.simulacao.entities.vehicle import Vehicle
from src.simulacao.rendering.environment import Environment
from src.simulacao.rendering.hud import HUD, HUDData
from src.simulacao.rendering.eeg_panel import EEGPanel
from src.constantes import CLASS_NAMES


def run_simulation(
    model_name: str = "cnn",
    use_mock: bool = False,
    paradigma: str = "especialista",
    participante: str = "user_a",
    velocidade: float = 8.0,
    config: SimulationConfig | None = None,
) -> None:
    """
    Ponto de entrada da simulação.

    Args:
        model_name:   cnn | bilstm | bigru | transformer | cnn_transformer
        use_mock:     True → predições sorteadas (sem TensorFlow, abre na hora)
        paradigma:    generalista | especialista (qual modelo treinado usar)
        participante: de quem é o EEG de teste
        velocidade:   janelas por segundo (8 = o ritmo em que o dataset foi gravado)
    """
    if config is None:
        config = SimulationConfig()

    if use_mock:
        controller = MockController(hold_ms=config.mock_class_hold_ms)
    else:
        from src.simulacao.controllers.model_controller import ModelController
        controller = ModelController(config=config, modelo=model_name, paradigma=paradigma,
                                     participante=participante)
        config.prediction_interval_ms = max(1, int(round(1000 / velocidade)))
        controller.carregar()  # antes de abrir a janela: TensorFlow e predições (alguns segundos)

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

    # Histórico visual da timeline: (classe prevista, confiança, classe verdadeira ou None)
    pred_history: deque[tuple[int, float, int | None]] = deque(maxlen=hud.TIMELINE_MAX)

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
            pred_history.append((current_pred.class_id, confidence, current_pred.true_class_id))

        confidence = float(current_pred.probabilities[current_pred.class_id])

        # Atualização de física
        vehicle.apply_command(current_pred.class_id, confidence)
        env.update(moving=True, confidence=confidence)

        # Renderização
        env.draw(screen)
        vehicle.draw(screen)
        eeg_panel.draw(screen, current_pred.registro, current_pred.atencao)
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
                true_class_id=current_pred.true_class_id,
                acertos=current_pred.acertos,
                total=current_pred.total,
                progresso=current_pred.progresso,
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python main.py simulador", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modelo", default="cnn",
                        choices=["cnn", "bilstm", "bigru", "transformer", "cnn_transformer"])
    parser.add_argument("--paradigma", default="especialista", choices=["especialista", "generalista"])
    parser.add_argument("--participante", default="user_a",
                        choices=["user_a", "user_b", "user_c", "user_d", "user_e"])
    parser.add_argument("--velocidade", type=float, default=8.0,
                        help="janelas por segundo (padrão 8 = tempo real da gravação)")
    parser.add_argument("--mock", action="store_true", help="sem modelo: predições sorteadas")
    args = parser.parse_args(argv)
    run_simulation(model_name=args.modelo, use_mock=args.mock, paradigma=args.paradigma,
                   participante=args.participante, velocidade=args.velocidade)


if __name__ == "__main__":
    main()
