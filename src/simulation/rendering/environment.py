from __future__ import annotations

import pygame

from src.simulation.config import SimulationConfig


class Environment:
    """
    Desenha o corredor 2D com scroll vertical que simula movimento para frente.

    O fundo scrolla para baixo a cada frame. O veículo fica parado
    verticalmente enquanto o ambiente se move — convenção padrão de jogos
    de corrida top-down.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.scroll_y: float = 0.0
        self._period = config.track_dash_length + config.track_dash_gap

    def update(self, moving: bool = True, confidence: float = 0.8) -> None:
        if moving:
            # Velocidade de scroll sobe com confiança: decisão clara = resposta mais ágil
            speed = self.config.track_scroll_speed * (0.5 + confidence * 0.8)
            self.scroll_y = (self.scroll_y + speed) % self._period

    def draw(self, surface: pygame.Surface) -> None:
        cfg = self.config
        w, h = cfg.window_width, cfg.window_height
        cx = w // 2

        # Fundo geral
        surface.fill(cfg.color_background)

        # Leito do corredor
        track_left = cx - cfg.track_lane_width // 2
        track_right = cx + cfg.track_lane_width // 2
        pygame.draw.rect(surface, cfg.color_track,
                         pygame.Rect(track_left, 0, cfg.track_lane_width, h))

        # Paredes laterais
        wall_w = 6
        pygame.draw.rect(surface, cfg.color_hud_border,
                         pygame.Rect(track_left - wall_w, 0, wall_w, h))
        pygame.draw.rect(surface, cfg.color_hud_border,
                         pygame.Rect(track_right, 0, wall_w, h))

        # Linha tracejada central scrollável
        y = -self._period + self.scroll_y
        while y < h:
            pygame.draw.rect(surface, cfg.color_track_line,
                             pygame.Rect(cx - 2, int(y), 4, cfg.track_dash_length))
            y += self._period

        # Vinheta nas bordas do corredor (efeito de profundidade)
        vw = 35
        for i in range(vw):
            alpha = int(55 * (1 - i / vw))
            r, g, b = cfg.color_background
            strip = pygame.Surface((1, h), pygame.SRCALPHA)
            strip.fill((r, g, b, alpha))
            surface.blit(strip, (track_left + i, 0))
            surface.blit(strip, (track_right - vw + i, 0))
