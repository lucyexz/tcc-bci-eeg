from __future__ import annotations

from dataclasses import dataclass

import pygame

from src.simulation.config import SimulationConfig


@dataclass
class VehicleState:
    x: float
    y: float
    angle: float = 0.0   # graus; 0 = reto, negativo = esquerda, positivo = direita
    last_class_id: int = 2
    last_confidence: float = 0.8


class Vehicle:
    """
    Veículo 2D visto de cima (top-down) que reage aos comandos BCI.

    O corredor scrolla verticalmente, então y permanece fixo.
    Apenas x (lateral) e angle (esterçamento visual) mudam.

    Desenhado inteiramente com primitivas Pygame — sem assets externos.
    A surface rotacionada é cacheada e só reconstruída quando angle muda >0.1°.
    """

    BODY_W = 36
    BODY_H = 52
    BOUNCE_MARGIN = 40   # px a partir da borda onde começa a repulsão suave

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.state = VehicleState(
            x=float(config.window_width // 2),
            y=float(config.window_height * 0.65),
        )
        self._cached_surface: pygame.Surface | None = None
        self._cached_angle: float = float("nan")

    def apply_command(self, class_id: int, confidence: float = 0.8) -> None:
        """
        Atualiza posição e ângulo a partir da classe predita.
            0 = Esquerda, 1 = Direita, 2 = Neutro
        """
        cfg = self.config
        self.state.last_class_id = class_id
        self.state.last_confidence = confidence

        half = cfg.track_lane_width // 2
        cx = cfg.window_width // 2
        margin = self.BODY_W // 2 + 10
        left_bound = cx - half + margin
        right_bound = cx + half - margin

        if class_id == 0:
            self.state.x -= cfg.vehicle_lateral_speed
            self.state.angle = max(
                self.state.angle - cfg.vehicle_turn_rate, -cfg.vehicle_max_angle
            )
        elif class_id == 1:
            self.state.x += cfg.vehicle_lateral_speed
            self.state.angle = min(
                self.state.angle + cfg.vehicle_turn_rate, cfg.vehicle_max_angle
            )
        else:
            if abs(self.state.angle) <= cfg.vehicle_turn_rate * 0.5:
                self.state.angle = 0.0
            elif self.state.angle > 0:
                self.state.angle -= cfg.vehicle_turn_rate * 0.5
            else:
                self.state.angle += cfg.vehicle_turn_rate * 0.5

        # Bounce suave: repulsão proporcional à penetração na zona de borda
        left_soft = left_bound + self.BOUNCE_MARGIN
        right_soft = right_bound - self.BOUNCE_MARGIN

        if self.state.x < left_soft:
            penetration = (left_soft - self.state.x) / self.BOUNCE_MARGIN
            self.state.x += penetration * cfg.vehicle_lateral_speed * 0.9
        elif self.state.x > right_soft:
            penetration = (self.state.x - right_soft) / self.BOUNCE_MARGIN
            self.state.x -= penetration * cfg.vehicle_lateral_speed * 0.9

        # Limite físico absoluto
        self.state.x = max(left_bound, min(right_bound, self.state.x))

    def draw(self, surface: pygame.Surface) -> None:
        if self._cached_surface is None or abs(self.state.angle - self._cached_angle) > 0.1:
            self._cached_surface = self._build_surface()
            self._cached_angle = self.state.angle

        rect = self._cached_surface.get_rect(
            center=(int(self.state.x), int(self.state.y))
        )
        surface.blit(self._cached_surface, rect)

    def reset(self) -> None:
        self.state.x = float(self.config.window_width // 2)
        self.state.angle = 0.0
        self.state.last_class_id = 2
        self.state.last_confidence = 0.8
        self._cached_surface = None
        self._cached_angle = float("nan")

    def _build_surface(self) -> pygame.Surface:
        pad = 20
        w, h = self.BODY_W + pad, self.BODY_H + pad
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        cx, cy = w // 2, h // 2

        # Corpo principal
        body = pygame.Rect(cx - self.BODY_W // 2, cy - self.BODY_H // 2,
                           self.BODY_W, self.BODY_H)
        pygame.draw.rect(surf, (200, 210, 230), body, border_radius=6)
        pygame.draw.rect(surf, (100, 120, 160), body, width=2, border_radius=6)

        # Assento/cockpit
        seat = pygame.Rect(cx - 8, cy - 10, 16, 14)
        pygame.draw.ellipse(surf, (80, 100, 140), seat)

        # Rodas (4 cantos)
        ww, wh = 7, 14
        for dx, dy in [
            (-self.BODY_W // 2 - 3, -self.BODY_H // 2 + 4),
            (+self.BODY_W // 2 - 4, -self.BODY_H // 2 + 4),
            (-self.BODY_W // 2 - 3, +self.BODY_H // 2 - 18),
            (+self.BODY_W // 2 - 4, +self.BODY_H // 2 - 18),
        ]:
            pygame.draw.rect(surf, (50, 60, 80),
                             pygame.Rect(cx + dx, cy + dy, ww, wh),
                             border_radius=2)

        # Triângulo de direção (frente do veículo)
        nose = [
            (cx, cy - self.BODY_H // 2 - 6),
            (cx - 5, cy - self.BODY_H // 2 + 2),
            (cx + 5, cy - self.BODY_H // 2 + 2),
        ]
        pygame.draw.polygon(surf, (180, 200, 255), nose)

        return pygame.transform.rotate(surf, -self.state.angle)
