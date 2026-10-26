from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import pygame

from src.simulation.config import SimulationConfig


@dataclass
class HUDData:
    """Dados necessários para renderizar o HUD em um frame."""
    predicted_class_id: int
    class_name: str
    probabilities: tuple        # (P0, P1, P2)
    model_display_name: str
    fps: float
    use_mock: bool = False
    history: deque = field(default_factory=deque)   # deque of (class_id, confidence)
    data_frozen: bool = False


class HUD:
    """
    Painel semi-transparente no canto superior direito.

    Layout:
        ┌──────────────────────────────┐
        │  Modelo: CNN [MOCK]   FPS:60 │
        │  ───────────────────────── │
        │  Esquerda  (grande, colorido)│
        │  Esquerda  ████░░░  67.3%   │
        │  Direita   ██░░░░░  22.1%   │
        │  Neutro    █░░░░░░  10.6%   │
        │  ─── Histórico ──────────── │
        │  [▓][░][▓▓][▓][░░] ... (15) │
        └──────────────────────────────┘

    A surface de fundo é construída uma vez e copiada a cada frame.
    Apenas texto e barras são desenhados dinamicamente.
    """

    PANEL_W = 290
    PANEL_H = 310
    MARGIN = 10
    BAR_W = 130
    BAR_H = 14

    TIMELINE_MAX = 15         # número de predições no histórico visual
    TIMELINE_BLOCK_W = 14     # largura de cada bloco
    TIMELINE_BLOCK_GAP = 2    # espaço entre blocos
    TIMELINE_MAX_H = 42       # altura máxima do bloco (confiança = 100%)
    TIMELINE_MIN_H = 10       # altura mínima do bloco (confiança ≈ 0%)

    _CLASS_NAMES = ("Esquerda", "Direita", "Neutro")
    _CLASS_ABBREV = ("Esq", "Dir", "Neu")

    def __init__(self, config: SimulationConfig):
        self.config = config
        self._px = config.window_width - self.PANEL_W - self.MARGIN
        self._py = self.MARGIN
        self._class_colors = (
            config.color_class_left,
            config.color_class_right,
            config.color_class_neutral,
        )
        self._bg: pygame.Surface | None = None
        self._fl: pygame.font.Font | None = None  # large
        self._fm: pygame.font.Font | None = None  # medium
        self._fs: pygame.font.Font | None = None  # small
        self._ft: pygame.font.Font | None = None  # tiny (timeline labels)

    def init_fonts(self) -> None:
        """Deve ser chamado após pygame.init()."""
        try:
            self._fl = pygame.font.SysFont("Segoe UI", 19, bold=True)
            self._fm = pygame.font.SysFont("Segoe UI", 15)
            self._fs = pygame.font.SysFont("Segoe UI", 12)
            self._ft = pygame.font.SysFont("Segoe UI", 10)
        except Exception:
            self._fl = pygame.font.Font(None, 24)
            self._fm = pygame.font.Font(None, 20)
            self._fs = pygame.font.Font(None, 16)
            self._ft = pygame.font.Font(None, 13)

    def _make_bg(self) -> pygame.Surface:
        r, g, b = self.config.color_hud_bg
        br, bg, bb = self.config.color_hud_border
        surf = pygame.Surface((self.PANEL_W, self.PANEL_H), pygame.SRCALPHA)
        surf.fill((r, g, b, 210))
        pygame.draw.rect(surf, (br, bg, bb, 255),
                         pygame.Rect(0, 0, self.PANEL_W, self.PANEL_H),
                         width=1, border_radius=8)
        return surf

    def draw(self, surface: pygame.Surface, data: HUDData) -> None:
        if self._bg is None:
            self._bg = self._make_bg()

        panel = self._bg.copy()

        # --- Cabeçalho ---
        mock_tag = " [MOCK]" if data.use_mock else ""
        self._text(panel, f"Modelo: {data.model_display_name}{mock_tag}",
                   (10, 10), self._fs, self.config.color_text_secondary)

        fps_color = self.config.color_fps if data.fps >= 45 else (220, 100, 100)
        fps_surf = self._fs.render(f"FPS: {data.fps:.0f}", True, fps_color)
        panel.blit(fps_surf, (self.PANEL_W - fps_surf.get_width() - 10, 10))

        # Badge de freeze
        if data.data_frozen:
            badge = self._fs.render("● FREEZE", True, (255, 180, 0))
            panel.blit(badge, (self.PANEL_W - badge.get_width() - 10, 22))

        # Linha separadora
        sep_color = (255, 180, 0) if data.data_frozen else self.config.color_hud_border
        pygame.draw.line(panel, sep_color,
                         (10, 30), (self.PANEL_W - 10, 30), 1)

        # --- Classe atual ---
        cls_color = self._class_colors[data.predicted_class_id]
        self._text(panel, data.class_name, (10, 38), self._fl, cls_color)

        lbl = self._fs.render("Classe Atual", True, self.config.color_text_secondary)
        panel.blit(lbl, (self.PANEL_W - lbl.get_width() - 10, 42))

        # --- Barras de probabilidade ---
        for i, (cname, prob) in enumerate(zip(self._CLASS_NAMES, data.probabilities)):
            row_y = 78 + i * 34
            color = self._class_colors[i]

            self._text(panel, cname, (10, row_y), self._fs,
                       self.config.color_text_primary)

            # Fundo da barra
            pygame.draw.rect(panel, (40, 50, 70),
                             pygame.Rect(80, row_y + 1, self.BAR_W, self.BAR_H),
                             border_radius=3)

            # Preenchimento proporcional
            fill_w = max(2, int(self.BAR_W * float(prob)))
            pygame.draw.rect(panel, color,
                             pygame.Rect(80, row_y + 1, fill_w, self.BAR_H),
                             border_radius=3)

            # Destaque na classe predita
            if i == data.predicted_class_id:
                pygame.draw.rect(panel, (255, 255, 255),
                                 pygame.Rect(80, row_y + 1, self.BAR_W, self.BAR_H),
                                 width=1, border_radius=3)

            # Percentual
            pct = self._fs.render(f"{float(prob) * 100:.1f}%", True,
                                  self.config.color_text_primary)
            panel.blit(pct, (80 + self.BAR_W + 6, row_y))

        # --- Separador timeline ---
        sep_y = 78 + 3 * 34 + 6
        pygame.draw.line(panel, self.config.color_hud_border,
                         (10, sep_y), (self.PANEL_W - 10, sep_y), 1)

        lbl_t = self._fs.render("Histórico", True, self.config.color_text_secondary)
        panel.blit(lbl_t, (10, sep_y + 5))

        # --- Timeline de decisões ---
        self._draw_timeline(panel, data.history, sep_y + 22)

        surface.blit(panel, (self._px, self._py))

    def _draw_timeline(
        self,
        panel: pygame.Surface,
        history: deque,
        top_y: int,
    ) -> None:
        """
        Desenha o histórico de predições como blocos coloridos.

        Cada bloco tem altura proporcional à confiança (TIMELINE_MIN_H a TIMELINE_MAX_H).
        O bloco mais recente fica à direita. Labels abreviados abaixo.
        """
        items = list(history)   # mais antigo [0] → mais recente [-1]
        n = len(items)
        if n == 0:
            no_data = self._ft.render("aguardando predições…", True,
                                      self.config.color_text_secondary)
            panel.blit(no_data, (10, top_y + 20))
            return

        # Área disponível: panel_w - 20px margens
        area_w = self.PANEL_W - 20
        block_w = self.TIMELINE_BLOCK_W
        gap = self.TIMELINE_BLOCK_GAP
        baseline_y = top_y + self.TIMELINE_MAX_H + 14  # baseline para texto

        # Alinha os blocos à direita
        total_w = n * (block_w + gap) - gap
        start_x = 10 + max(0, area_w - total_w)

        for idx, (class_id, conf) in enumerate(items):
            bx = start_x + idx * (block_w + gap)
            block_h = int(
                self.TIMELINE_MIN_H + conf * (self.TIMELINE_MAX_H - self.TIMELINE_MIN_H)
            )
            by = baseline_y - block_h

            color = (
                self.config.color_class_left,
                self.config.color_class_right,
                self.config.color_class_neutral,
            )[class_id]

            # Bloco colorido
            pygame.draw.rect(panel, color,
                             pygame.Rect(bx, by, block_w, block_h),
                             border_radius=2)

            # Destaque no bloco mais recente
            if idx == n - 1:
                pygame.draw.rect(panel, (255, 255, 255),
                                 pygame.Rect(bx, by, block_w, block_h),
                                 width=1, border_radius=2)

            # Abreviação abaixo (apenas para cada 3 blocos para não poluir)
            if n <= 8 or idx % 3 == 0:
                abbrev = self._CLASS_ABBREV[class_id]
                t = self._ft.render(abbrev, True, self.config.color_text_secondary)
                panel.blit(t, (bx + block_w // 2 - t.get_width() // 2, baseline_y + 2))

    @staticmethod
    def _text(surface: pygame.Surface, text: str, pos: tuple,
              font: pygame.font.Font, color: tuple) -> None:
        surface.blit(font.render(text, True, color), pos)
