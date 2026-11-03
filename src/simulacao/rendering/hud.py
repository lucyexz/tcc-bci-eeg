from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import pygame

from src.simulacao.config import SimulationConfig


@dataclass
class HUDData:
    """Dados necessários para renderizar o HUD em um frame."""
    predicted_class_id: int
    class_name: str
    probabilities: tuple        # (P0, P1, P2)
    model_display_name: str
    fps: float
    use_mock: bool = False
    history: deque = field(default_factory=deque)   # deque de (classe prevista, confiança, classe verdadeira)
    data_frozen: bool = False
    true_class_id: int | None = None   # só no modo real
    acertos: int = 0
    total: int = 0
    progresso: str = ""


class HUD:
    """
    Painel semi-transparente no canto superior direito.

    Layout:
        ┌──────────────────────────────┐
        │  Modelo: CNN [MOCK]   FPS:60 │
        │  ───────────────────────── │
        │  Esquerda  (grande, colorido)│
        │  Verdade: Direita    errou   │
        │  Acerto acumulado: 46% (n)   │
        │  Esquerda  ████░░░  67.3%   │
        │  Direita   ██░░░░░  22.1%   │
        │  Neutro    █░░░░░░  10.6%   │
        │  ─── Histórico ──────────── │
        │  [▓][░][▓▓][▓][░░] ... (15) │
        │  ▬ ▬ ▬ ▬ ▬ (classe verdadeira)│
        │  user_a · bloco 31 · janela  │
        └──────────────────────────────┘

    A surface de fundo é construída uma vez e copiada a cada frame.
    Apenas texto e barras são desenhados dinamicamente.
    """

    PANEL_W = 290
    PANEL_H = 345
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

        fps_surf = self._fs.render(f"FPS: {data.fps:.0f}", True, self.config.color_text_secondary)
        panel.blit(fps_surf, (self.PANEL_W - fps_surf.get_width() - 10, 10))

        # Badge de freeze
        if data.data_frozen:
            badge = self._fs.render("● CONGELADO", True, self.config.color_text_primary)
            panel.blit(badge, (self.PANEL_W - badge.get_width() - 10, 22))

        # Linha separadora
        sep_color = self.config.color_text_primary if data.data_frozen else self.config.color_hud_border
        pygame.draw.line(panel, sep_color,
                         (10, 30), (self.PANEL_W - 10, 30), 1)

        # --- Classe atual ---
        cls_color = self._class_colors[data.predicted_class_id]
        self._text(panel, data.class_name, (10, 38), self._fl, cls_color)

        lbl = self._fs.render("Classe prevista", True, self.config.color_text_secondary)
        panel.blit(lbl, (self.PANEL_W - lbl.get_width() - 10, 42))

        # --- Verdade e acerto acumulado (modo real) ---
        if data.true_class_id is not None:
            verdade = self._CLASS_NAMES[data.true_class_id]
            self._text(panel, "Verdade:", (10, 68), self._fs, self.config.color_text_secondary)
            self._text(panel, verdade, (62, 68), self._fs, self._class_colors[data.true_class_id])
            acertou = data.true_class_id == data.predicted_class_id
            marca = self._fs.render("acertou" if acertou else "errou", True,
                                    self.config.color_text_primary if acertou else self.config.color_text_secondary)
            panel.blit(marca, (self.PANEL_W - marca.get_width() - 10, 68))
            taxa = f"{100 * data.acertos / data.total:.1f}".replace(".", ",") if data.total else "—"
            self._text(panel, f"Acerto acumulado: {taxa}% ({data.acertos}/{data.total})", (10, 86),
                       self._fs, self.config.color_text_primary)

        # --- Barras de probabilidade ---
        for i, (cname, prob) in enumerate(zip(self._CLASS_NAMES, data.probabilities)):
            row_y = 112 + i * 30
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
        sep_y = 112 + 3 * 30 + 2
        pygame.draw.line(panel, self.config.color_hud_border,
                         (10, sep_y), (self.PANEL_W - 10, sep_y), 1)

        lbl_t = self._fs.render("Histórico", True, self.config.color_text_secondary)
        panel.blit(lbl_t, (10, sep_y + 5))

        # --- Timeline de decisões ---
        self._draw_timeline(panel, data.history, sep_y + 22)

        if data.progresso:
            self._text(panel, data.progresso, (10, self.PANEL_H - 20), self._ft, self.config.color_text_secondary)

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

        color_of = self._class_colors
        for idx, (class_id, conf, true_id) in enumerate(items):
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

            # Faixa fina com a classe verdadeira, para comparar com a prevista (modo real)
            if true_id is not None:
                pygame.draw.rect(panel, color_of[true_id], pygame.Rect(bx, baseline_y + 18, block_w, 5),
                                 border_radius=1)

        if any(item[2] is not None for item in items):
            t = self._ft.render("faixa de baixo: classe verdadeira", True, self.config.color_text_secondary)
            panel.blit(t, (10, baseline_y + 26))

    @staticmethod
    def _text(surface: pygame.Surface, text: str, pos: tuple,
              font: pygame.font.Font, color: tuple) -> None:
        surface.blit(font.render(text, True, color), pos)
