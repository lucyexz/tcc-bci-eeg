from __future__ import annotations

import numpy as np
import pygame

from src.simulacao.config import SimulationConfig

_CHANNELS = ("AF3", "T7", "Pz", "T8", "AF4")
_BANDS = ("Theta", "Alpha", "BetaL", "BetaH", "Gamma")
_ESCALA_Z = 3.0  # a grade vai de −3 a +3 desvios-padrão (cores fixas, comparáveis ao longo do tempo)


# Escala divergente das figuras do trabalho: marrom (abaixo da média) ↔ cinza ↔ azul (acima)
_DIVERGENTE = ["#5c3d14", "#8a5a22", "#b98546", "#dcc29b", "#f0efec", "#9ec5f4", "#3987e5", "#184f95", "#0d366b"]
_DIVERGENTE_RGB = [tuple(int(c[i:i + 2], 16) for i in (1, 3, 5)) for c in _DIVERGENTE]


def _divergente(v: float) -> tuple[int, int, int]:
    """0 → marrom escuro, 0,5 → cinza claro, 1 → azul escuro (interpolação linear)."""
    v = max(0.0, min(1.0, v)) * (len(_DIVERGENTE_RGB) - 1)
    i = min(int(v), len(_DIVERGENTE_RGB) - 2)
    t = v - i
    a, b = _DIVERGENTE_RGB[i], _DIVERGENTE_RGB[i + 1]
    return tuple(int(a[k] + t * (b[k] - a[k])) for k in range(3))


class EEGPanel:
    """
    Painel esquerdo com a entrada do modelo.

    - Grade 5×5 (eletrodo × banda) com o registro mais recente da janela: log da
      potência padronizado (z), na escala fixa de −3 a +3.
    - Nos Transformers, abaixo: atenção do token [CLS] em cada token de tempo da
      janela (do mais antigo ao atual), ou seja, para que trecho o modelo olhou.
    """

    CELL = 34
    ROW_LABEL_W = 32
    COL_LABEL_H = 20
    MARGIN_LEFT = 16
    MARGIN_TOP = 12
    LEGEND_H = 16
    ATTN_H = 70

    def __init__(self, config: SimulationConfig):
        self.config = config
        self._fl: pygame.font.Font | None = None
        self._fs: pygame.font.Font | None = None
        self._ft: pygame.font.Font | None = None

    def init_fonts(self) -> None:
        try:
            self._fl = pygame.font.SysFont("Segoe UI", 13, bold=True)
            self._fs = pygame.font.SysFont("Segoe UI", 11)
            self._ft = pygame.font.SysFont("Segoe UI", 10)
        except Exception:
            self._fl = pygame.font.Font(None, 16)
            self._fs = pygame.font.Font(None, 14)
            self._ft = pygame.font.Font(None, 13)

    def draw(self, surface: pygame.Surface, registro: np.ndarray | None, atencao: np.ndarray | None = None) -> None:
        cfg = self.config
        px, pw, ph = cfg.eeg_panel_x, cfg.eeg_panel_width, cfg.window_height

        bg = pygame.Surface((pw, ph), pygame.SRCALPHA)
        r, g, b = cfg.color_hud_bg
        bg.fill((r, g, b, 200))
        pygame.draw.rect(bg, cfg.color_hud_border, pygame.Rect(0, 0, pw, ph), width=1)
        surface.blit(bg, (px, 0))

        if registro is None:
            self._draw_mock_placeholder(surface, px, pw)
            return

        z = registro.reshape(5, 5).astype(float)
        fim = self._draw_grid(surface, px, z, start_y=self.MARGIN_TOP)
        if atencao is not None:
            self._draw_atencao(surface, px, np.asarray(atencao, dtype=float), fim + 28)

    # ── Primitivas de renderização ────────────────────────────────────────────

    def _draw_grid(self, surface: pygame.Surface, px: int, z: np.ndarray, start_y: int) -> int:
        """Desenha a grade e a legenda; devolve o y do fim."""
        cfg, cs, rl, ml = self.config, self.CELL, self.ROW_LABEL_W, self.MARGIN_LEFT
        surface.blit(self._fl.render("Entrada do modelo", True, cfg.color_text_primary), (px + ml, start_y))
        surface.blit(self._fs.render("registro atual: eletrodo × banda (z)", True, cfg.color_text_secondary),
                     (px + ml, start_y + 18))

        grid_top = start_y + 36
        for j, lbl in enumerate(_BANDS):
            cx = px + ml + rl + j * cs + cs // 2
            t = self._ft.render(lbl[:5], True, cfg.color_text_secondary)
            surface.blit(t, (cx - t.get_width() // 2, grid_top))

        cells_top = grid_top + self.COL_LABEL_H
        for i, row_lbl in enumerate(_CHANNELS):
            ry = cells_top + i * cs
            t = self._fs.render(row_lbl, True, cfg.color_text_primary)
            surface.blit(t, (px + ml, ry + cs // 2 - t.get_height() // 2))
            for j in range(5):
                v = float(z[i, j])
                pos = (v + _ESCALA_Z) / (2 * _ESCALA_Z)
                rx = px + ml + rl + j * cs
                pygame.draw.rect(surface, _divergente(pos), pygame.Rect(rx + 1, ry + 1, cs - 2, cs - 2), border_radius=3)
                val = self._ft.render(f"{v:+.1f}".replace(".", ","), True,
                                      (20, 20, 20) if 0.3 < pos < 0.7 else (235, 235, 235))
                surface.blit(val, (rx + cs // 2 - val.get_width() // 2, ry + cs // 2 - val.get_height() // 2))

        legend_y = cells_top + 5 * cs + 8
        legend_w = cfg.eeg_panel_width - ml * 2
        for k in range(legend_w):
            pygame.draw.line(surface, _divergente(k / legend_w), (px + ml + k, legend_y),
                             (px + ml + k, legend_y + self.LEGEND_H))
        low = self._ft.render("−3", True, cfg.color_text_secondary)
        mid = self._ft.render("0", True, cfg.color_text_secondary)
        high = self._ft.render("+3", True, cfg.color_text_secondary)
        surface.blit(low, (px + ml, legend_y + self.LEGEND_H + 2))
        surface.blit(mid, (px + ml + legend_w // 2 - mid.get_width() // 2, legend_y + self.LEGEND_H + 2))
        surface.blit(high, (px + ml + legend_w - high.get_width(), legend_y + self.LEGEND_H + 2))
        return legend_y + self.LEGEND_H + 16

    def _draw_atencao(self, surface: pygame.Surface, px: int, atencao: np.ndarray, top: int) -> None:
        cfg, ml = self.config, self.MARGIN_LEFT
        pygame.draw.line(surface, cfg.color_hud_border, (px + ml, top - 12),
                         (px + cfg.eeg_panel_width - ml, top - 12), 1)
        surface.blit(self._fl.render("Atenção do [CLS]", True, cfg.color_text_primary), (px + ml, top))
        surface.blit(self._fs.render("para que trecho da janela o modelo olhou", True, cfg.color_text_secondary),
                     (px + ml, top + 18))

        area_w = cfg.eeg_panel_width - 2 * ml
        base = top + 40 + self.ATTN_H
        n = len(atencao)
        uniforme = 1.0 / n
        maximo = max(float(atencao.max()), 2 * uniforme)
        larg = area_w / n
        for k, a in enumerate(atencao):
            h = max(1, int(self.ATTN_H * float(a) / maximo))
            x0 = px + ml + int(k * larg)
            x1 = px + ml + int((k + 1) * larg)
            pygame.draw.rect(surface, cfg.color_class_left, pygame.Rect(x0, base - h, max(1, x1 - x0 - 1), h))
        # linha da atenção uniforme
        yu = base - int(self.ATTN_H * uniforme / maximo)
        pygame.draw.line(surface, cfg.color_text_secondary, (px + ml, yu), (px + ml + area_w, yu), 1)
        surface.blit(self._ft.render("mais antigo", True, cfg.color_text_secondary), (px + ml, base + 3))
        agora = self._ft.render("agora", True, cfg.color_text_secondary)
        surface.blit(agora, (px + ml + area_w - agora.get_width(), base + 3))
        unif = self._ft.render("linha: atenção uniforme", True, cfg.color_text_secondary)
        surface.blit(unif, (px + ml, base + 16))

    def _draw_mock_placeholder(self, surface: pygame.Surface, px: int, pw: int) -> None:
        cfg = self.config
        lines = ["Modo mock", "", "Sem EEG real:", "predições sorteadas"]
        for i, line in enumerate(lines):
            t = self._fs.render(line, True, cfg.color_text_secondary)
            surface.blit(t, (px + pw // 2 - t.get_width() // 2, cfg.window_height // 2 - 30 + i * 20))
