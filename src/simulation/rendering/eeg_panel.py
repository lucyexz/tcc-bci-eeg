from __future__ import annotations

import numpy as np
import pygame

from src.simulation.config import SimulationConfig

_CHANNELS = ("AF3", "AF4", "T7", "T8", "Pz")
_BANDS = ("Delta", "Theta", "Alpha", "Beta", "Gamma")


def _coolwarm(v: float) -> tuple[int, int, int]:
    """Mapa de cor cool-warm: azul (0) → branco (0.5) → vermelho (1)."""
    v = max(0.0, min(1.0, v))
    if v < 0.5:
        t = v * 2
        r = int(59 + t * (221 - 59))
        g = int(76 + t * (221 - 76))
        b = int(192 + t * (221 - 192))
    else:
        t = (v - 0.5) * 2
        r = int(221 + t * (180 - 221))
        g = int(221 + t * (4 - 221))
        b = int(221 + t * (38 - 221))
    return (r, g, b)


class EEGPanel:
    """
    Painel esquerdo com visualização das features EEG de entrada.

    Modos:
        single   — grade 5×5 (canais × bandas), usada para CNN/LSTM/GRU.
        dual     — heatmap (canais × bandas) empilhado com attention weights
                   (canal → canal), exclusivo do Transformer quando o extrator
                   foi inicializado com sucesso.

    O painel é renderizado sobre a surface principal.
    """

    # Tamanho de célula para cada modo
    CELL_SINGLE = 34    # single grid, maior
    CELL_DUAL   = 26    # dual grid, compacto

    ROW_LABEL_W = 32
    COL_LABEL_H = 20
    MARGIN_LEFT = 16
    MARGIN_TOP  = 12    # topo do primeiro grid
    LEGEND_H    = 16
    DUAL_GAP    = 20    # espaço entre os dois grids no modo dual

    def __init__(self, config: SimulationConfig):
        self.config = config
        self._attn_fn = None
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

    def try_init_attention(self, model) -> bool:
        """
        Tenta construir o extrator de atenção do Transformer carregado.
        Retorna True em caso de sucesso.
        """
        try:
            import tensorflow as tf

            mha_layers = [
                l for l in model.layers
                if isinstance(l, tf.keras.layers.MultiHeadAttention)
            ]
            if not mha_layers:
                print("[EEGPanel] Nenhuma camada MHA encontrada no modelo.")
                return False

            first_mha = mha_layers[0]

            all_dense = [
                l for l in model.layers
                if isinstance(l, tf.keras.layers.Dense)
            ]
            all_ln = [
                l for l in model.layers
                if isinstance(l, tf.keras.layers.LayerNormalization)
            ]
            pos_emb_layers = [
                l for l in model.layers
                if type(l).__name__ == "_LearnablePositionalEmbedding"
            ]
            cls_layers = [
                l for l in model.layers
                if type(l).__name__ == "_PrependCLSToken"
            ]

            if not (all_dense and all_ln and pos_emb_layers and cls_layers):
                print("[EEGPanel] Layers esperadas não encontradas no modelo.")
                return False

            embed_dense = all_dense[0]
            pos_emb     = pos_emb_layers[0]
            cls_prepend = cls_layers[0]
            pre_norm    = all_ln[0]

            # Sem @tf.function para evitar problemas de tracing com custom layers
            def _extract(x_np: np.ndarray) -> np.ndarray:
                x_in = tf.cast(x_np.reshape(1, 5, 5), tf.float32)
                x    = embed_dense(x_in)
                x    = pos_emb(x)
                x    = cls_prepend(x)
                x_n  = pre_norm(x)
                _, scores = first_mha(
                    x_n, x_n,
                    return_attention_scores=True,
                    training=False,
                )
                avg = tf.reduce_mean(scores, axis=1)[0].numpy()  # (6, 6)
                return avg[1:, 1:]  # (5, 5) — exclui token CLS

            test = _extract(np.zeros(25, dtype=np.float32))
            if test.shape != (5, 5):
                print(f"[EEGPanel] Shape inesperado: {test.shape}")
                return False

            self._attn_fn = _extract
            print("[EEGPanel] Attention extractor inicializado com sucesso.")
            return True

        except Exception as e:
            import traceback
            print(f"[EEGPanel] Falha ao inicializar attention extractor: {e}")
            traceback.print_exc()
            self._attn_fn = None
            return False

    def draw(
        self,
        surface: pygame.Surface,
        raw_x: np.ndarray | None,
        model_name: str = "",
    ) -> None:
        cfg = self.config
        px  = cfg.eeg_panel_x
        pw  = cfg.eeg_panel_width
        ph  = cfg.window_height

        # Fundo do painel
        bg = pygame.Surface((pw, ph), pygame.SRCALPHA)
        r, g, b = cfg.color_hud_bg
        bg.fill((r, g, b, 200))
        pygame.draw.rect(bg, cfg.color_hud_border,
                         pygame.Rect(0, 0, pw, ph), width=1)
        surface.blit(bg, (px, 0))

        if raw_x is None:
            self._draw_mock_placeholder(surface, px, pw)
            return

        # Heatmap (normalizado para 0-1)
        heatmap = raw_x.reshape(5, 5).astype(float)
        mn, mx  = heatmap.min(), heatmap.max()
        heatmap = (heatmap - mn) / (mx - mn) if mx > mn else np.zeros_like(heatmap)

        dual = (model_name == "transformer" and self._attn_fn is not None)

        if dual:
            try:
                attn = self._attn_fn(raw_x)  # (5, 5)
                self._draw_dual(surface, px, heatmap, attn)
                return
            except Exception:
                dual = False

        # Modo single (CNN / LSTM / GRU ou fallback)
        self._draw_grid(
            surface, px, heatmap,
            row_labels=_CHANNELS, col_labels=_BANDS,
            title="Input do Modelo", subtitle="canal × banda",
            start_y=self.MARGIN_TOP,
            cell_size=self.CELL_SINGLE,
            show_legend=True,
        )

    # ── Modo dual ────────────────────────────────────────────────────────────

    def _draw_dual(
        self,
        surface: pygame.Surface,
        px: int,
        heatmap: np.ndarray,
        attn: np.ndarray,
    ) -> None:
        cs  = self.CELL_DUAL
        ml  = self.MARGIN_LEFT
        mt  = self.MARGIN_TOP

        # Altura de cada bloco: título(20) + subtítulo(16) + col_labels(COL_LABEL_H) + 5×cs
        block_h = 20 + 16 + self.COL_LABEL_H + 5 * cs

        # Grid 1 — heatmap
        y1 = mt
        self._draw_grid(
            surface, px, heatmap,
            row_labels=_CHANNELS, col_labels=_BANDS,
            title="Input do Modelo", subtitle="canal × banda",
            start_y=y1,
            cell_size=cs,
            show_legend=False,
        )

        # Separador
        sep_y = y1 + block_h + self.DUAL_GAP // 2
        pygame.draw.line(
            surface, self.config.color_hud_border,
            (px + ml, sep_y), (px + self.config.eeg_panel_width - ml, sep_y), 1,
        )

        # Grid 2 — attention weights
        y2 = sep_y + self.DUAL_GAP // 2 + 4
        self._draw_grid(
            surface, px, attn,
            row_labels=_CHANNELS, col_labels=_CHANNELS,
            title="Atenção (Transformer)", subtitle="canal → canal",
            start_y=y2,
            cell_size=cs,
            show_legend=True,
        )

    # ── Primitivas de renderização ────────────────────────────────────────────

    def _draw_grid(
        self,
        surface: pygame.Surface,
        px: int,
        matrix: np.ndarray,
        row_labels: tuple,
        col_labels: tuple,
        title: str,
        subtitle: str,
        start_y: int,
        cell_size: int,
        show_legend: bool = True,
    ) -> None:
        cfg = self.config
        cs  = cell_size
        rl  = self.ROW_LABEL_W
        cl  = self.COL_LABEL_H
        ml  = self.MARGIN_LEFT

        # Título e subtítulo
        if self._fl:
            surface.blit(
                self._fl.render(title, True, cfg.color_text_primary),
                (px + ml, start_y),
            )
        if self._fs:
            surface.blit(
                self._fs.render(subtitle, True, cfg.color_text_secondary),
                (px + ml, start_y + 18),
            )

        grid_top = start_y + 20 + 16  # após título + subtítulo

        # Labels das colunas
        if self._ft:
            for j, lbl in enumerate(col_labels):
                cx = px + ml + rl + j * cs + cs // 2
                t  = self._ft.render(lbl[:5], True, cfg.color_text_secondary)
                surface.blit(t, (cx - t.get_width() // 2, grid_top))

        cells_top = grid_top + cl

        # Células + labels das linhas
        for i, row_lbl in enumerate(row_labels):
            ry = cells_top + i * cs

            if self._fs:
                t = self._fs.render(row_lbl, True, cfg.color_text_primary)
                surface.blit(t, (px + ml, ry + cs // 2 - t.get_height() // 2))

            for j in range(5):
                v     = float(matrix[i, j])
                color = _coolwarm(v)
                rx    = px + ml + rl + j * cs
                pygame.draw.rect(surface, color,
                                 pygame.Rect(rx + 1, ry + 1, cs - 2, cs - 2),
                                 border_radius=3)

                # Valor numérico (só se a célula for grande o suficiente)
                if self._ft and cs >= 26:
                    val = self._ft.render(
                        f"{v:.2f}", True,
                        (20, 20, 20) if v > 0.5 else (220, 220, 220),
                    )
                    surface.blit(val, (
                        rx + cs // 2 - val.get_width() // 2,
                        ry + cs // 2 - val.get_height() // 2,
                    ))

        if show_legend:
            legend_y = cells_top + 5 * cs + 8
            self._draw_legend(surface, px, legend_y)

    def _draw_legend(self, surface: pygame.Surface, px: int, top_y: int) -> None:
        cfg      = self.config
        ml       = self.MARGIN_LEFT
        legend_w = cfg.eeg_panel_width - ml * 2

        for i in range(legend_w):
            color = _coolwarm(i / legend_w)
            pygame.draw.line(surface, color,
                             (px + ml + i, top_y),
                             (px + ml + i, top_y + self.LEGEND_H))

        if self._ft:
            low  = self._ft.render("baixo", True, cfg.color_text_secondary)
            high = self._ft.render("alto",  True, cfg.color_text_secondary)
            surface.blit(low,  (px + ml, top_y + self.LEGEND_H + 2))
            surface.blit(high, (px + ml + legend_w - high.get_width(),
                                top_y + self.LEGEND_H + 2))

    def _draw_mock_placeholder(self, surface: pygame.Surface, px: int, pw: int) -> None:
        cfg = self.config
        if self._fs:
            lines = ["Modo Mock", "", "Sem features", "EEG reais"]
            for i, line in enumerate(lines):
                t = self._fs.render(line, True, cfg.color_text_secondary)
                surface.blit(t, (px + pw // 2 - t.get_width() // 2,
                                 cfg.window_height // 2 - 30 + i * 20))
