// Tradução de src/simulacao/rendering/eeg_panel.py: painel esquerdo com a entrada do modelo.
//  - Grade 5×5 (eletrodo × banda) do registro mais recente da janela: log da potência padronizado (z),
//    na escala fixa de −3 a +3.
//  - Nos Transformers, abaixo: atenção do [CLS] em cada token de tempo da janela.
import { CONFIG as cfg, fonte, larguraTexto, linha, retangulo, rgb, texto, type RGB } from "./config";

const CHANNELS = ["AF3", "T7", "Pz", "T8", "AF4"];
const BANDS = ["Theta", "Alpha", "BetaL", "BetaH", "Gamma"];
const ESCALA_Z = 3.0;

const CELL = 34;
const ROW_LABEL_W = 32;
const COL_LABEL_H = 20;
const MARGIN_LEFT = 16;
const MARGIN_TOP = 12;
const LEGEND_H = 16;
const ATTN_H = 70;

const FL = fonte(13, true);
const FS = fonte(11);
const FT = fonte(10);

// Escala divergente das figuras: marrom (abaixo da média) ↔ cinza ↔ azul (acima)
const DIVERGENTE = ["#5c3d14", "#8a5a22", "#b98546", "#dcc29b", "#f0efec", "#9ec5f4", "#3987e5", "#184f95", "#0d366b"]
  .map((c) => [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16)) as unknown as RGB);

function divergente(v: number): RGB {
  const x = Math.max(0, Math.min(1, v)) * (DIVERGENTE.length - 1);
  const i = Math.min(Math.trunc(x), DIVERGENTE.length - 2);
  const t = x - i;
  const a = DIVERGENTE[i];
  const b = DIVERGENTE[i + 1];
  return [0, 1, 2].map((k) => Math.trunc(a[k] + t * (b[k] - a[k]))) as unknown as RGB;
}

const sinal = (v: number) => `${v >= 0 ? "+" : "-"}${Math.abs(v).toFixed(1)}`.replace(".", ",");

export function desenharPainelEeg(ctx: CanvasRenderingContext2D, registro: readonly number[] | null,
  atencao: readonly number[] | null) {
  const px = cfg.eeg_panel_x;
  const pw = cfg.eeg_panel_width;
  const ph = cfg.window_height;

  ctx.fillStyle = rgb(cfg.color_hud_bg, 200 / 255);
  ctx.fillRect(px, 0, pw, ph);
  retangulo(ctx, px, 0, pw, ph, cfg.color_hud_border, 0, true);

  if (registro === null) {
    const linhas = ["Modo mock", "", "Sem EEG real:", "predições sorteadas"];
    linhas.forEach((l, i) => {
      texto(ctx, l, px + pw / 2 - Math.floor(larguraTexto(ctx, l, FS) / 2), ph / 2 - 30 + i * 20, FS,
        cfg.color_text_secondary);
    });
    return;
  }
  const fim = desenharGrade(ctx, px, registro, MARGIN_TOP);
  if (atencao) desenharAtencao(ctx, px, atencao, fim + 28);
}

function desenharGrade(ctx: CanvasRenderingContext2D, px: number, z: readonly number[], startY: number): number {
  const cs = CELL;
  const ml = MARGIN_LEFT;
  texto(ctx, "Entrada do modelo", px + ml, startY, FL, cfg.color_text_primary);
  texto(ctx, "registro atual: eletrodo × banda (z)", px + ml, startY + 18, FS, cfg.color_text_secondary);

  const gridTop = startY + 36;
  BANDS.forEach((lbl, j) => {
    const cx = px + ml + ROW_LABEL_W + j * cs + cs / 2;
    const s = lbl.slice(0, 5);
    texto(ctx, s, cx - Math.floor(larguraTexto(ctx, s, FT) / 2), gridTop, FT, cfg.color_text_secondary);
  });

  const cellsTop = gridTop + COL_LABEL_H;
  CHANNELS.forEach((rowLbl, i) => {
    const ry = cellsTop + i * cs;
    texto(ctx, rowLbl, px + ml, ry + cs / 2 - 7, FS, cfg.color_text_primary);
    for (let j = 0; j < 5; j++) {
      const v = z[i * 5 + j];
      const pos = (v + ESCALA_Z) / (2 * ESCALA_Z);
      const rx = px + ml + ROW_LABEL_W + j * cs;
      retangulo(ctx, rx + 1, ry + 1, cs - 2, cs - 2, divergente(pos), 3);
      const s = sinal(v);
      const cor: RGB = pos > 0.3 && pos < 0.7 ? [20, 20, 20] : [235, 235, 235];
      texto(ctx, s, rx + cs / 2 - Math.floor(larguraTexto(ctx, s, FT) / 2), ry + cs / 2 - 6, FT, cor);
    }
  });

  const legendY = cellsTop + 5 * cs + 8;
  const legendW = cfg.eeg_panel_width - ml * 2;
  for (let k = 0; k < legendW; k++) {
    ctx.fillStyle = rgb(divergente(k / legendW));
    ctx.fillRect(px + ml + k, legendY, 1, LEGEND_H + 1);
  }
  const y = legendY + LEGEND_H + 2;
  texto(ctx, "−3", px + ml, y, FT, cfg.color_text_secondary);
  texto(ctx, "0", px + ml + legendW / 2 - Math.floor(larguraTexto(ctx, "0", FT) / 2), y, FT, cfg.color_text_secondary);
  texto(ctx, "+3", px + ml + legendW - larguraTexto(ctx, "+3", FT), y, FT, cfg.color_text_secondary);
  return legendY + LEGEND_H + 16;
}

function desenharAtencao(ctx: CanvasRenderingContext2D, px: number, atencao: readonly number[], top: number) {
  const ml = MARGIN_LEFT;
  linha(ctx, px + ml, top - 12, px + cfg.eeg_panel_width - ml, top - 12, cfg.color_hud_border);
  texto(ctx, "Atenção do [CLS]", px + ml, top, FL, cfg.color_text_primary);
  texto(ctx, "para que trecho da janela o modelo olhou", px + ml, top + 18, FS, cfg.color_text_secondary);

  const areaW = cfg.eeg_panel_width - 2 * ml;
  const base = top + 40 + ATTN_H;
  const n = atencao.length;
  const uniforme = 1 / n;
  const maximo = Math.max(Math.max(...atencao), 2 * uniforme);
  const larg = areaW / n;
  ctx.fillStyle = rgb(cfg.color_class_left);
  atencao.forEach((a, k) => {
    const h = Math.max(1, Math.trunc((ATTN_H * a) / maximo));
    const x0 = px + ml + Math.trunc(k * larg);
    const x1 = px + ml + Math.trunc((k + 1) * larg);
    ctx.fillRect(x0, base - h, Math.max(1, x1 - x0 - 1), h);
  });
  const yu = base - Math.trunc((ATTN_H * uniforme) / maximo);
  linha(ctx, px + ml, yu, px + ml + areaW, yu, cfg.color_text_secondary);
  texto(ctx, "mais antigo", px + ml, base + 3, FT, cfg.color_text_secondary);
  texto(ctx, "agora", px + ml + areaW - larguraTexto(ctx, "agora", FT), base + 3, FT, cfg.color_text_secondary);
  texto(ctx, "linha: atenção uniforme", px + ml, base + 16, FT, cfg.color_text_secondary);
}
