// Tradução de src/simulacao/rendering/hud.py: painel no canto superior direito.
import { CONFIG as cfg, CORES_CLASSES, fonte, larguraTexto, linha, NOMES_CLASSES, retangulo, rgb, texto } from "./config";

export interface ItemHistorico {
  classe: number; // prevista
  confianca: number;
  verdade: number | null;
}

export interface DadosHUD {
  predicted_class_id: number;
  probabilities: readonly number[];
  model_display_name: string;
  fps: number;
  use_mock: boolean;
  history: ItemHistorico[];
  data_frozen: boolean;
  true_class_id: number | null;
  acertos: number;
  total: number;
  progresso: string;
}

export const PANEL_W = 290;
const PANEL_H = 345;
const MARGIN = 10;
const BAR_W = 130;
const BAR_H = 14;

export const TIMELINE_MAX = 15; // número de predições no histórico visual
const TIMELINE_BLOCK_W = 14;
const TIMELINE_BLOCK_GAP = 2;
const TIMELINE_MAX_H = 42;
const TIMELINE_MIN_H = 10;

const ABREV = ["Esq", "Dir", "Neu"];
const FL = fonte(19, true);
const FS = fonte(12);
const FT = fonte(10);

export function desenharHUD(ctx: CanvasRenderingContext2D, d: DadosHUD) {
  const px = cfg.window_width - PANEL_W - MARGIN;
  const py = MARGIN;
  ctx.save();
  ctx.translate(px, py);

  // Fundo semitransparente e borda arredondada
  ctx.fillStyle = rgb(cfg.color_hud_bg, 210 / 255);
  ctx.fillRect(0, 0, PANEL_W, PANEL_H);
  retangulo(ctx, 0, 0, PANEL_W, PANEL_H, cfg.color_hud_border, 8, true);

  // Cabeçalho
  texto(ctx, `Modelo: ${d.model_display_name}${d.use_mock ? " [MOCK]" : ""}`, 10, 10, FS, cfg.color_text_secondary);
  const fps = `FPS: ${d.fps.toFixed(0)}`;
  texto(ctx, fps, PANEL_W - larguraTexto(ctx, fps, FS) - 10, 10, FS, cfg.color_text_secondary);

  if (d.data_frozen) {
    const badge = "● CONGELADO";
    texto(ctx, badge, PANEL_W - larguraTexto(ctx, badge, FS) - 10, 22, FS, cfg.color_text_primary);
  }
  linha(ctx, 10, 30, PANEL_W - 10, 30, d.data_frozen ? cfg.color_text_primary : cfg.color_hud_border);

  // Classe atual
  texto(ctx, NOMES_CLASSES[d.predicted_class_id], 10, 38, FL, CORES_CLASSES[d.predicted_class_id]);
  const lbl = "Classe prevista";
  texto(ctx, lbl, PANEL_W - larguraTexto(ctx, lbl, FS) - 10, 42, FS, cfg.color_text_secondary);

  // Verdade e acerto acumulado (modo real)
  if (d.true_class_id !== null) {
    texto(ctx, "Verdade:", 10, 68, FS, cfg.color_text_secondary);
    texto(ctx, NOMES_CLASSES[d.true_class_id], 62, 68, FS, CORES_CLASSES[d.true_class_id]);
    const acertou = d.true_class_id === d.predicted_class_id;
    const marca = acertou ? "acertou" : "errou";
    texto(ctx, marca, PANEL_W - larguraTexto(ctx, marca, FS) - 10, 68, FS,
      acertou ? cfg.color_text_primary : cfg.color_text_secondary);
    const taxa = d.total ? ((100 * d.acertos) / d.total).toFixed(1).replace(".", ",") : "—";
    texto(ctx, `Acerto acumulado: ${taxa}% (${d.acertos}/${d.total})`, 10, 86, FS, cfg.color_text_primary);
  }

  // Barras de probabilidade
  for (let i = 0; i < 3; i++) {
    const rowY = 112 + i * 30;
    const prob = d.probabilities[i];
    texto(ctx, NOMES_CLASSES[i], 10, rowY, FS, cfg.color_text_primary);
    retangulo(ctx, 80, rowY + 1, BAR_W, BAR_H, "rgb(40,50,70)", 3);
    retangulo(ctx, 80, rowY + 1, Math.max(2, Math.trunc(BAR_W * prob)), BAR_H, CORES_CLASSES[i], 3);
    if (i === d.predicted_class_id) retangulo(ctx, 80, rowY + 1, BAR_W, BAR_H, "rgb(255,255,255)", 3, true);
    texto(ctx, `${(prob * 100).toFixed(1)}%`, 80 + BAR_W + 6, rowY, FS, cfg.color_text_primary);
  }

  // Separador e histórico
  const sepY = 112 + 3 * 30 + 2;
  linha(ctx, 10, sepY, PANEL_W - 10, sepY, cfg.color_hud_border);
  texto(ctx, "Histórico", 10, sepY + 5, FS, cfg.color_text_secondary);
  desenharTimeline(ctx, d.history, sepY + 22);

  if (d.progresso) texto(ctx, d.progresso, 10, PANEL_H - 20, FT, cfg.color_text_secondary);
  ctx.restore();
}

// Blocos com altura proporcional à confiança; o mais recente à direita. Faixa de baixo = classe verdadeira.
function desenharTimeline(ctx: CanvasRenderingContext2D, items: ItemHistorico[], topY: number) {
  const n = items.length;
  if (n === 0) {
    texto(ctx, "aguardando predições…", 10, topY + 20, FT, cfg.color_text_secondary);
    return;
  }
  const areaW = PANEL_W - 20;
  const baselineY = topY + TIMELINE_MAX_H + 14;
  const totalW = n * (TIMELINE_BLOCK_W + TIMELINE_BLOCK_GAP) - TIMELINE_BLOCK_GAP;
  const startX = 10 + Math.max(0, areaW - totalW);

  items.forEach(({ classe, confianca, verdade }, idx) => {
    const bx = startX + idx * (TIMELINE_BLOCK_W + TIMELINE_BLOCK_GAP);
    const blockH = Math.trunc(TIMELINE_MIN_H + confianca * (TIMELINE_MAX_H - TIMELINE_MIN_H));
    const by = baselineY - blockH;
    retangulo(ctx, bx, by, TIMELINE_BLOCK_W, blockH, CORES_CLASSES[classe], 2);
    if (idx === n - 1) retangulo(ctx, bx, by, TIMELINE_BLOCK_W, blockH, "rgb(255,255,255)", 2, true);

    if (n <= 8 || idx % 3 === 0) {
      const ab = ABREV[classe];
      texto(ctx, ab, bx + TIMELINE_BLOCK_W / 2 - Math.floor(larguraTexto(ctx, ab, FT) / 2), baselineY + 2, FT,
        cfg.color_text_secondary);
    }
    if (verdade !== null) retangulo(ctx, bx, baselineY + 18, TIMELINE_BLOCK_W, 5, CORES_CLASSES[verdade], 1);
  });

  if (items.some((it) => it.verdade !== null)) {
    texto(ctx, "faixa de baixo: classe verdadeira", 10, baselineY + 26, FT, cfg.color_text_secondary);
  }
}
