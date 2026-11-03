// Tradução de src/simulacao/rendering/environment.py: corredor com scroll vertical.
import { CONFIG as cfg, rgb } from "./config";

export class Ambiente {
  scrollY = 0;
  private periodo = cfg.track_dash_length + cfg.track_dash_gap;

  update(confidence: number) {
    // Velocidade de scroll sobe com a confiança: decisão clara = resposta mais ágil
    const speed = cfg.track_scroll_speed * (0.5 + confidence * 0.8);
    this.scrollY = (this.scrollY + speed) % this.periodo;
  }

  draw(ctx: CanvasRenderingContext2D) {
    const w = cfg.window_width;
    const h = cfg.window_height;
    const cx = Math.floor(w / 2);

    // Fundo geral
    ctx.fillStyle = rgb(cfg.color_background);
    ctx.fillRect(0, 0, w, h);

    // Leito do corredor
    const left = cx - Math.floor(cfg.track_lane_width / 2);
    const right = cx + Math.floor(cfg.track_lane_width / 2);
    ctx.fillStyle = rgb(cfg.color_track);
    ctx.fillRect(left, 0, cfg.track_lane_width, h);

    // Paredes laterais
    const wallW = 6;
    ctx.fillStyle = rgb(cfg.color_hud_border);
    ctx.fillRect(left - wallW, 0, wallW, h);
    ctx.fillRect(right, 0, wallW, h);

    // Linha tracejada central
    ctx.fillStyle = rgb(cfg.color_track_line);
    for (let y = -this.periodo + this.scrollY; y < h; y += this.periodo) {
      ctx.fillRect(cx - 2, Math.trunc(y), 4, cfg.track_dash_length);
    }

    // Vinheta nas bordas do corredor (efeito de profundidade)
    const vw = 35;
    for (let i = 0; i < vw; i++) {
      ctx.fillStyle = rgb(cfg.color_background, (55 * (1 - i / vw)) / 255);
      ctx.fillRect(left + i, 0, 1, h);
      ctx.fillRect(right - vw + i, 0, 1, h);
    }
  }
}
