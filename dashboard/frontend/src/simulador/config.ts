// Tradução de src/simulacao/config.py (SimulationConfig): mesmos valores do simulador Pygame.
// A exibição é a mesma do Pygame, inclusive a paleta escura, independente do tema do dashboard.

export type RGB = readonly [number, number, number];

export const CONFIG = {
  // Janela
  window_width: 1280,
  window_height: 720,

  // Painel EEG (esquerda)
  eeg_panel_x: 10,
  eeg_panel_width: 260,

  // Mock: tempo de cada classe no ciclo
  mock_class_hold_ms: 1500,

  // Física do veículo
  vehicle_lateral_speed: 2.5, // px/frame ao desviar
  vehicle_turn_rate: 1.8, // graus/frame ao virar
  vehicle_max_angle: 30.0, // ângulo máximo de esterçamento

  // Geometria do corredor
  track_lane_width: 320,
  track_dash_length: 40,
  track_dash_gap: 30,
  track_scroll_speed: 2.0, // px/frame de scroll do fundo

  // Paleta (neutra, como nas figuras do trabalho). Classes: azul, magenta e ocre.
  color_background: [15, 20, 35] as RGB,
  color_track: [30, 38, 58] as RGB,
  color_track_line: [60, 70, 95] as RGB,
  color_hud_bg: [10, 15, 28] as RGB,
  color_hud_border: [70, 90, 130] as RGB,
  color_class_left: [91, 155, 232] as RGB, // azul — Esquerda
  color_class_right: [208, 122, 180] as RGB, // magenta — Direita
  color_class_neutral: [201, 150, 79] as RGB, // ocre — Neutro
  color_text_primary: [220, 225, 240] as RGB,
  color_text_secondary: [140, 155, 180] as RGB,
};

export const CORES_CLASSES: readonly RGB[] = [CONFIG.color_class_left, CONFIG.color_class_right, CONFIG.color_class_neutral];
export const NOMES_CLASSES = ["Esquerda", "Direita", "Neutro"] as const;

export const rgb = (c: RGB, alfa = 1) => (alfa >= 1 ? `rgb(${c[0]},${c[1]},${c[2]})` : `rgba(${c[0]},${c[1]},${c[2]},${alfa})`);

// Fonte como no pygame.font.SysFont("Segoe UI", tamanho)
export const fonte = (tamanho: number, negrito = false) =>
  `${negrito ? "bold " : ""}${tamanho}px "Segoe UI", system-ui, sans-serif`;

// Texto com o canto superior esquerdo em (x, y), como o blit de uma surface de texto no Pygame
export function texto(ctx: CanvasRenderingContext2D, s: string, x: number, y: number, f: string, cor: RGB | string) {
  ctx.font = f;
  ctx.fillStyle = typeof cor === "string" ? cor : rgb(cor);
  ctx.textBaseline = "top";
  ctx.textAlign = "left";
  ctx.fillText(s, x, y);
}

export function larguraTexto(ctx: CanvasRenderingContext2D, s: string, f: string): number {
  ctx.font = f;
  return ctx.measureText(s).width;
}

// pygame.draw.rect com border_radius (preenchido ou só contorno de 1 px)
export function retangulo(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  cor: RGB | string,
  raio = 0,
  contorno = false,
) {
  const c = typeof cor === "string" ? cor : rgb(cor);
  ctx.beginPath();
  if (contorno) {
    ctx.roundRect(x + 0.5, y + 0.5, w - 1, h - 1, raio);
    ctx.strokeStyle = c;
    ctx.lineWidth = 1;
    ctx.stroke();
  } else {
    ctx.roundRect(x, y, w, h, raio);
    ctx.fillStyle = c;
    ctx.fill();
  }
}

export function linha(ctx: CanvasRenderingContext2D, x0: number, y0: number, x1: number, y1: number, cor: RGB | string) {
  ctx.beginPath();
  ctx.moveTo(x0 + 0.5, y0 + 0.5);
  ctx.lineTo(x1 + 0.5, y1 + 0.5);
  ctx.strokeStyle = typeof cor === "string" ? cor : rgb(cor);
  ctx.lineWidth = 1;
  ctx.stroke();
}
