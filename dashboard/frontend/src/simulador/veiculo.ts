// Tradução de src/simulacao/entities/vehicle.py: carro visto de cima que segue a classe prevista.
// O corredor rola na vertical, então só x (lateral) e o ângulo mudam. Física por frame, como no Pygame.
import { CONFIG as cfg } from "./config";

const BODY_W = 36;
const BODY_H = 52;
const BOUNCE_MARGIN = 40; // px a partir da borda onde começa a repulsão suave

export class Veiculo {
  x = cfg.window_width / 2;
  y = cfg.window_height * 0.65;
  angle = 0; // graus; 0 = reto, negativo = esquerda, positivo = direita
  private sprite: HTMLCanvasElement | null = null;

  /** 0 = Esquerda, 1 = Direita, 2 = Neutro. */
  applyCommand(classId: number) {
    const half = Math.floor(cfg.track_lane_width / 2);
    const cx = Math.floor(cfg.window_width / 2);
    const margin = Math.floor(BODY_W / 2) + 10;
    const leftBound = cx - half + margin;
    const rightBound = cx + half - margin;

    if (classId === 0) {
      this.x -= cfg.vehicle_lateral_speed;
      this.angle = Math.max(this.angle - cfg.vehicle_turn_rate, -cfg.vehicle_max_angle);
    } else if (classId === 1) {
      this.x += cfg.vehicle_lateral_speed;
      this.angle = Math.min(this.angle + cfg.vehicle_turn_rate, cfg.vehicle_max_angle);
    } else if (Math.abs(this.angle) <= cfg.vehicle_turn_rate * 0.5) {
      this.angle = 0;
    } else if (this.angle > 0) {
      this.angle -= cfg.vehicle_turn_rate * 0.5;
    } else {
      this.angle += cfg.vehicle_turn_rate * 0.5;
    }

    // Bounce suave: repulsão proporcional à penetração na zona de borda
    const leftSoft = leftBound + BOUNCE_MARGIN;
    const rightSoft = rightBound - BOUNCE_MARGIN;
    if (this.x < leftSoft) {
      this.x += ((leftSoft - this.x) / BOUNCE_MARGIN) * cfg.vehicle_lateral_speed * 0.9;
    } else if (this.x > rightSoft) {
      this.x -= ((this.x - rightSoft) / BOUNCE_MARGIN) * cfg.vehicle_lateral_speed * 0.9;
    }
    // Limite físico absoluto
    this.x = Math.max(leftBound, Math.min(rightBound, this.x));
  }

  reset() {
    this.x = cfg.window_width / 2;
    this.angle = 0;
  }

  draw(ctx: CanvasRenderingContext2D) {
    const s = this.sprite ?? (this.sprite = construirSprite());
    ctx.save();
    ctx.translate(Math.trunc(this.x), Math.trunc(this.y));
    ctx.rotate((this.angle * Math.PI) / 180); // pygame.transform.rotate(surf, -angle)
    ctx.drawImage(s, -s.width / 2, -s.height / 2);
    ctx.restore();
  }
}

function construirSprite(): HTMLCanvasElement {
  const pad = 20;
  const w = BODY_W + pad;
  const h = BODY_H + pad;
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d")!;
  const cx = Math.floor(w / 2);
  const cy = Math.floor(h / 2);
  const bx = cx - BODY_W / 2;
  const by = cy - BODY_H / 2;

  const rect = (x: number, y: number, rw: number, rh: number, cor: string, raio: number, borda = 0) => {
    ctx.beginPath();
    if (borda) {
      ctx.roundRect(x + borda / 2, y + borda / 2, rw - borda, rh - borda, raio);
      ctx.strokeStyle = cor;
      ctx.lineWidth = borda;
      ctx.stroke();
    } else {
      ctx.roundRect(x, y, rw, rh, raio);
      ctx.fillStyle = cor;
      ctx.fill();
    }
  };

  // Corpo principal
  rect(bx, by, BODY_W, BODY_H, "rgb(200,210,230)", 6);
  rect(bx, by, BODY_W, BODY_H, "rgb(100,120,160)", 6, 2);

  // Assento/cockpit
  ctx.beginPath();
  ctx.ellipse(cx, cy - 10 + 7, 8, 7, 0, 0, Math.PI * 2);
  ctx.fillStyle = "rgb(80,100,140)";
  ctx.fill();

  // Rodas (4 cantos)
  for (const [dx, dy] of [
    [-BODY_W / 2 - 3, -BODY_H / 2 + 4],
    [BODY_W / 2 - 4, -BODY_H / 2 + 4],
    [-BODY_W / 2 - 3, BODY_H / 2 - 18],
    [BODY_W / 2 - 4, BODY_H / 2 - 18],
  ]) {
    rect(cx + dx, cy + dy, 7, 14, "rgb(50,60,80)", 2);
  }

  // Triângulo de direção (frente do veículo)
  ctx.beginPath();
  ctx.moveTo(cx, cy - BODY_H / 2 - 6);
  ctx.lineTo(cx - 5, cy - BODY_H / 2 + 2);
  ctx.lineTo(cx + 5, cy - BODY_H / 2 + 2);
  ctx.closePath();
  ctx.fillStyle = "rgb(180,200,255)";
  ctx.fill();
  return c;
}
