// Tradução do loop de src/simulacao/pygame_simulator.py (run_simulation) para o navegador.
// requestAnimationFrame limitado ao FPS escolhido; física por frame, como no Pygame (então o FPS
// também muda a velocidade do carro, como lá). Uma predição a cada intervaloMs.
import { Ambiente } from "./ambiente";
import { CONFIG as cfg, fonte, larguraTexto, rgb, texto } from "./config";
import type { Controlador, Predicao } from "./controladores";
import { desenharHUD, TIMELINE_MAX, type ItemHistorico } from "./hud";
import { desenharPainelEeg } from "./painelEeg";
import { Veiculo } from "./veiculo";

export interface OpcoesSimulacao {
  controlador: Controlador;
  useMock: boolean;
  modelName: string; // nome exibido antes da primeira predição
  fps: number;
  intervaloMs: number; // 1000 / velocidade no modo real; 1000 no mock, como no Pygame
}

export interface EstadoControles {
  pausado: boolean;
  congelado: boolean;
}

const LEGENDA = "[SPACE] Pausar   [F] Congelar Dados   [R] Resetar";

export class Simulacao {
  private ctx: CanvasRenderingContext2D;
  private o: OpcoesSimulacao;
  private aoMudar: (e: EstadoControles) => void;
  private env = new Ambiente();
  private veiculo = new Veiculo();
  private historico: ItemHistorico[] = [];
  private atual: Predicao;
  private inicio = performance.now();
  private ultimaPredMs = 0;
  private ultimoFrame = 0;
  private duracoes: number[] = []; // últimos frames, para o FPS medido (como clock.get_fps)
  private raf = 0;
  private suspenso = false;
  private pausado = false;
  private congelado = false;
  private overlayDesenhado = false;

  constructor(canvas: HTMLCanvasElement, opcoes: OpcoesSimulacao, aoMudar: (e: EstadoControles) => void) {
    this.ctx = canvas.getContext("2d")!;
    // O canvas pode ter mais pixels que 1280×720 (texto nítido em telas de alta densidade)
    const escala = canvas.width / cfg.window_width;
    this.ctx.setTransform(escala, 0, 0, escala, 0, 0);
    this.o = opcoes;
    this.aoMudar = aoMudar;
    this.atual = {
      class_id: 2,
      probabilities: [0.05, 0.05, 0.9],
      model_name: opcoes.useMock ? "Mock" : opcoes.modelName,
      true_class_id: null,
      acertos: 0,
      total: 0,
      progresso: "",
      registro: null,
      atencao: null,
    };
  }

  iniciar() {
    const passo = (agora: number) => {
      this.raf = requestAnimationFrame(passo);
      const alvo = 1000 / this.o.fps;
      if (this.ultimoFrame && agora - this.ultimoFrame < alvo - 0.5) return;
      if (this.ultimoFrame) {
        this.duracoes.push(agora - this.ultimoFrame);
        if (this.duracoes.length > 10) this.duracoes.shift();
      }
      this.ultimoFrame = agora;
      this.frame(agora - this.inicio);
    };
    this.raf = requestAnimationFrame(passo);
  }

  parar() {
    cancelAnimationFrame(this.raf);
    this.raf = 0;
    this.suspenso = false;
  }

  /** Fora da tela: para de desenhar sem mexer na pausa (o dashboard não gasta CPU à toa). */
  suspender() {
    if (!this.raf) return;
    cancelAnimationFrame(this.raf);
    this.raf = 0;
    this.suspenso = true;
  }

  retomar() {
    if (!this.suspenso) return;
    this.suspenso = false;
    this.ultimoFrame = 0;
    this.duracoes = [];
    this.iniciar();
  }

  alternarPausa() {
    this.pausado = !this.pausado;
    this.overlayDesenhado = false;
    this.notificar();
  }

  alternarCongelar() {
    this.congelado = !this.congelado;
    this.notificar();
  }

  resetar() {
    this.veiculo.reset();
    this.env.scrollY = 0;
    this.historico = [];
    this.o.controlador.reset();
    this.ultimaPredMs = performance.now() - this.inicio;
  }

  private notificar() {
    this.aoMudar({ pausado: this.pausado, congelado: this.congelado });
  }

  private frame(ticks: number) {
    const ctx = this.ctx;
    if (this.pausado) {
      // Overlay de pausa sobre o último frame (desenhado uma vez)
      if (!this.overlayDesenhado) {
        desenharPausa(ctx);
        this.overlayDesenhado = true;
      }
      return;
    }

    // Gate de predição: dispara a cada intervaloMs (inativo quando congelado)
    if (!this.congelado && ticks - this.ultimaPredMs >= this.o.intervaloMs) {
      this.atual = this.o.controlador.update(ticks);
      this.ultimaPredMs = ticks;
      const p = this.atual;
      this.historico.push({ classe: p.class_id, confianca: p.probabilities[p.class_id], verdade: p.true_class_id });
      if (this.historico.length > TIMELINE_MAX) this.historico.shift();
    }
    const p = this.atual;
    const confianca = p.probabilities[p.class_id];

    // Física
    this.veiculo.applyCommand(p.class_id);
    this.env.update(confianca);

    // Renderização
    this.env.draw(ctx);
    this.veiculo.draw(ctx);
    desenharPainelEeg(ctx, p.registro, p.atencao);
    const media = this.duracoes.length ? this.duracoes.reduce((a, b) => a + b, 0) / this.duracoes.length : 0;
    desenharHUD(ctx, {
      predicted_class_id: p.class_id,
      probabilities: p.probabilities,
      model_display_name: p.model_name,
      fps: media ? 1000 / media : 0,
      use_mock: this.o.useMock,
      history: this.historico,
      data_frozen: this.congelado,
      true_class_id: p.true_class_id,
      acertos: p.acertos,
      total: p.total,
      progresso: p.progresso,
    });
    const f = fonte(12);
    const w = larguraTexto(ctx, LEGENDA, f);
    texto(ctx, LEGENDA, Math.floor(cfg.window_width / 2 - w / 2), cfg.window_height - 16 - 6, f,
      cfg.color_text_secondary);
  }
}

function desenharPausa(ctx: CanvasRenderingContext2D) {
  ctx.fillStyle = rgb([0, 0, 0], 130 / 255);
  ctx.fillRect(0, 0, cfg.window_width, cfg.window_height);
  const f = fonte(40, true);
  const s = "PAUSADO  —  SPACE para continuar";
  const w = larguraTexto(ctx, s, f);
  texto(ctx, s, cfg.window_width / 2 - w / 2, cfg.window_height / 2 - 26, f, [220, 225, 240]);
}
