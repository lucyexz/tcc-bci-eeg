// Tradução de src/simulacao/controllers/: o controlador real (predições do modelo) e o mock.
import type { DadosSimulador } from "../api/client";

export interface Predicao {
  class_id: number;
  probabilities: readonly number[]; // (P(Esquerda), P(Direita), P(Neutro))
  model_name: string;
  // Só no modo real:
  true_class_id: number | null;
  acertos: number;
  total: number;
  progresso: string;
  registro: readonly number[] | null; // as 25 features do registro mais recente (log padronizado)
  atencao: readonly number[] | null; // atenção do [CLS] nos tokens de tempo (Transformers)
}

export interface Controlador {
  update(ticksMs: number): Predicao;
  reset(): void;
}

const argmax = (p: readonly number[]) => p.reduce((m, v, i) => (v > p[m] ? i : m), 0);

/** model_controller.py: avança uma janela de teste por atualização; zera o acerto ao dar a volta. */
export class ControladorModelo implements Controlador {
  private index = 0;
  private acertos = 0;
  private total = 0;
  private d: DadosSimulador;

  constructor(d: DadosSimulador) {
    this.d = d;
  }

  update(): Predicao {
    const d = this.d;
    const i = this.index;
    const proba = d.proba[i];
    const classId = argmax(proba);
    const verdade = d.y[i];
    this.acertos += Number(classId === verdade);
    this.total += 1;
    const resultado: Predicao = {
      class_id: classId,
      probabilities: proba,
      model_name: `${d.nome} · ${d.paradigma}`,
      true_class_id: verdade,
      acertos: this.acertos,
      total: this.total,
      progresso: `${d.participante} · bloco ${d.blocos[i] + 1} · janela ${i + 1}/${d.n}`,
      registro: d.registros[i],
      atencao: d.atencao ? d.atencao[i] : null,
    };
    this.index = (i + 1) % d.n;
    if (this.index === 0) this.acertos = this.total = 0; // recomeça o trecho
    return resultado;
  }

  reset() {
    this.index = 0;
    this.acertos = this.total = 0;
  }
}

// PRNG simples com semente (mulberry32): o mock não precisa reproduzir os números do numpy
function mulberry32(semente: number) {
  let a = semente >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** mock_controller.py: cicla Neutro → Esquerda → Neutro → Direita, trocando a cada holdMs. */
export class ControladorMock implements Controlador {
  static CYCLE_ORDER = [2, 0, 2, 1, 2, 0, 2, 1];
  private rng = mulberry32(42);
  private cycleIndex = 0;
  private lastSwitchMs = 0;
  private current: Predicao;
  private holdMs: number;

  constructor(holdMs = 1500) {
    this.holdMs = holdMs;
    this.current = this.build(ControladorMock.CYCLE_ORDER[0]);
  }

  update(ticksMs: number): Predicao {
    if (ticksMs - this.lastSwitchMs >= this.holdMs) {
      this.cycleIndex = (this.cycleIndex + 1) % ControladorMock.CYCLE_ORDER.length;
      this.current = this.build(ControladorMock.CYCLE_ORDER[this.cycleIndex]);
      this.lastSwitchMs = ticksMs;
    }
    return this.current;
  }

  reset() {
    this.cycleIndex = 0;
    this.lastSwitchMs = 0;
    this.current = this.build(ControladorMock.CYCLE_ORDER[0]);
  }

  private build(classId: number): Predicao {
    const confidence = 0.72 + this.rng() * (0.93 - 0.72);
    const residual = 1 - confidence;
    const split = this.rng() * residual;
    const others = [0, 1, 2].filter((i) => i !== classId);
    const probs = [0, 0, 0];
    probs[classId] = confidence;
    probs[others[0]] = split;
    probs[others[1]] = residual - split;
    return {
      class_id: classId,
      probabilities: probs,
      model_name: "Mock",
      true_class_id: null,
      acertos: 0,
      total: 0,
      progresso: "",
      registro: null,
      atencao: null,
    };
  }
}
