// Estilo comum dos gráficos das figuras (eixos, dica, marcadores), nas cores do tema.
import type { CSSProperties } from "react";
import { num } from "../api/client";
import { useTema, type CoresGrafico } from "../tema";

export const ACASO = 100 / 3;

export function useEstilo() {
  const { cores } = useTema();
  return {
    cores,
    eixo: { tick: { fill: cores.eixo, fontSize: 12 }, stroke: cores.grade },
    eixoPequeno: { tick: { fill: cores.eixo, fontSize: 10 }, stroke: cores.grade },
    dica: {
      contentStyle: { background: cores.dicaFundo, border: `1px solid ${cores.dicaBorda}`, borderRadius: 6 },
      labelStyle: { color: cores.texto },
      itemStyle: { color: cores.texto },
    },
    caixaDica: {
      background: cores.dicaFundo,
      border: `1px solid ${cores.dicaBorda}`,
      borderRadius: 6,
      padding: "6px 10px",
      color: cores.texto,
      fontSize: 13,
    } as CSSProperties,
  };
}

/** Fração (0–1) → "42,3%". */
export const pctFracao = (v: number, casas = 1) => `${num(v * 100, casas)}%`;
/** Já em % → "42,3%". */
export const pct = (v: number, casas = 1) => `${num(v, casas)}%`;

export const corParadigma = (cores: CoresGrafico, paradigma: string) =>
  paradigma.startsWith("Generalista") ? cores.generalista : cores.especialista;

// Generalista ■, especialista ●, como nas figuras da monografia
export const formaParadigma = (paradigma: string): "square" | "circle" =>
  paradigma.startsWith("Generalista") ? "square" : "circle";

interface PontoForma {
  cx?: number;
  cy?: number;
}

/** Marcador para o `shape` de um <Scatter>: círculo, quadrado ou triângulo. */
export function marcador(forma: "circle" | "square" | "triangle", r: number, cor: string,
  opcoes: { opacidade?: number; vazado?: boolean; borda?: string } = {}) {
  const { opacidade = 1, vazado = false, borda } = opcoes;
  const MarcadorForma = (props: unknown) => {
    const { cx = 0, cy = 0 } = props as PontoForma;
    const comum = {
      fill: vazado ? "none" : cor,
      fillOpacity: opacidade,
      stroke: vazado ? cor : borda ?? "none",
      strokeWidth: vazado ? 1.6 : borda ? 1.2 : 0,
    };
    if (forma === "square") return <rect x={cx - r} y={cy - r} width={2 * r} height={2 * r} {...comum} />;
    if (forma === "triangle")
      return <polygon points={`${cx},${cy - r * 1.2} ${cx - r * 1.1},${cy + r * 0.8} ${cx + r * 1.1},${cy + r * 0.8}`} {...comum} />;
    return <circle cx={cx} cy={cy} r={r} {...comum} />;
  };
  return MarcadorForma;
}

export const FORMAS_CLASSES = ["circle", "square", "triangle"] as const;
