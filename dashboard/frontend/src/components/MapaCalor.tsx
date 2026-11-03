// Mapa de calor em CSS grid (o recharts não tem): matriz de confusão, IG, correlação, extremos, ANOVA etc.
// As cores vêm das escalas do tema (sequencial ou divergente marrom ↔ neutro ↔ azul).

import { corNaEscala, textoLegivel } from "./escalas";

interface Props {
  linhas: string[];
  colunas: string[];
  valores: (number | null)[][];
  escala: string[];
  posicao: (v: number) => number; // valor → [0, 1] na escala
  texto?: (v: number) => string; // rótulo da célula (omitido = sem texto)
  dica?: (linha: string, coluna: string, v: number | null) => string;
  contorno?: (v: number) => boolean; // destaca a célula (ex.: p < 0,05)
  titulo?: string;
  legenda?: { min: string; max: string; meio?: string; titulo?: string };
  rotuloColunasVertical?: boolean;
  larguraRotulo?: number;
}

export default function MapaCalor({
  linhas,
  colunas,
  valores,
  escala,
  posicao,
  texto,
  dica,
  contorno,
  titulo,
  legenda,
  rotuloColunasVertical,
  larguraRotulo = 56,
}: Props) {
  return (
    <div className="mapa">
      {titulo && <div className="mapa-titulo">{titulo}</div>}
      <div
        className="mapa-grade"
        style={{ gridTemplateColumns: `${larguraRotulo}px repeat(${colunas.length}, minmax(0, 1fr))` }}
      >
        {linhas.map((l, i) => (
          <div key={l} className="mapa-linha" style={{ display: "contents" }}>
            <div className="mapa-rotulo-linha">{l}</div>
            {colunas.map((c, j) => {
              const v = valores[i]?.[j] ?? null;
              const cor = v === null ? "transparent" : corNaEscala(escala, posicao(v));
              return (
                <div
                  key={c}
                  className={`mapa-celula${v !== null && contorno?.(v) ? " mapa-contorno" : ""}`}
                  style={{ background: cor, color: v === null ? "inherit" : textoLegivel(cor) }}
                  title={dica ? dica(l, c, v) : `${l} · ${c}`}
                >
                  {v !== null && texto ? texto(v) : ""}
                </div>
              );
            })}
          </div>
        ))}
        <div />
        {colunas.map((c) => (
          <div key={c} className={`mapa-rotulo-coluna${rotuloColunasVertical ? " vertical" : ""}`}>
            {c}
          </div>
        ))}
      </div>
      {legenda && <LegendaEscala escala={escala} {...legenda} recuo={larguraRotulo} />}
    </div>
  );
}

/** Barra de cores com os valores mínimo, central e máximo. */
export function LegendaEscala({ escala, min, max, meio, titulo, recuo = 0 }: {
  escala: string[];
  min: string;
  max: string;
  meio?: string;
  titulo?: string;
  recuo?: number;
}) {
  return (
    <div className="mapa-legenda" style={{ marginLeft: recuo }}>
      <div className="mapa-legenda-barra" style={{ background: `linear-gradient(to right, ${escala.join(", ")})` }} />
      <div className="mapa-legenda-rotulos">
        <span>{min}</span>
        {meio && <span>{meio}</span>}
        <span>{max}</span>
      </div>
      {titulo && <div className="mapa-legenda-titulo">{titulo}</div>}
    </div>
  );
}
