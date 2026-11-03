// Escalas de cor dos mapas de calor (valor → posição → cor).

function hexParaRgb(c: string): [number, number, number] {
  return [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16)) as [number, number, number];
}

/** Cor da escala na posição t ∈ [0, 1] (interpolação linear entre as paradas). */
export function corNaEscala(escala: string[], t: number): string {
  const x = Math.max(0, Math.min(1, t)) * (escala.length - 1);
  const i = Math.min(Math.floor(x), escala.length - 2);
  const a = hexParaRgb(escala[i]);
  const b = hexParaRgb(escala[i + 1]);
  const f = x - i;
  const c = a.map((v, k) => Math.round(v + f * (b[k] - v)));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

/** Posição de v numa escala divergente com centro em `centro` (como o TwoSlopeNorm do matplotlib). */
export const duasInclinacoes = (min: number, centro: number, max: number) => (v: number) =>
  v < centro ? 0.5 * ((v - min) / (centro - min)) : 0.5 + 0.5 * ((v - centro) / (max - centro));

export function textoLegivel(cor: string): string {
  const [r, g, b] = cor.match(/\d+/g)!.map(Number);
  return 0.299 * r + 0.587 * g + 0.114 * b > 150 ? "#0b0b0b" : "#ffffff";
}
