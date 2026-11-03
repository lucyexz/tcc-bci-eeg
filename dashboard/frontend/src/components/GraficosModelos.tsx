// Gráficos interativos das figuras dos modelos e do comparativo (os dados vêm de resultados/<etapa>/graficos/).
// Mesmo conteúdo dos PNGs da monografia; números como fração (0–1) nos .json.
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { num } from "../api/client";
import {
  ACASO,
  corParadigma,
  FORMAS_CLASSES,
  formaParadigma,
  marcador,
  pct,
  pctFracao,
  useEstilo,
} from "./estiloGrafico";
import { duasInclinacoes } from "./escalas";
import LegendaItens from "./LegendaItens";
import MapaCalor, { LegendaEscala } from "./MapaCalor";

// ── Tipos dos .json ─────────────────────────────────────────────────────────

interface SerieResultado {
  paradigma: string;
  valores: Record<string, number>;
  media: number;
}
export interface DadosResultado {
  protocolos: { protocolo: string; linhas: { rotulo: string; series: SerieResultado[] }[] }[];
}
export interface DadosPermutacao {
  paradigmas: { paradigma: string; linhas: { rotulo: string; verdadeiro: number; permutado: number }[] }[];
}
interface Curva {
  epocas: number[];
  valores: number[];
}
export interface LinhaCurvas {
  rotulo: string;
  treino: Curva;
  validacao: Curva;
  melhor_epoca: number;
}
export interface DadosCurvas {
  paradigmas: { paradigma: string; linhas: LinhaCurvas[] }[];
}
export interface SerieCalibracao {
  rotulo: string;
  paradigma: string;
  confianca: number[];
  acuracia: number[];
  contagens: number[];
  ece: number;
  confianca_media: number;
  acuracia_geral: number;
}
export interface DadosCalibracao {
  series: SerieCalibracao[];
}
export interface DadosMatriz {
  classes: string[];
  paradigmas: { paradigma: string; matriz: number[][] }[];
}
export interface DadosGradesIG {
  eletrodos: string[];
  bandas: string[];
  classes: string[];
  vmax: number;
  paradigmas: { paradigma: string; grades: number[][][] }[];
}
export interface DadosTempo {
  classes: string[];
  paradigmas: { paradigma: string; T: number; segundos: number[]; fracoes: number[][] }[];
}
export interface DadosAtencao {
  classes: string[];
  paradigmas: { paradigma: string; n_tokens: number; segundos: number[]; atencao: number[][] }[];
}
export interface DadosPorParticipante {
  participantes: string[];
  modelos: string[];
  cenarios: { protocolo: string; paradigma: string; matriz: (number | null)[][] }[];
}

interface PontoDica {
  rotulo?: string;
  texto: string;
}

function DicaTexto({ active, payload }: { active?: boolean; payload?: { payload: PontoDica }[] }) {
  const { caixaDica } = useEstilo();
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <div style={caixaDica}>
      {p.rotulo && <div style={{ fontWeight: 600 }}>{p.rotulo}</div>}
      <div>{p.texto}</div>
    </div>
  );
}

// ── 01 resultado: um ponto por participante + a média, por protocolo ───────

export function GraficoPontosResultado({ dados }: { dados: DadosResultado }) {
  const { cores, eixo } = useEstilo();
  const paradigmas = dados.protocolos[0]?.linhas[0]?.series.map((s) => s.paradigma) ?? [];
  return (
    <>
      <div className="graficos-duplos">
        {dados.protocolos.map(({ protocolo, linhas }) => {
          const n = linhas.length;
          const rotulos = linhas.map((l) => l.rotulo);
          return (
            <div key={protocolo} className="grafico-metrica">
              <h4 className="grafico-subtitulo">{protocolo}</h4>
              <ResponsiveContainer width="100%" height={70 + Math.max(2, n) * 46}>
                <ScatterChart margin={{ top: 22, right: 16, left: 0, bottom: 4 }}>
                  <CartesianGrid stroke={cores.grade} strokeDasharray="3 3" horizontal={false} />
                  <XAxis type="number" dataKey="x" domain={[0, 80]} ticks={[0, 10, 20, 30, 40, 50, 60, 70, 80]}
                    tickFormatter={(v) => `${v}%`} {...eixo} />
                  <YAxis type="number" dataKey="y" domain={[-0.5, n - 0.5]} ticks={rotulos.map((_, i) => i)}
                    tickFormatter={(i) => rotulos[i] ?? ""} reversed width={n > 1 ? 120 : 70} {...eixo} />
                  <Tooltip content={<DicaTexto />} cursor={false} />
                  <ReferenceLine x={ACASO} stroke={cores.acaso} strokeDasharray="4 4"
                    label={{ value: "acaso", fill: cores.acaso, fontSize: 11, position: "top" }} />
                  {paradigmas.map((par, k) => {
                    const dy = paradigmas.length > 1 ? (k === 0 ? -0.17 : 0.17) : 0;
                    const cor = corParadigma(cores, par);
                    const pontos = linhas.flatMap((l, i) => {
                      const s = l.series.find((x) => x.paradigma === par);
                      return s ? Object.entries(s.valores).map(([u, v]) => ({
                        x: v * 100, y: i + dy, rotulo: `${l.rotulo} · ${par}`, texto: `${u}: ${pctFracao(v)}`,
                      })) : [];
                    });
                    const medias = linhas.flatMap((l, i) => {
                      const s = l.series.find((x) => x.paradigma === par);
                      return s ? [{ x: s.media * 100, y: i + dy, rotulo: `${l.rotulo} · ${par}`,
                        texto: `média: ${pctFracao(s.media)}` }] : [];
                    });
                    return [
                      <Scatter key={`${par}-p`} data={pontos} isAnimationActive={false}
                        shape={marcador(formaParadigma(par), 4, cor, { opacidade: 0.55 })} />,
                      <Scatter key={`${par}-m`} data={medias} isAnimationActive={false}
                        shape={marcador(formaParadigma(par), 7, cor, { borda: cores.dicaFundo })} />,
                    ];
                  })}
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          );
        })}
      </div>
      <LegendaItens itens={[
        ...paradigmas.map((p) => ({ rotulo: `${p} (média)`, cor: corParadigma(cores, p), forma: formaParadigma(p) })),
        { rotulo: "pontos claros: cada participante", cor: cores.eixo, forma: "circle" },
      ]} />
    </>
  );
}

// ── Permutação: rótulos permutados (○) → verdadeiros (● cor do paradigma) ──

export function GraficoHalteres({ dados }: { dados: DadosPermutacao }) {
  const { cores, eixo } = useEstilo();
  return (
    <>
      <div className="graficos-duplos">
        {dados.paradigmas.map(({ paradigma, linhas }) => {
          const n = linhas.length;
          const rotulos = linhas.map((l) => l.rotulo);
          const cor = corParadigma(cores, paradigma);
          const ponto = (l: (typeof linhas)[number], i: number, qual: "verdadeiro" | "permutado") => ({
            x: l[qual] * 100, y: i, rotulo: l.rotulo,
            texto: `${qual === "verdadeiro" ? "rótulos verdadeiros" : "rótulos permutados"}: ${pctFracao(l[qual])}`,
          });
          return (
            <div key={paradigma} className="grafico-metrica">
              <h4 className="grafico-subtitulo">{paradigma}</h4>
              <ResponsiveContainer width="100%" height={70 + Math.max(2, n) * 40}>
                <ScatterChart margin={{ top: 22, right: 16, left: 0, bottom: 4 }}>
                  <CartesianGrid stroke={cores.grade} strokeDasharray="3 3" horizontal={false} />
                  <XAxis type="number" dataKey="x" domain={[15, 60]} ticks={[20, 30, 40, 50, 60]}
                    tickFormatter={(v) => `${v}%`} {...eixo} />
                  <YAxis type="number" dataKey="y" domain={[-0.5, n - 0.5]} ticks={rotulos.map((_, i) => i)}
                    tickFormatter={(i) => rotulos[i] ?? ""} reversed width={n > 1 ? 120 : 70} {...eixo} />
                  <Tooltip content={<DicaTexto />} cursor={false} />
                  <ReferenceLine x={ACASO} stroke={cores.acaso} strokeDasharray="4 4"
                    label={{ value: "acaso", fill: cores.acaso, fontSize: 11, position: "top" }} />
                  {linhas.map((l, i) => (
                    <ReferenceLine key={l.rotulo} stroke={cores.contexto} strokeWidth={3}
                      segment={[{ x: l.permutado * 100, y: i }, { x: l.verdadeiro * 100, y: i }]} />
                  ))}
                  <Scatter data={linhas.map((l, i) => ponto(l, i, "permutado"))} isAnimationActive={false}
                    shape={marcador("circle", 6, cores.treino, { vazado: true })} />
                  <Scatter data={linhas.map((l, i) => ponto(l, i, "verdadeiro"))} isAnimationActive={false}
                    shape={marcador(formaParadigma(paradigma), 6.5, cor)} />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          );
        })}
      </div>
      <LegendaItens itens={[
        { rotulo: "rótulos permutados entre blocos", cor: cores.treino, forma: "vazado" },
        ...dados.paradigmas.map((p) => ({ rotulo: `rótulos verdadeiros (${p.paradigma.toLowerCase()})`,
          cor: corParadigma(cores, p.paradigma), forma: formaParadigma(p.paradigma) })),
      ]} />
    </>
  );
}

// ── Curvas de treino: mediana dos treinos por época ─────────────────────────

export function CurvaTreino({ linha, cor, altura = 220, titulo, compacto }: {
  linha: LinhaCurvas;
  cor: string;
  altura?: number;
  titulo?: string;
  compacto?: boolean;
}) {
  const { cores, eixo, eixoPequeno, dica } = useEstilo();
  const porEpoca = new Map<number, { epoca: number; treino?: number; validacao?: number }>();
  linha.treino.epocas.forEach((e, i) => porEpoca.set(e, { ...porEpoca.get(e), epoca: e, treino: linha.treino.valores[i] * 100 }));
  linha.validacao.epocas.forEach((e, i) =>
    porEpoca.set(e, { ...porEpoca.get(e), epoca: e, validacao: linha.validacao.valores[i] * 100 }));
  const dados = [...porEpoca.values()].sort((a, b) => a.epoca - b.epoca);
  const ex = compacto ? eixoPequeno : eixo;
  return (
    <div className="grafico-metrica">
      {titulo && <h4 className="grafico-subtitulo">{titulo}</h4>}
      <ResponsiveContainer width="100%" height={altura}>
        <LineChart data={dados} margin={{ top: 8, right: 12, left: compacto ? -18 : 0, bottom: 0 }}>
          <CartesianGrid stroke={cores.grade} strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="epoca" type="number" domain={[1, "dataMax"]} allowDecimals={false} {...ex} />
          <YAxis domain={[25, 100]} ticks={[25, 50, 75, 100]} tickFormatter={(v) => `${v}%`} {...ex} />
          <Tooltip {...dica} labelFormatter={(e) => `Época ${e}`} formatter={(v) => pct(Number(v))} />
          <ReferenceLine y={ACASO} stroke={cores.acaso} strokeDasharray="4 4" />
          <ReferenceLine x={linha.melhor_epoca} stroke={cores.eixo} strokeDasharray="2 2"
            label={compacto ? undefined : { value: `melhor época (mediana ${linha.melhor_epoca})`, fill: cores.eixo,
              fontSize: 11, position: "insideTopLeft" }} />
          <Line dataKey="treino" name="treino" stroke={cores.treino} dot={{ r: 2 }} strokeWidth={2}
            isAnimationActive={false} connectNulls />
          <Line dataKey="validacao" name="validação" stroke={cor} dot={{ r: 2 }} strokeWidth={2}
            isAnimationActive={false} connectNulls />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

/** Página do modelo: um gráfico por paradigma. */
export function GraficoCurvasModelo({ dados }: { dados: DadosCurvas }) {
  const { cores } = useEstilo();
  return (
    <>
      <div className="graficos-duplos">
        {dados.paradigmas.map(({ paradigma, linhas }) => (
          <CurvaTreino key={paradigma} linha={linhas[0]} cor={corParadigma(cores, paradigma)} titulo={paradigma} />
        ))}
      </div>
      <LegendaItens itens={[
        { rotulo: "treino", cor: cores.treino, forma: "linha" },
        ...dados.paradigmas.map((p) => ({ rotulo: `validação (${p.paradigma.toLowerCase()})`,
          cor: corParadigma(cores, p.paradigma), forma: "linha" })),
        { rotulo: "acaso", cor: cores.acaso, tracejado: true },
      ]} />
    </>
  );
}

/** Comparativo: os cinco modelos de um paradigma, em pequenos múltiplos. */
export function GraficoCurvasTodos({ dados, paradigma }: { dados: DadosCurvas; paradigma: string }) {
  const { cores } = useEstilo();
  const p = dados.paradigmas.find((x) => x.paradigma === paradigma) ?? dados.paradigmas[0];
  const cor = corParadigma(cores, p.paradigma);
  return (
    <>
      <div className="pequenos-multiplos">
        {p.linhas.map((l) => (
          <CurvaTreino key={l.rotulo} linha={l} cor={cor} titulo={`${l.rotulo} · melhor época ${l.melhor_epoca}`}
            altura={170} compacto />
        ))}
      </div>
      <LegendaItens itens={[
        { rotulo: "treino", cor: cores.treino, forma: "linha" },
        { rotulo: `validação (${p.paradigma.toLowerCase()})`, cor, forma: "linha" },
        { rotulo: "melhor época (mediana)", cor: cores.eixo, tracejado: true },
      ]} />
    </>
  );
}

// ── Calibração: confiança prevista × acurácia observada (diagonal = perfeita) ──

export function GraficoCalibracao({ dados, cor }: { dados: DadosCalibracao; cor: (s: SerieCalibracao, i: number) => string }) {
  const { cores, eixo, dica } = useEstilo();
  return (
    <ResponsiveContainer width="100%" height={340}>
      <ScatterChart margin={{ top: 8, right: 20, left: 0, bottom: 4 }}>
        <CartesianGrid stroke={cores.grade} strokeDasharray="3 3" />
        <XAxis type="number" dataKey="x" name="Confiança prevista" domain={[30, 100]} ticks={[40, 60, 80, 100]}
          tickFormatter={(v) => `${v}%`} {...eixo}
          label={{ value: "Confiança prevista", position: "insideBottom", offset: -2, fill: cores.eixo, fontSize: 12 }} />
        <YAxis type="number" dataKey="y" name="Acurácia observada" domain={[20, 100]} ticks={[20, 40, 60, 80, 100]}
          tickFormatter={(v) => `${v}%`} {...eixo} />
        <Tooltip {...dica} formatter={(v) => pct(Number(v))} cursor={{ strokeDasharray: "3 3" }} />
        <Legend itemSorter={() => 0} wrapperStyle={{ color: cores.texto, paddingTop: 8 }} />
        <ReferenceLine segment={[{ x: ACASO, y: ACASO }, { x: 100, y: 100 }]} stroke={cores.contexto} strokeWidth={1.5}
          ifOverflow="extendDomain" />
        {dados.series.map((s, i) => (
          <Scatter key={s.rotulo} name={`${s.rotulo} · ECE ${num(s.ece, 2)}`} isAnimationActive={false}
            data={s.confianca.map((c, k) => ({ x: c * 100, y: s.acuracia[k] * 100 }))}
            fill={cor(s, i)} line={{ stroke: cor(s, i), strokeWidth: 2 }} shape="circle" />
        ))}
      </ScatterChart>
    </ResponsiveContainer>
  );
}

// ── Matriz de confusão (% de cada classe verdadeira) ────────────────────────

export function GraficoMatrizConfusao({ dados }: { dados: DadosMatriz }) {
  const { cores } = useEstilo();
  return (
    <div className="graficos-duplos">
      {dados.paradigmas.map(({ paradigma, matriz }) => (
        <div key={paradigma} className="grafico-metrica">
          <h4 className="grafico-subtitulo">{paradigma} (hold-out)</h4>
          <MapaCalor linhas={dados.classes} colunas={dados.classes} valores={matriz} escala={cores.sequencial}
            posicao={(v) => v} texto={(v) => `${Math.round(v * 100)}%`} larguraRotulo={72}
            dica={(l, c, v) => `verdadeira ${l} → prevista ${c}: ${v === null ? "—" : pctFracao(v)}`} />
          <p className="muted mapa-eixos">linhas: classe verdadeira · colunas: classe prevista</p>
        </div>
      ))}
    </div>
  );
}

// ── Integrated Gradients por eletrodo × banda (um mapa por classe) ──────────

export function GraficoGradesIG({ dados, paradigma }: { dados: DadosGradesIG; paradigma: string }) {
  const { cores } = useEstilo();
  const p = dados.paradigmas.find((x) => x.paradigma === paradigma) ?? dados.paradigmas[0];
  const vmax = dados.vmax || 1;
  return (
    <>
      <div className="mapas-tres">
        {dados.classes.map((classe, k) => (
          <MapaCalor key={classe} titulo={classe} linhas={dados.eletrodos} colunas={dados.bandas}
            valores={p.grades[k]} escala={cores.divergente} posicao={(v) => (v + vmax) / (2 * vmax)}
            larguraRotulo={40} rotuloColunasVertical
            dica={(l, c, v) => `${classe} · ${l} ${c}: ${v === null ? "—" : num(v, 4)}`} />
        ))}
      </div>
      <div className="mapa-legenda-solta">
        <LegendaEscala escala={cores.divergente} min={`−${num(vmax, 3)}`} meio="0" max={`+${num(vmax, 3)}`}
          titulo="Contribuição média para a classe (IG): azul = a favor · marrom = contra" />
      </div>
    </>
  );
}

// ── Curvas por classe ao longo da janela (IG no tempo e atenção do [CLS]) ───

function CurvasPorClasse({ classes, paineis, rotuloY }: {
  classes: string[];
  paineis: { titulo: string; segundos: number[]; valores: number[][] }[];
  rotuloY: string;
}) {
  const { cores, eixo, dica } = useEstilo();
  return (
    <>
      <div className="graficos-duplos">
        {paineis.map(({ titulo, segundos, valores }) => {
          const uniforme = 100 / segundos.length;
          const dados = segundos.map((s, i) => ({
            s, ...Object.fromEntries(classes.map((c, k) => [c, valores[k][i] * 100])),
          }));
          return (
            <div key={titulo} className="grafico-metrica">
              <h4 className="grafico-subtitulo">{titulo}</h4>
              <ResponsiveContainer width="100%" height={240}>
                <LineChart data={dados} margin={{ top: 8, right: 12, left: 0, bottom: 14 }}>
                  <CartesianGrid stroke={cores.grade} strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="s" type="number" domain={["dataMin", 0]} tickFormatter={(v) => num(v, 1)} {...eixo}
                    label={{ value: "segundos antes do registro classificado (0 = atual)", position: "insideBottom",
                      offset: -10, fill: cores.eixo, fontSize: 11 }} />
                  <YAxis tickFormatter={(v) => `${num(v, 1)}%`} {...eixo} width={52} />
                  <Tooltip {...dica} labelFormatter={(s) => `${num(Number(s), 2)} s`}
                    formatter={(v) => pct(Number(v), 2)} />
                  <ReferenceLine y={uniforme} stroke={cores.acaso} strokeDasharray="4 4" />
                  {classes.map((c, k) => (
                    <Line key={c} dataKey={c} stroke={cores.classes[k]} strokeWidth={1.8} isAnimationActive={false}
                      dot={segundos.length <= 40 ? { r: 2.5 } : false} />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
          );
        })}
      </div>
      <LegendaItens itens={[
        ...classes.map((c, k) => ({ rotulo: c, cor: cores.classes[k], forma: FORMAS_CLASSES[k] })),
        { rotulo: `uniforme (${rotuloY})`, cor: cores.acaso, tracejado: true },
      ]} />
    </>
  );
}

export function GraficoTempoIG({ dados }: { dados: DadosTempo }) {
  return (
    <CurvasPorClasse classes={dados.classes} rotuloY="mesma fração em toda a janela"
      paineis={dados.paradigmas.map((p) => ({ titulo: `${p.paradigma} (T = ${p.T})`, segundos: p.segundos,
        valores: p.fracoes }))} />
  );
}

export function GraficoAtencao({ dados }: { dados: DadosAtencao }) {
  return (
    <CurvasPorClasse classes={dados.classes} rotuloY="mesma atenção em todos os tokens"
      paineis={dados.paradigmas.map((p) => ({ titulo: `${p.paradigma} (${p.n_tokens} tokens)`, segundos: p.segundos,
        valores: p.atencao }))} />
  );
}

// ── Comparativo: acurácia por participante × modelo (um cenário) ───────────

export function GraficoMapaParticipantes({ dados, cenario }: { dados: DadosPorParticipante; cenario: number }) {
  const { cores } = useEstilo();
  const c = dados.cenarios[cenario] ?? dados.cenarios[0];
  const posicao = duasInclinacoes(10, ACASO, 80);
  return (
    <MapaCalor linhas={dados.participantes} colunas={dados.modelos}
      valores={c.matriz.map((linha) => linha.map((v) => (v === null ? null : v * 100)))}
      escala={cores.divergente} posicao={posicao} texto={(v) => num(v, 0)} larguraRotulo={64}
      dica={(u, m, v) => `${u} · ${m}: ${v === null ? "—" : pct(v)}`}
      legenda={{ min: "10%", meio: "33% (acaso)", max: "80%", titulo: "Acurácia balanceada no teste" }} />
  );
}
