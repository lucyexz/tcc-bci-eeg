// Gráficos interativos da EDA e da divisão dos dados (os dados vêm de resultados/<etapa>/graficos/).
// Mesmo conteúdo dos PNGs; participantes sem cor (pequenos múltiplos), classes em azul/magenta/ocre.
import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
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
import { FORMAS_CLASSES, marcador, pct, useEstilo } from "./estiloGrafico";
import LegendaItens from "./LegendaItens";
import MapaCalor, { LegendaEscala } from "./MapaCalor";

// ── Tipos dos .json ─────────────────────────────────────────────────────────

interface Bloco {
  bloco: number;
  classe: number;
  segundos?: number;
  nivel?: number;
}
export interface DadosBlocos {
  classes: string[];
  participantes: { participante: string; rho?: number; blocos: Bloco[] }[];
}
export interface DadosDistribuicao {
  exemplo: string;
  bordas: number[];
  contagens: Record<string, number[]>;
  assimetria: { feature: string; bruta: number; log: number }[];
}
export interface DadosCorrelacao {
  rotulos: string[];
  matriz: number[][];
}
export interface DadosExtremos {
  eletrodos: string[];
  bandas: string[];
  por_feature: number[][];
  participantes: string[];
  tercos_rotulos: string[];
  tercos: number[][];
}
export interface DadosAutocorrelacao {
  bloco: number;
  lags: number[];
  participantes: Record<string, number[]>;
}
export interface DadosPoder {
  eta2_min: number;
  eta2: number[];
  poder: number[];
  participantes: { participante: string; min: number; mediana: number; max: number }[];
}
export interface DadosVariancia {
  participantes: { participante: string; classe: number; bloco: number }[];
}
export interface DadosAnova {
  eletrodos: string[];
  bandas: string[];
  assimetrias: string[];
  vmax: number;
  limiar: number;
  participantes: { participante: string; grade: number[][]; grade_asy: number[][]; significativos: number }[];
}
export interface DadosEspectro {
  participantes: string[];
  eletrodos: string[];
  bandas: string[];
  classes: string[];
  media: number[][][][]; // participante × eletrodo × classe × banda
  ic: number[][][][];
}
export interface DadosTsne {
  participantes: string[];
  classes: string[];
  x: number[];
  y: number[];
  participante: number[];
  classe: number[];
}
export interface DadosDivisao {
  partes: Record<string, string>;
  participantes: string[];
  divisoes: { nome: string; trechos: Record<string, [number, number, string][]> }[];
}

function DicaLinhas({ active, payload, titulo }: {
  active?: boolean;
  payload?: { payload: Record<string, unknown> }[];
  titulo: (p: Record<string, unknown>) => string[];
}) {
  const { caixaDica } = useEstilo();
  if (!active || !payload?.length) return null;
  const [cabecalho, ...resto] = titulo(payload[0].payload);
  return (
    <div style={caixaDica}>
      <div style={{ fontWeight: 600 }}>{cabecalho}</div>
      {resto.map((l) => (
        <div key={l}>{l}</div>
      ))}
    </div>
  );
}

const legendaClasses = (classes: string[], cores: string[]) =>
  classes.map((c, k) => ({ rotulo: c, cor: cores[k], forma: FORMAS_CLASSES[k] }));

// ── eda_01 e eda_08: um valor por bloco, na ordem de gravação, por participante ──

export function GraficoBlocos({ dados, chave, log, formatoY, rotuloValor, eixoProprio }: {
  dados: DadosBlocos;
  chave: "segundos" | "nivel";
  log?: boolean;
  eixoProprio?: boolean; // eixo Y de cada participante (como na figura da deriva); senão, comum a todos
  formatoY: (v: number) => string;
  rotuloValor: string;
}) {
  const { cores, eixoPequeno } = useEstilo();
  const valores = dados.participantes.flatMap((p) => p.blocos.map((b) => b[chave] ?? 0));
  const min = Math.min(...valores);
  const max = Math.max(...valores);
  const dominio: [number, number] = log ? [Math.max(1, min * 0.8), max * 1.2] : [min - (max - min) * 0.08, max + (max - min) * 0.08];
  return (
    <>
      <div className="multiplos-empilhados">
        {dados.participantes.map((p) => (
          <div key={p.participante} className="multiplo-linha">
            <div className="multiplo-rotulo">
              {p.participante}
              {p.rho !== undefined && <span className="muted"> ρ = {num(p.rho, 2)}</span>}
            </div>
            <ResponsiveContainer width="100%" height={110}>
              <LineChart data={p.blocos} margin={{ top: 6, right: 12, left: 0, bottom: 0 }}>
                <CartesianGrid stroke={cores.grade} strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="bloco" type="number" domain={[0, 35]} ticks={[0, 5, 10, 15, 20, 25, 30, 35]}
                  {...eixoPequeno} />
                <YAxis scale={log ? "log" : "auto"} domain={eixoProprio ? ["auto", "auto"] : dominio}
                  tickFormatter={formatoY} width={44} tickCount={3} allowDataOverflow={!eixoProprio} {...eixoPequeno} />
                <Tooltip cursor={{ stroke: cores.grade }} content={
                  <DicaLinhas titulo={(b) => [`${p.participante} · bloco ${b.bloco}`,
                    `${dados.classes[b.classe as number]}: ${rotuloValor ? `${rotuloValor} ` : ""}${formatoY(b[chave] as number)}`]} />} />
                <Line dataKey={chave} stroke={cores.contexto} strokeWidth={1} isAnimationActive={false}
                  dot={(props: { cx?: number; cy?: number; index?: number; payload?: Bloco }) => {
                    const k = props.payload?.classe ?? 2;
                    const forma = marcador(FORMAS_CLASSES[k], 4, cores.classes[k], { borda: cores.dicaFundo });
                    return <g key={props.index}>{forma({ cx: props.cx, cy: props.cy })}</g>;
                  }} activeDot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        ))}
      </div>
      <p className="muted mapa-eixos">eixo horizontal: bloco (ordem de gravação)</p>
      <LegendaItens itens={legendaClasses(dados.classes, cores.classes)} />
    </>
  );
}

// ── eda_02: histograma de uma feature e assimetria das 25 ───────────────────

export function GraficoDistribuicao({ dados }: { dados: DadosDistribuicao }) {
  const { cores, eixo, eixoPequeno, dica } = useEstilo();
  const series = Object.keys(dados.contagens); // "user_b a user_e", "user_a"
  const barras = dados.bordas.slice(0, -1).map((b, i) => ({
    x: (b + dados.bordas[i + 1]) / 2,
    ...Object.fromEntries(series.map((s) => [s, dados.contagens[s][i]])),
  }));
  const n = dados.assimetria.length;
  const ponto = (qual: "bruta" | "log") =>
    dados.assimetria.map((a, i) => ({ x: a[qual], y: i, rotulo: a.feature, qual }));
  return (
    <div className="graficos-duplos">
      <div className="grafico-metrica">
        <h4 className="grafico-subtitulo">Distribuição de {dados.exemplo} (log10 da potência)</h4>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={barras} margin={{ top: 8, right: 12, left: 0, bottom: 4 }} barCategoryGap={0}>
            <CartesianGrid stroke={cores.grade} strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="x" tickFormatter={(v) => num(Number(v), 1)} interval={9} {...eixo} />
            <YAxis {...eixo} width={52} />
            <Tooltip {...dica} labelFormatter={(x) => `log10(potência) ≈ ${num(Number(x), 2)}`} />
            <Bar dataKey={series[0]} stackId="h" fill={cores.contexto} isAnimationActive={false} />
            <Bar dataKey={series[1]} stackId="h" fill={cores.classes[0]} isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
        <LegendaItens itens={[
          { rotulo: series[0], cor: cores.contexto, forma: "square" },
          { rotulo: series[1], cor: cores.classes[0], forma: "square" },
        ]} />
      </div>
      <div className="grafico-metrica">
        <h4 className="grafico-subtitulo">Assimetria de cada feature (mediana dos participantes)</h4>
        <ResponsiveContainer width="100%" height={n * 15 + 50}>
          <ScatterChart margin={{ top: 8, right: 12, left: 0, bottom: 4 }}>
            <CartesianGrid stroke={cores.grade} strokeDasharray="3 3" horizontal={false} />
            <XAxis type="number" dataKey="x" tickFormatter={(v) => num(v, 0)} {...eixo} />
            <YAxis type="number" dataKey="y" domain={[-0.6, n - 0.4]} ticks={dados.assimetria.map((_, i) => i)}
              tickFormatter={(i) => dados.assimetria[i]?.feature ?? ""} reversed width={78} interval={0}
              {...eixoPequeno} />
            <Tooltip cursor={false} content={
              <DicaLinhas titulo={(p) => [String(p.rotulo), `assimetria ${p.qual}: ${num(p.x as number, 2)}`]} />} />
            <ReferenceLine x={0} stroke={cores.eixo} />
            <Scatter data={ponto("bruta")} isAnimationActive={false} shape={marcador("square", 3.5, cores.treino)} />
            <Scatter data={ponto("log")} isAnimationActive={false} shape={marcador("circle", 4, cores.classes[0])} />
          </ScatterChart>
        </ResponsiveContainer>
        <LegendaItens itens={[
          { rotulo: "bruta", cor: cores.treino, forma: "square" },
          { rotulo: "log", cor: cores.classes[0], forma: "circle" },
        ]} />
      </div>
    </div>
  );
}

// ── eda_03: correlação entre as 25 features ─────────────────────────────────

export function GraficoCorrelacao({ dados }: { dados: DadosCorrelacao }) {
  const { cores } = useEstilo();
  return (
    <div className="mapa-largo">
      <MapaCalor linhas={dados.rotulos} colunas={dados.rotulos} valores={dados.matriz} escala={cores.divergente}
        posicao={(v) => (v + 1) / 2} larguraRotulo={72} rotuloColunasVertical
        dica={(l, c, v) => `${l} × ${c}: r = ${v === null ? "—" : num(v, 2)}`}
        legenda={{ min: "−1", meio: "0", max: "+1", titulo: "Correlação de Pearson (log), média dos participantes" }} />
    </div>
  );
}

// ── eda_04: valores fora de 1,5×IQR ─────────────────────────────────────────

export function GraficoExtremos({ dados }: { dados: DadosExtremos }) {
  const { cores } = useEstilo();
  const maxF = Math.max(...dados.por_feature.flat());
  const maxT = Math.max(...dados.tercos.flat());
  return (
    <div className="graficos-duplos">
      <div className="grafico-metrica">
        <h4 className="grafico-subtitulo">Valores fora de 1,5×IQR por feature (%)</h4>
        <MapaCalor linhas={dados.eletrodos} colunas={dados.bandas} valores={dados.por_feature}
          escala={cores.sequencial} posicao={(v) => v / maxF} texto={(v) => num(v, 1)} larguraRotulo={44}
          dica={(l, c, v) => `${l} ${c}: ${v === null ? "—" : pct(v)}`} />
      </div>
      <div className="grafico-metrica">
        <h4 className="grafico-subtitulo">Registros com algum valor fora de 1,5×IQR, por terço da sessão</h4>
        <MapaCalor linhas={dados.participantes} colunas={dados.tercos_rotulos} valores={dados.tercos}
          escala={cores.sequencial} posicao={(v) => v / maxT} texto={(v) => `${num(v, 0)}%`} larguraRotulo={56}
          dica={(l, c, v) => `${l} · ${c}: ${v === null ? "—" : pct(v)}`} />
      </div>
    </div>
  );
}

// ── eda_05: autocorrelação média por distância ──────────────────────────────

export function GraficoAutocorrelacao({ dados }: { dados: DadosAutocorrelacao }) {
  const { cores, eixoPequeno, dica } = useEstilo();
  return (
    <>
      <div className="pequenos-multiplos cinco">
        {Object.entries(dados.participantes).map(([u, acf]) => (
          <div key={u} className="grafico-metrica">
            <h4 className="grafico-subtitulo">{u}</h4>
            <ResponsiveContainer width="100%" height={170}>
              <LineChart data={dados.lags.map((l, i) => ({ l, v: acf[i] }))}
                margin={{ top: 6, right: 8, left: -22, bottom: 0 }}>
                <CartesianGrid stroke={cores.grade} strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="l" type="number" scale="log" domain={[1, 1200]} ticks={[1, 10, 100, 1000]}
                  {...eixoPequeno} />
                <YAxis domain={[-0.1, 1]} ticks={[0, 0.5, 1]} tickFormatter={(v) => num(v, 1)} {...eixoPequeno} />
                <Tooltip {...dica} labelFormatter={(l) => `distância ${l} registros`}
                  formatter={(v) => num(Number(v), 3)} />
                <ReferenceLine y={0} stroke={cores.eixo} />
                <ReferenceLine x={dados.bloco} stroke={cores.acaso} strokeDasharray="4 4" />
                <Line dataKey="v" name="autocorrelação" stroke={cores.classes[0]} dot={false} strokeWidth={1.8}
                  isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        ))}
      </div>
      <p className="muted mapa-eixos">
        eixo horizontal: distância entre registros (escala log); linha tracejada: duração mediana de um bloco
        ({dados.bloco} registros)
      </p>
    </>
  );
}

// ── eda_06: poder da ANOVA × efeito observado ───────────────────────────────

export function GraficoPoder({ dados }: { dados: DadosPoder }) {
  const { cores, eixo, dica } = useEstilo();
  const n = dados.participantes.length;
  return (
    <>
      <ResponsiveContainer width="100%" height={240}>
        <LineChart data={dados.eta2.map((e, i) => ({ e, p: dados.poder[i] }))}
          margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
          <CartesianGrid stroke={cores.grade} strokeDasharray="3 3" />
          <XAxis dataKey="e" type="number" domain={[0, 0.6]} ticks={[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
            tickFormatter={(v) => num(v, 1)} {...eixo} />
          <YAxis domain={[0, 1]} tickFormatter={(v) => num(v, 1)} {...eixo} width={48}
            label={{ value: "poder da ANOVA", angle: -90, position: "insideLeft", fill: cores.eixo, fontSize: 11 }} />
          <Tooltip {...dica} labelFormatter={(e) => `η² = ${num(Number(e), 3)}`}
            formatter={(v) => num(Number(v), 2)} />
          <ReferenceLine y={0.8} stroke={cores.acaso} strokeDasharray="4 4" />
          <ReferenceLine x={dados.eta2_min} stroke={cores.acaso} strokeDasharray="4 4"
            label={{ value: `η² = ${num(dados.eta2_min, 2)}: menor efeito detectável com poder 0,8`,
              fill: cores.eixo, fontSize: 11, position: "insideBottomRight" }} />
          <Line dataKey="p" name="poder" stroke={cores.classes[0]} dot={false} strokeWidth={2} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
      <ResponsiveContainer width="100%" height={n * 26 + 50}>
        <ScatterChart margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
          <CartesianGrid stroke={cores.grade} strokeDasharray="3 3" horizontal={false} />
          <XAxis type="number" dataKey="x" domain={[0, 0.6]} ticks={[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
            tickFormatter={(v) => num(v, 1)} {...eixo} />
          <YAxis type="number" dataKey="y" domain={[-0.6, n - 0.4]} ticks={dados.participantes.map((_, i) => i)}
            tickFormatter={(i) => dados.participantes[i]?.participante ?? ""} reversed width={48} {...eixo} />
          <Tooltip cursor={false} content={<DicaLinhas titulo={(p) => [String(p.u),
            `η² mediano ${num(p.x as number, 3)} (faixa ${num(p.min as number, 3)} a ${num(p.max as number, 3)})`]} />} />
          <ReferenceLine x={dados.eta2_min} stroke={cores.acaso} strokeDasharray="4 4" />
          {dados.participantes.map((p, i) => (
            <ReferenceLine key={p.participante} stroke={cores.treino} strokeWidth={1.4}
              segment={[{ x: p.min, y: i }, { x: p.max, y: i }]} />
          ))}
          <Scatter data={dados.participantes.map((p, i) => ({ x: p.mediana, y: i, u: p.participante, min: p.min, max: p.max }))}
            isAnimationActive={false} shape={marcador("circle", 5, cores.treino)} />
        </ScatterChart>
      </ResponsiveContainer>
      <p className="muted mapa-eixos">
        Embaixo: efeito da classe (η² nas médias por bloco) em cada participante, com a mediana (●) e a faixa das 25
        features.
      </p>
    </>
  );
}

// ── eda_07: variância entre classes × entre blocos ──────────────────────────

export function GraficoVariancia({ dados }: { dados: DadosVariancia }) {
  const { cores, eixo } = useEstilo();
  const n = dados.participantes.length;
  const ponto = (qual: "classe" | "bloco") => dados.participantes.map((p, i) => ({
    x: p[qual], y: i, u: p.participante, qual: qual === "classe" ? "η² classe" : "η² bloco além da classe",
  }));
  return (
    <>
      <ResponsiveContainer width="100%" height={n * 34 + 60}>
        <ScatterChart margin={{ top: 8, right: 16, left: 0, bottom: 4 }}>
          <CartesianGrid stroke={cores.grade} strokeDasharray="3 3" horizontal={false} />
          <XAxis type="number" dataKey="x" scale="log" domain={[0.3, 60]} ticks={[0.5, 1, 2, 5, 10, 20, 50]}
            tickFormatter={(v) => `${num(v, v < 1 ? 1 : 0)}%`} allowDataOverflow {...eixo} />
          <YAxis type="number" dataKey="y" domain={[-0.6, n - 0.4]} ticks={dados.participantes.map((_, i) => i)}
            tickFormatter={(i) => dados.participantes[i]?.participante ?? ""} reversed width={52} {...eixo} />
          <Tooltip cursor={false} content={<DicaLinhas titulo={(p) => [String(p.u), `${p.qual}: ${pct(p.x as number)}`]} />} />
          {dados.participantes.map((p, i) => (
            <ReferenceLine key={p.participante} stroke={cores.contexto} segment={[{ x: p.classe, y: i }, { x: p.bloco, y: i }]} />
          ))}
          <Scatter data={ponto("classe")} isAnimationActive={false} shape={marcador("square", 4.5, cores.treino)} />
          <Scatter data={ponto("bloco")} isAnimationActive={false} shape={marcador("circle", 5, cores.classes[0])} />
        </ScatterChart>
      </ResponsiveContainer>
      <LegendaItens itens={[
        { rotulo: "η² classe", cor: cores.treino, forma: "square" },
        { rotulo: "η² bloco além da classe", cor: cores.classes[0], forma: "circle" },
      ]} />
      <p className="muted mapa-eixos">fração da variância explicada (%, escala log), mediana das 25 features</p>
    </>
  );
}

// ── eda_09: −log10(p) da ANOVA entre classes, por participante ──────────────

export function GraficoAnova({ dados }: { dados: DadosAnova }) {
  const { cores } = useEstilo();
  const posicao = (v: number) => v / dados.vmax;
  const dica = (u: string) => (l: string, c: string, v: number | null) =>
    `${u} · ${l} ${c}: −log10(p) = ${v === null ? "—" : num(v, 2)}${v !== null && v >= dados.limiar ? " (p < 0,05)" : ""}`;
  return (
    <>
      <div className="pequenos-multiplos cinco">
        {dados.participantes.map((p, i) => (
          <div key={p.participante} className="grafico-metrica">
            <h4 className="grafico-subtitulo">{p.participante} ({p.significativos}/25)</h4>
            <MapaCalor linhas={dados.eletrodos} colunas={dados.bandas} valores={p.grade} escala={cores.sequencial}
              posicao={posicao} contorno={(v) => v >= dados.limiar} larguraRotulo={i === 0 ? 58 : 0}
              dica={dica(p.participante)} rotuloColunasVertical />
            <MapaCalor linhas={dados.assimetrias} colunas={dados.bandas} valores={p.grade_asy}
              escala={cores.sequencial} posicao={posicao} contorno={(v) => v >= dados.limiar}
              larguraRotulo={i === 0 ? 58 : 0} dica={dica(p.participante)} rotuloColunasVertical />
          </div>
        ))}
      </div>
      <LegendaEscala escala={cores.sequencial} min="0" max={num(dados.vmax, 1)}
        titulo="−log10(p), ANOVA entre classes (contorno = p < 0,05). Em cima: eletrodo × banda; embaixo: assimetria hemisférica" />
    </>
  );
}

// ── eda_10: espectro médio por classe (um participante por vez) ─────────────

export function GraficoEspectro({ dados, participante }: { dados: DadosEspectro; participante: number }) {
  const { cores, eixoPequeno, dica } = useEstilo();
  const todos = dados.media.flat(3);
  const lim = Math.ceil(Math.max(...todos.map(Math.abs)) * 1.6 * 10) / 10;
  return (
    <>
      <div className="pequenos-multiplos cinco">
        {dados.eletrodos.map((el, e) => {
          const pontos = dados.bandas.map((b, j) => ({
            banda: b,
            ...Object.fromEntries(dados.classes.flatMap((c, k) => {
              const m = dados.media[participante][e][k][j];
              const ic = dados.ic[participante][e][k][j];
              return [[c, m], [`${c} IC`, [m - ic, m + ic]]];
            })),
          }));
          return (
            <div key={el} className="grafico-metrica">
              <h4 className="grafico-subtitulo">{el}</h4>
              <ResponsiveContainer width="100%" height={190}>
                <ComposedChart data={pontos} margin={{ top: 6, right: 8, left: -22, bottom: 0 }}>
                  <CartesianGrid stroke={cores.grade} strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="banda" interval={0} angle={-35} textAnchor="end" height={40} {...eixoPequeno} />
                  <YAxis domain={[-lim, lim]} tickFormatter={(v) => num(v, 1)} {...eixoPequeno} />
                  <Tooltip {...dica} formatter={(v, nome) => Array.isArray(v)
                    ? [`${num(Number(v[0]), 2)} a ${num(Number(v[1]), 2)}`, String(nome)]
                    : [num(Number(v), 2), String(nome)]} />
                  <ReferenceLine y={0} stroke={cores.eixo} />
                  {dados.classes.map((c, k) => (
                    <Area key={`${c} IC`} dataKey={`${c} IC`} name={`${c} (IC 95%)`} stroke="none"
                      fill={cores.classes[k]} fillOpacity={0.12} isAnimationActive={false} />
                  ))}
                  {dados.classes.map((c, k) => (
                    <Line key={c} dataKey={c} stroke={cores.classes[k]} strokeWidth={1.6} isAnimationActive={false}
                      dot={{ r: 2.5, fill: cores.classes[k] }} />
                  ))}
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          );
        })}
      </div>
      <LegendaItens itens={legendaClasses(dados.classes, cores.classes)} />
      <p className="muted mapa-eixos">log padronizado dentro do participante; faixa = IC 95% entre blocos</p>
    </>
  );
}

// ── eda_11 / eda_12: t-SNE, destacando um participante ou uma classe ────────

export function GraficoTsne({ dados, destaque }: { dados: DadosTsne; destaque: string }) {
  const { cores } = useEstilo();
  const [tipo, valor] = destaque.split(":"); // "participante:2" ou "classe:0"
  const idx = Number(valor);
  const sel = (i: number) => (tipo === "classe" ? dados.classe[i] === idx : dados.participante[i] === idx);
  const contexto: { x: number; y: number }[] = [];
  const foco: { x: number; y: number }[] = [];
  dados.x.forEach((x, i) => (sel(i) ? foco : contexto).push({ x, y: dados.y[i] }));
  const cor = tipo === "classe" ? cores.classes[idx] : cores.classes[0];
  const forma = tipo === "classe" ? FORMAS_CLASSES[idx] : "circle";
  return (
    <>
      <ResponsiveContainer width="100%" height={420}>
        <ScatterChart margin={{ top: 8, right: 8, left: 8, bottom: 8 }}>
          <XAxis type="number" dataKey="x" hide domain={["dataMin", "dataMax"]} />
          <YAxis type="number" dataKey="y" hide domain={["dataMin", "dataMax"]} />
          <Scatter data={contexto} isAnimationActive={false} shape={marcador("circle", 1.6, cores.contexto)} />
          <Scatter data={foco} isAnimationActive={false} shape={marcador(forma, 2.2, cor)} />
        </ScatterChart>
      </ResponsiveContainer>
      <LegendaItens itens={[
        { rotulo: tipo === "classe" ? dados.classes[idx] : dados.participantes[idx], cor, forma },
        { rotulo: "demais registros da amostra", cor: cores.contexto, forma: "circle" },
      ]} />
    </>
  );
}

// ── Divisão: linha do tempo de cada participante, colorida pela partição ────

export function GraficoDivisao({ dados, divisao }: { dados: DadosDivisao; divisao: number }) {
  const { cores } = useEstilo();
  const d = dados.divisoes[divisao] ?? dados.divisoes[0];
  const total = Math.max(...Object.values(d.trechos).map((t) => t[t.length - 1][1]));
  const L = 1000; // largura do viewBox
  const linhaH = 26;
  return (
    <>
      <div className="linha-tempo">
        {dados.participantes.map((u) => (
          <div key={u} className="linha-tempo-linha">
            <span className="linha-tempo-rotulo">{u}</span>
            <svg viewBox={`0 0 ${L} ${linhaH}`} preserveAspectRatio="none" className="linha-tempo-svg">
              {d.trechos[u].map(([ini, fim, parte]) =>
                parte === "purge" ? null : (
                  <rect key={ini} x={(ini / total) * L} y={2} width={Math.max(0.5, ((fim - ini) / total) * L)}
                    height={linhaH - 4} fill={cores.particoes[parte]}>
                    <title>{`${u} · ${dados.partes[parte]}: registros ${ini} a ${fim - 1} (${fim - ini})`}</title>
                  </rect>
                ),
              )}
            </svg>
          </div>
        ))}
        <div className="linha-tempo-eixo">
          <span>0</span>
          <span>{num(total / 2, 0)}</span>
          <span>{num(total, 0)} registros</span>
        </div>
      </div>
      <LegendaItens itens={[
        ...Object.entries(dados.partes).map(([p, nome]) => ({ rotulo: nome, cor: cores.particoes[p], forma: "square" })),
        { rotulo: "purge (descartado, em branco)", cor: cores.grade, forma: "vazado" },
      ]} />
    </>
  );
}
