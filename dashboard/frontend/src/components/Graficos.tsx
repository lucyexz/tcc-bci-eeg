import {
  Bar,
  BarChart,
  CartesianGrid,
  ErrorBar,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { CENARIOS, num, type Tabela } from "../api/client";
import { useTema, type CoresGrafico } from "../tema";

const pct = (v: unknown) => (typeof v === "number" ? `${num(v)}%` : String(v));
const MARGEM = { top: 10, right: 44, left: 0, bottom: 5 };

// Peças comuns (grade, eixos, dica, legenda e linha do acaso) nas cores do tema
function useBase() {
  const { cores } = useTema();
  const eixo = { fill: cores.eixo, fontSize: 12 };
  return {
    cores,
    grade: <CartesianGrid stroke={cores.grade} strokeDasharray="3 3" vertical={false} />,
    eixoProps: { tick: eixo, stroke: cores.grade },
    dica: (
      <Tooltip
        formatter={pct}
        contentStyle={{ background: cores.dicaFundo, border: `1px solid ${cores.dicaBorda}`, borderRadius: 6 }}
        labelStyle={{ color: cores.texto }}
        itemStyle={{ color: cores.texto }}
        cursor={{ fill: cores.grade, opacity: 0.4 }}
      />
    ),
    legenda: <Legend itemSorter={() => 0} wrapperStyle={{ color: cores.texto }} />,
    acaso: (
      <ReferenceLine
        y={33.3}
        stroke={cores.acaso}
        strokeDasharray="4 4"
        label={{ value: "acaso", fill: cores.acaso, fontSize: 11, position: "right" }}
      />
    ),
  };
}

// Cenário → cor: generalista magenta, especialista azul; hold-out no tom claro, CV no tom cheio
function corCenario(cores: CoresGrafico, cenario: string): string {
  const esp = cenario.includes("especialista");
  if (cenario.startsWith("Hold-out")) return esp ? cores.especialistaClaro : cores.generalistaClaro;
  return esp ? cores.especialista : cores.generalista;
}

interface CenariosProps {
  linhas: Tabela;
  rotulo: string; // coluna com o nome de cada série (ex.: "Modelo")
  altura?: number;
}

// Barras por cenário (hold-out/CV × generalista/especialista), uma série por linha da tabela, com DP.
// Uma cor por modelo.
export function GraficoCenarios({ linhas, rotulo, altura = 320 }: CenariosProps) {
  const { cores, grade, eixoProps, dica, legenda, acaso } = useBase();
  const series = linhas.map((l) => String(l[rotulo]));
  const corDe = series.map((_, i) => cores.modelos[i % cores.modelos.length]);
  const dados = CENARIOS.map((c) => {
    const ponto: Record<string, string | number | null> = { cenario: c };
    linhas.forEach((l) => {
      ponto[String(l[rotulo])] = l[`${c} (%)`] as number;
      ponto[`${l[rotulo]} DP`] = l[`${c} DP (pp)`] as number;
    });
    return ponto;
  });
  return (
    <ResponsiveContainer width="100%" height={altura}>
      <BarChart data={dados} margin={MARGEM}>
        {grade}
        <XAxis dataKey="cenario" {...eixoProps} />
        <YAxis domain={[0, 70]} tickFormatter={(v) => `${v}%`} {...eixoProps} />
        {dica}
        {legenda}
        {acaso}
        {series.map((s, i) => (
          <Bar key={s} dataKey={s} fill={corDe[i]} radius={[3, 3, 0, 0]}>
            <ErrorBar dataKey={`${s} DP`} stroke={cores.dp} strokeWidth={1} width={3} />
          </Bar>
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}

// Acurácia por usuário nos quatro cenários (colunas "<cenário> (%)").
// O eixo X é a 1ª coluna da tabela (user_a…user_e), qualquer que seja o título dela.
export function GraficoParticipantes({ linhas, altura = 280 }: { linhas: Tabela; altura?: number }) {
  const { cores, grade, eixoProps, dica, legenda, acaso } = useBase();
  const eixoX = Object.keys(linhas[0] ?? {})[0];
  return (
    <ResponsiveContainer width="100%" height={altura}>
      <BarChart data={linhas} margin={MARGEM}>
        {grade}
        <XAxis dataKey={eixoX} {...eixoProps} />
        <YAxis domain={[0, 80]} tickFormatter={(v) => `${v}%`} {...eixoProps} />
        {dica}
        {legenda}
        {acaso}
        {CENARIOS.map((c) => (
          <Bar key={c} dataKey={`${c} (%)`} name={c} fill={corCenario(cores, c)} radius={[3, 3, 0, 0]} />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}

const CLASSES = ["Esquerda", "Direita", "Neutro"];
const METRICAS = ["Sensibilidade", "Precisão"];

interface PorClasseProps {
  linhas: Tabela; // metricas_por_classe já sem F1 e sem Macro (o backend filtra)
  eixo: string; // coluna do eixo X: "Modelo" (comparativo, um cenário) ou "Cenário" (um modelo)
  altura?: number;
}

// Sensibilidade e precisão por classe: dois gráficos lado a lado, barras agrupadas por classe.
// Sem barras de DP (chegam a 44 pp no hold-out e cobririam as barras); o DP aparece na dica e na tabela.
export function GraficoPorClasse({ linhas, eixo, altura = 280 }: PorClasseProps) {
  const { cores, grade, eixoProps, legenda, acaso } = useBase();
  const corClasse = [cores.modelos[0], cores.modelos[1], cores.modelos[2]];
  const valoresX = [...new Set(linhas.map((l) => String(l[eixo])))];
  return (
    <div className="graficos-duplos">
      {METRICAS.map((m) => {
        const dados = valoresX.map((x) => {
          const ponto: Record<string, string | number | null> = { x };
          linhas
            .filter((l) => String(l[eixo]) === x)
            .forEach((l) => {
              ponto[String(l["Classe"])] = l[`${m} (%)`] as number;
              ponto[`${l["Classe"]} DP`] = l[`${m} DP (pp)`] as number;
            });
          return ponto;
        });
        return (
          <div key={m} className="grafico-metrica">
            <h3 className="tabela-titulo">{m}</h3>
            <ResponsiveContainer width="100%" height={altura}>
              <BarChart data={dados} margin={MARGEM}>
                {grade}
                <XAxis dataKey="x" {...eixoProps} interval={0} />
                <YAxis domain={[0, 80]} tickFormatter={(v) => `${v}%`} {...eixoProps} />
                <Tooltip
                  formatter={(v, nome, item) => {
                    const dp = (item.payload as Record<string, unknown>)[`${nome} DP`];
                    return typeof dp === "number" ? `${pct(v)} (DP ${num(dp)} pp)` : pct(v);
                  }}
                  contentStyle={{ background: cores.dicaFundo, border: `1px solid ${cores.dicaBorda}`, borderRadius: 6 }}
                  labelStyle={{ color: cores.texto }}
                  itemStyle={{ color: cores.texto }}
                  cursor={{ fill: cores.grade, opacity: 0.4 }}
                />
                {legenda}
                {acaso}
                {CLASSES.map((c, i) => (
                  <Bar key={c} dataKey={c} fill={corClasse[i]} radius={[3, 3, 0, 0]} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
        );
      })}
    </div>
  );
}

// Rótulos verdadeiros (tom cheio) × permutados (tom claro), por paradigma.
// Colunas esperadas: "Generalista: rótulos verdadeiros", "Generalista: permutados", idem Especialista.
export function GraficoPermutacao({ linhas, altura = 280 }: { linhas: Tabela; altura?: number }) {
  const { cores, grade, eixoProps, dica, legenda, acaso } = useBase();
  const series = [
    { coluna: "Generalista: rótulos verdadeiros", cor: cores.generalista },
    { coluna: "Generalista: permutados", cor: cores.generalistaClaro },
    { coluna: "Especialista: rótulos verdadeiros", cor: cores.especialista },
    { coluna: "Especialista: permutados", cor: cores.especialistaClaro },
  ];
  return (
    <ResponsiveContainer width="100%" height={altura}>
      <BarChart data={linhas} margin={MARGEM}>
        {grade}
        <XAxis dataKey="Modelo" {...eixoProps} />
        <YAxis domain={[0, 60]} tickFormatter={(v) => `${v}%`} {...eixoProps} />
        {dica}
        {legenda}
        {acaso}
        {series.map((s) => (
          <Bar key={s.coluna} dataKey={s.coluna} fill={s.cor} radius={[3, 3, 0, 0]} />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}
