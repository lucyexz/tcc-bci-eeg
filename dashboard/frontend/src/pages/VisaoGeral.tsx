import { useState } from "react";
import { Link } from "react-router-dom";
import { CENARIOS, NOMES, num, type Linha } from "../api/client";
import { useEtapa } from "../api/useEtapa";
import Estado from "../components/Estado";
import { FiguraInterativa } from "../components/Figura";
import { GraficoCenarios, GraficoPermutacao, GraficoPorClasse } from "../components/Graficos";
import {
  GraficoCalibracao,
  GraficoCurvasTodos,
  GraficoHalteres,
  GraficoMapaParticipantes,
  GraficoPontosResultado,
  type DadosCalibracao,
  type DadosCurvas,
  type DadosPermutacao,
  type DadosPorParticipante,
  type DadosResultado,
} from "../components/GraficosModelos";
import Tabela from "../components/Tabela";
import { useTema } from "../tema";

const CHAVE: Record<string, string> = Object.fromEntries(Object.entries(NOMES).map(([k, v]) => [v, k]));

function Cartao({ linha }: { linha: Linha }) {
  const nome = String(linha["Modelo"]);
  return (
    <Link to={`/modelo/${CHAVE[nome]}`} className="card card-clickable">
      <div className="card-header">
        <span className="model-name">{nome}</span>
      </div>
      <div className="card-body">
        <div className="metric-row">
          <span className="metric-label">CV especialista</span>
          <span className="accuracy-badge">{num(linha["CV especialista (%)"])}%</span>
        </div>
        <div className="metric-row">
          <span className="metric-label">CV generalista</span>
          <span className="metric-value">{num(linha["CV generalista (%)"])}%</span>
        </div>
        <div className="metric-row">
          <span className="metric-label">Hold-out especialista</span>
          <span className="metric-value">{num(linha["Hold-out especialista (%)"])}%</span>
        </div>
      </div>
    </Link>
  );
}

// Sensibilidade e precisão por classe dos cinco modelos, um cenário por vez (gráfico e tabela)
function SecaoPorClasse({ linhas }: { linhas: Linha[] }) {
  const [cenario, setCenario] = useState("Hold-out especialista");
  if (!linhas.length) return null;
  const doCenario = linhas
    .filter((l) => l["Cenário"] === cenario)
    .map((l) => Object.fromEntries(Object.entries(l).filter(([c]) => c !== "Cenário")));
  return (
    <section className="section">
      <div className="cabecalho-secao">
        <h2>Sensibilidade e precisão por classe</h2>
        <select className="seletor" value={cenario} onChange={(e) => setCenario(e.target.value)}>
          {CENARIOS.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
      </div>
      <p className="muted section-nota">
        Sensibilidade: proporção dos registros de uma classe identificados corretamente. Precisão: proporção das
        previsões atribuídas a uma classe que estavam corretas.
      </p>
      <GraficoPorClasse linhas={doCenario} eixo="Modelo" />
      <Tabela linhas={doCenario} nota={`${cenario}. Média e DP (pp) entre os 5 participantes.`} />
    </section>
  );
}

export default function VisaoGeral() {
  const estado = useEtapa("comparativo");
  const { cores } = useTema();
  const [cenarioMapa, setCenarioMapa] = useState(3); // CV especialista
  const [paradigmaCurvas, setParadigmaCurvas] = useState("Especialista");
  return (
    <div className="page">
      <div className="page-header">
        <h1>Resultados</h1>
        <p className="muted">
          Acurácia balanceada no teste comum, média entre os 5 participantes (acaso = 33,3%). As cinco redes,
          nos dois protocolos por blocos e nos dois paradigmas.
        </p>
      </div>
      <Estado estado={estado}>
        {(etapa) => {
          const { tabelas } = etapa;
          const principal = tabelas["resultado_principal"] ?? [];
          const verdadeiro = (modelo: string, paradigma: string) =>
            principal.find((l) => l["Modelo"] === modelo)?.[`Hold-out ${paradigma.toLowerCase()} (%)`] ?? null;
          const permutacao = (tabelas["permutacao"] ?? []).reduce<Record<string, Linha>>((acc, l) => {
            const m = String(l["Modelo"]);
            const p = String(l["Paradigma"]);
            acc[m] ??= { Modelo: m };
            acc[m][`${p}: rótulos verdadeiros`] = verdadeiro(String(l["Modelo"]), p);
            acc[m][`${p}: permutados`] = l["Média (%)"];
            return acc;
          }, {});
          return (
            <>
              <div className="cards-grid">
                {principal.map((l) => (
                  <Cartao key={String(l["Modelo"])} linha={l} />
                ))}
              </div>

              <section className="section">
                <h2>Resultado principal</h2>
                <GraficoCenarios linhas={principal} rotulo="Modelo" />
                <Tabela linhas={principal} nota="Média e desvio-padrão (DP, em pontos percentuais) entre os 5 participantes." />
                <FiguraInterativa<DadosResultado> etapa={etapa} nome="01_resultado_principal"
                  titulo="Cada participante e a média"
                  nota="Mesmos números das barras, mostrando a dispersão entre os participantes que o DP resume.">
                  {(d) => <GraficoPontosResultado dados={d} />}
                </FiguraInterativa>
              </section>

              <SecaoPorClasse linhas={tabelas["metricas_por_classe"] ?? []} />

              <section className="section">
                <h2>Controle de permutação de rótulos</h2>
                <p className="muted section-nota">
                  Rótulos embaralhados entre blocos no treino; teste com os rótulos verdadeiros. Sem vazamento, a
                  acurácia cai para perto do acaso.
                </p>
                <GraficoPermutacao linhas={Object.values(permutacao)} />
                <FiguraInterativa<DadosPermutacao> etapa={etapa} nome="03_permutacao"
                  titulo="Distância entre permutados e verdadeiros">
                  {(d) => <GraficoHalteres dados={d} />}
                </FiguraInterativa>
                <Tabela linhas={tabelas["permutacao"]} />
              </section>

              <section className="section">
                <h2>Por participante</h2>
                <FiguraInterativa<DadosPorParticipante> etapa={etapa} nome="02_por_participante"
                  titulo="Acurácia de cada participante em cada modelo"
                  nota="Marrom: abaixo do acaso; azul: acima. Passe o mouse para ver o valor."
                  controles={
                    <select className="seletor" value={cenarioMapa} onChange={(e) => setCenarioMapa(Number(e.target.value))}>
                      {CENARIOS.map((c, i) => (
                        <option key={c} value={i}>
                          {c}
                        </option>
                      ))}
                    </select>
                  }>
                  {(d) => <GraficoMapaParticipantes dados={d} cenario={cenarioMapa} />}
                </FiguraInterativa>
                <Tabela linhas={tabelas["faixa_por_participante"]} nota="Menor e maior acurácia entre os 5 modelos." />
              </section>

              <section className="section">
                <h2>Treino e calibração</h2>
                <Tabela linhas={tabelas["configuracao"]} titulo="Configuração final" />
                <FiguraInterativa<DadosCurvas> etapa={etapa} nome="04_curvas_treino"
                  titulo="Curvas de treino (mediana dos treinos)"
                  controles={
                    <select className="seletor" value={paradigmaCurvas} onChange={(e) => setParadigmaCurvas(e.target.value)}>
                      <option>Generalista</option>
                      <option>Especialista</option>
                    </select>
                  }>
                  {(d) => <GraficoCurvasTodos dados={d} paradigma={paradigmaCurvas} />}
                </FiguraInterativa>
                <Tabela linhas={tabelas["treino"]} titulo="Melhor época e diferença treino − teste" />
                <FiguraInterativa<DadosCalibracao> etapa={etapa} nome="05_calibracao"
                  titulo="Calibração do especialista (validação cruzada)"
                  nota="Confiança prevista × acurácia observada; a diagonal é a calibração perfeita.">
                  {(d) => <GraficoCalibracao dados={d} cor={(_, i) => cores.modelos[i % cores.modelos.length]} />}
                </FiguraInterativa>
                <Tabela linhas={tabelas["calibracao"]} titulo="Calibração (validação cruzada)" />
              </section>
            </>
          );
        }}
      </Estado>
    </div>
  );
}
