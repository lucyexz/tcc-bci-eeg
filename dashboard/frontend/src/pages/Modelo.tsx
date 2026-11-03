import { useEffect, useState } from "react";
import { useLocation, useParams } from "react-router-dom";
import { NOMES } from "../api/client";
import { useEtapa } from "../api/useEtapa";
import { corParadigma } from "../components/estiloGrafico";
import Estado from "../components/Estado";
import { FiguraInterativa } from "../components/Figura";
import { GraficoCenarios, GraficoParticipantes, GraficoPorClasse } from "../components/Graficos";
import {
  GraficoAtencao,
  GraficoCalibracao,
  GraficoCurvasModelo,
  GraficoGradesIG,
  GraficoHalteres,
  GraficoMatrizConfusao,
  GraficoPontosResultado,
  GraficoTempoIG,
  type DadosAtencao,
  type DadosCalibracao,
  type DadosCurvas,
  type DadosGradesIG,
  type DadosMatriz,
  type DadosPermutacao,
  type DadosResultado,
  type DadosTempo,
} from "../components/GraficosModelos";
import SimuladorEmbutido from "../components/SimuladorEmbutido";
import Tabela from "../components/Tabela";
import { useTema } from "../tema";

const PARADIGMAS = ["Generalista", "Especialista"];

function SeletorParadigma({ valor, mudar }: { valor: string; mudar: (v: string) => void }) {
  return (
    <select className="seletor" value={valor} onChange={(e) => mudar(e.target.value)}>
      {PARADIGMAS.map((p) => (
        <option key={p}>{p}</option>
      ))}
    </select>
  );
}

export default function Modelo() {
  const { nome = "" } = useParams();
  const estado = useEtapa(nome);
  const { cores } = useTema();
  const [paradigmaIG, setParadigmaIG] = useState("Especialista");
  const { hash } = useLocation();

  // /modelo/<nome>#simulador (ou o endereço antigo /modelo/<nome>/simulador) rola até o simulador,
  // depois que os resultados carregam (antes disso a página é curta demais para rolar)
  const carregou = !!estado.dados;
  useEffect(() => {
    if (hash === "#simulador") document.getElementById("simulador")?.scrollIntoView({ block: "start" });
  }, [hash, nome, carregou]);
  return (
    <div className="page">
      <div className="page-header">
        <h1>{NOMES[nome] ?? nome}</h1>
        <p className="muted">Hold-out por blocos (3 sementes) e validação cruzada por blocos (6 folds).</p>
      </div>
      <section className="section" id="simulador">
        <h2>Simulador</h2>
        <SimuladorEmbutido modelo={nome} key={nome} />
      </section>
      <Estado estado={estado}>
        {(etapa) => {
          const { tabelas } = etapa;
          return (
            <>
              <section className="section">
                <h2>Configuração</h2>
                <Tabela linhas={tabelas["configuracao"]} />
              </section>

              <section className="section">
                <h2>Resultado</h2>
                <GraficoCenarios linhas={tabelas["resultado"] ?? []} rotulo="Modelo" altura={280} />
                <Tabela linhas={tabelas["resultado"]} nota="Média e DP entre os 5 participantes." />
                <FiguraInterativa<DadosResultado> etapa={etapa} nome="01_resultado" titulo="Cada participante e a média"
                  nota="Acurácia balanceada no teste: pontos claros = cada participante; ponto cheio = média.">
                  {(d) => <GraficoPontosResultado dados={d} />}
                </FiguraInterativa>

                <h3 className="tabela-titulo">Por participante</h3>
                <GraficoParticipantes linhas={tabelas["por_participante"] ?? []} />
                <Tabela linhas={tabelas["por_participante"]} />
                {tabelas["metricas_por_classe"]?.length ? (
                  <>
                    <h3 className="tabela-titulo">Sensibilidade e precisão por classe</h3>
                    <p className="muted section-nota-sub">
                      Sensibilidade: proporção dos registros de uma classe identificados corretamente. Precisão:
                      proporção das previsões atribuídas a uma classe que estavam corretas.
                    </p>
                    <GraficoPorClasse linhas={tabelas["metricas_por_classe"]} eixo="Cenário" />
                    <Tabela linhas={tabelas["metricas_por_classe"]} nota="Média e DP (pp) entre os 5 participantes." />
                  </>
                ) : null}
                <FiguraInterativa<DadosMatriz> etapa={etapa} nome="02_matriz_confusao"
                  titulo="Matriz de confusão (% de cada classe verdadeira)"
                  nota="Com que classe cada uma é confundida. Hold-out, todos os treinos.">
                  {(d) => <GraficoMatrizConfusao dados={d} />}
                </FiguraInterativa>
              </section>

              <section className="section">
                <h2>Treino e controles</h2>
                <FiguraInterativa<DadosCurvas> etapa={etapa} nome="03_curvas_treino"
                  titulo="Curvas de treino (mediana dos treinos)">
                  {(d) => <GraficoCurvasModelo dados={d} />}
                </FiguraInterativa>
                <Tabela linhas={tabelas["treino"]} titulo="Validação, épocas e diferença treino − teste" />
                <FiguraInterativa<DadosPermutacao> etapa={etapa} nome="04_permutacao"
                  titulo="Permutação de rótulos (hold-out)"
                  nota="Rótulos embaralhados entre blocos no treino; sem vazamento, a acurácia cai para perto do acaso.">
                  {(d) => <GraficoHalteres dados={d} />}
                </FiguraInterativa>
                <Tabela linhas={tabelas["permutacao"]} />
                <FiguraInterativa<DadosCalibracao> etapa={etapa} nome="05_calibracao" titulo="Calibração (validação cruzada)"
                  nota="Confiança prevista × acurácia observada; a diagonal é a calibração perfeita.">
                  {(d) => <GraficoCalibracao dados={d} cor={(s) => corParadigma(cores, s.paradigma)} />}
                </FiguraInterativa>
                <Tabela linhas={tabelas["calibracao"]} />
              </section>

              <section className="section">
                <h2>Explicabilidade</h2>
                <FiguraInterativa<DadosGradesIG> etapa={etapa} nome="06_xai_eletrodo_banda"
                  titulo="Integrated Gradients por eletrodo × banda (soma na janela)"
                  controles={<SeletorParadigma valor={paradigmaIG} mudar={setParadigmaIG} />}>
                  {(d) => <GraficoGradesIG dados={d} paradigma={paradigmaIG} />}
                </FiguraInterativa>
                <Tabela
                  linhas={tabelas["explicabilidade"]}
                  titulo="Integrated Gradients"
                  nota="Eletrodo × banda com maior contribuição e fração da contribuição no registro mais recente da janela."
                />
                <FiguraInterativa<DadosTempo> etapa={etapa} nome="07_xai_tempo"
                  titulo="Onde na janela está a informação usada (fração do |IG| total)">
                  {(d) => <GraficoTempoIG dados={d} />}
                </FiguraInterativa>
                <FiguraInterativa<DadosAtencao> etapa={etapa} nome="08_atencao"
                  titulo="Atenção do [CLS] sobre os tokens de tempo">
                  {(d) => <GraficoAtencao dados={d} />}
                </FiguraInterativa>
                <Tabela linhas={tabelas["atencao"]} titulo="Atenção do token [CLS]" />
              </section>
            </>
          );
        }}
      </Estado>
    </div>
  );
}
