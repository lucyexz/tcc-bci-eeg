import { useState } from "react";
import { useSearchParams } from "react-router-dom";
import { rotulo as titulo } from "../api/rotulos";
import { useEtapa } from "../api/useEtapa";
import Estado from "../components/Estado";
import { FiguraInterativa } from "../components/Figura";
import {
  GraficoAnova,
  GraficoAutocorrelacao,
  GraficoBlocos,
  GraficoCorrelacao,
  GraficoDistribuicao,
  GraficoDivisao,
  GraficoEspectro,
  GraficoExtremos,
  GraficoPoder,
  GraficoTsne,
  GraficoVariancia,
  type DadosAnova,
  type DadosAutocorrelacao,
  type DadosBlocos,
  type DadosCorrelacao,
  type DadosDistribuicao,
  type DadosDivisao,
  type DadosEspectro,
  type DadosExtremos,
  type DadosPoder,
  type DadosTsne,
  type DadosVariancia,
} from "../components/GraficosDados";
import Tabela from "../components/Tabela";
import { num, PARTICIPANTES } from "../api/client";

const CLASSES = ["Esquerda", "Direita", "Neutro"];

// Destaque do t-SNE: um participante ou uma classe por vez (como os painéis da figura)
function SeletorDestaque({ valor, mudar }: { valor: string; mudar: (v: string) => void }) {
  return (
    <select className="seletor" value={valor} onChange={(e) => mudar(e.target.value)}>
      <optgroup label="Participante">
        {PARTICIPANTES.map((u, i) => (
          <option key={u} value={`participante:${i}`}>
            {u}
          </option>
        ))}
      </optgroup>
      <optgroup label="Classe">
        {CLASSES.map((c, k) => (
          <option key={c} value={`classe:${k}`}>
            {c}
          </option>
        ))}
      </optgroup>
    </select>
  );
}

function Eda() {
  const estado = useEtapa("eda");
  const [participanteEspectro, setParticipanteEspectro] = useState(0);
  const [destaqueGlobal, setDestaqueGlobal] = useState("participante:0");
  const [destaquePorPart, setDestaquePorPart] = useState("classe:0");
  return (
    <Estado estado={estado}>
      {(etapa) => {
        const t = etapa.tabelas;
        const tab = (nome: string) => <Tabela linhas={t[nome]} titulo={titulo(nome)} />;
        return (
          <>
            <section className="section">
              <h2>1. Descrição e qualidade</h2>
              {tab("registros_por_classe")}
              {tab("qualidade")}
              {tab("valores_nao_positivos")}
              {tab("estatisticas_descritivas")}
              {tab("balanco_classes")}
            </section>

            <section className="section">
              <h2>2. Estrutura da coleta</h2>
              <FiguraInterativa<DadosBlocos> etapa={etapa} nome="eda_01_duracao_blocos"
                titulo="Duração de cada bloco, na ordem de gravação (segundos, escala log)">
                {(d) => <GraficoBlocos dados={d} chave="segundos" log formatoY={(v) => `${num(v, 0)} s`}
                  rotuloValor="" />}
              </FiguraInterativa>
              {tab("duracao_blocos")}
            </section>

            <section className="section">
              <h2>3. Distribuição e correlação</h2>
              <FiguraInterativa<DadosDistribuicao> etapa={etapa} nome="eda_02_distribuicao_log"
                titulo="Distribuição dos valores (log)">
                {(d) => <GraficoDistribuicao dados={d} />}
              </FiguraInterativa>
              {tab("mediana_log")}
              <FiguraInterativa<DadosCorrelacao> etapa={etapa} nome="eda_03_correlacao"
                titulo="Correlação entre as 25 features (blocos = eletrodos)"
                nota="Calculada dentro de cada participante e depois promediada. Passe o mouse para ver o valor.">
                {(d) => <GraficoCorrelacao dados={d} />}
              </FiguraInterativa>
            </section>

            <section className="section">
              <h2>4. Valores extremos</h2>
              <FiguraInterativa<DadosExtremos> etapa={etapa} nome="eda_04_valores_extremos" titulo="Valores extremos">
                {(d) => <GraficoExtremos dados={d} />}
              </FiguraInterativa>
              {tab("valores_extremos")}
            </section>

            <section className="section">
              <h2>5. Dependência temporal</h2>
              <FiguraInterativa<DadosAutocorrelacao> etapa={etapa} nome="eda_05_autocorrelacao"
                titulo="Registros vizinhos são quase cópias (autocorrelação média das 25 features, log)">
                {(d) => <GraficoAutocorrelacao dados={d} />}
              </FiguraInterativa>
              {tab("autocorrelacao")}
              <FiguraInterativa<DadosPoder> etapa={etapa} nome="eda_06_poder_estatistico"
                titulo="Efeito de classe observado × efeito detectável">
                {(d) => <GraficoPoder dados={d} />}
              </FiguraInterativa>
              {tab("tamanho_amostral_efetivo")}
              {tab("efeito_classe")}
            </section>

            <section className="section">
              <h2>6. Bloco × classe</h2>
              <FiguraInterativa<DadosVariancia> etapa={etapa} nome="eda_07_variancia_bloco_classe"
                titulo="O sinal varia mais entre blocos do que entre classes">
                {(d) => <GraficoVariancia dados={d} />}
              </FiguraInterativa>
              {tab("variancia_bloco_classe")}
              <FiguraInterativa<DadosBlocos> etapa={etapa} nome="eda_08_deriva_sessao"
                titulo="Nível médio do sinal em cada bloco (log da potência)">
                {(d) => <GraficoBlocos dados={d} chave="nivel" formatoY={(v) => num(v, 2)} rotuloValor="nível" eixoProprio />}
              </FiguraInterativa>
              {tab("deriva_sessao")}
            </section>

            <section className="section">
              <h2>7. Diferenças entre classes</h2>
              <FiguraInterativa<DadosAnova> etapa={etapa} nome="eda_09_anova_classes"
                titulo="Diferença entre classes por eletrodo × banda e assimetria hemisférica">
                {(d) => <GraficoAnova dados={d} />}
              </FiguraInterativa>
              {tab("anova_classes")}
              <FiguraInterativa<DadosEspectro> etapa={etapa} nome="eda_10_espectro_classes"
                titulo="Espectro médio por classe"
                controles={
                  <select className="seletor" value={participanteEspectro}
                    onChange={(e) => setParticipanteEspectro(Number(e.target.value))}>
                    {PARTICIPANTES.map((u, i) => (
                      <option key={u} value={i}>
                        {u}
                      </option>
                    ))}
                  </select>
                }>
                {(d) => <GraficoEspectro dados={d} participante={participanteEspectro} />}
              </FiguraInterativa>
            </section>

            <section className="section">
              <h2>8. Participante × classe</h2>
              <FiguraInterativa<DadosTsne> etapa={etapa} nome="eda_11_tsne_global"
                titulo="t-SNE com padronização global: os grupos são os participantes"
                nota="600 registros por participante. Escolha quem destacar."
                controles={<SeletorDestaque valor={destaqueGlobal} mudar={setDestaqueGlobal} />}>
                {(d) => <GraficoTsne dados={d} destaque={destaqueGlobal} />}
              </FiguraInterativa>
              <FiguraInterativa<DadosTsne> etapa={etapa} nome="eda_12_tsne_por_participante"
                titulo="t-SNE com padronização por participante"
                controles={<SeletorDestaque valor={destaquePorPart} mudar={setDestaquePorPart} />}>
                {(d) => <GraficoTsne dados={d} destaque={destaquePorPart} />}
              </FiguraInterativa>
              {tab("silhueta")}
            </section>
          </>
        );
      }}
    </Estado>
  );
}

function Divisao() {
  const estado = useEtapa("divisao");
  const [divisao, setDivisao] = useState(0);
  return (
    <Estado estado={estado}>
      {(etapa) => {
        const { tabelas } = etapa;
        const nomes = (etapa.graficos?.["01_divisao_blocos"] as DadosDivisao | undefined)?.divisoes.map((d) => d.nome) ?? [];
        return (
          <>
            <section className="section">
              <FiguraInterativa<DadosDivisao> etapa={etapa} nome="01_divisao_blocos"
                titulo="Divisão dos registros de cada participante"
                nota="Cada faixa é a sessão de um participante, na ordem de gravação. Passe o mouse para ver os registros."
                controles={
                  <select className="seletor" value={divisao} onChange={(e) => setDivisao(Number(e.target.value))}>
                    {nomes.map((n, i) => (
                      <option key={n} value={i}>
                        {n}
                      </option>
                    ))}
                  </select>
                }>
                {(d) => <GraficoDivisao dados={d} divisao={divisao} />}
              </FiguraInterativa>
            </section>
            <section className="section">
              <Tabela linhas={tabelas["holdout"]} titulo="Hold-out por blocos (generalista)" />
              <Tabela linhas={tabelas["validacao_cruzada"]} titulo="Validação cruzada por blocos (generalista)" />
              <Tabela linhas={tabelas["especialista"]} titulo="Especialista (hold-out)" />
              <Tabela linhas={tabelas["janelas"]} titulo="Janelas causais (hold-out, generalista)" />
            </section>
          </>
        );
      }}
    </Estado>
  );
}

export default function Dados() {
  // ?aba=divisao abre direto a divisão (link compartilhável)
  const [params, setParams] = useSearchParams();
  const aba = params.get("aba") === "divisao" ? "divisao" : "eda";
  const setAba = (a: "eda" | "divisao") => setParams(a === "eda" ? {} : { aba: a }, { replace: true });
  return (
    <div className="page">
      <div className="page-header">
        <h1>Dados</h1>
        <p className="muted">Análise exploratória do dataset e divisão dos registros em treino, validação e teste.</p>
      </div>
      <div className="abas">
        <button className={aba === "eda" ? "aba ativa" : "aba"} onClick={() => setAba("eda")}>
          Análise exploratória
        </button>
        <button className={aba === "divisao" ? "aba ativa" : "aba"} onClick={() => setAba("divisao")}>
          Divisão dos dados
        </button>
      </div>
      {aba === "eda" ? <Eda /> : <Divisao />}
    </div>
  );
}
