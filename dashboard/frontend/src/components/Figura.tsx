import { useState, type ReactNode } from "react";
import type { Etapa, FiguraInfo } from "../api/client";
import { rotulo as legenda } from "../api/rotulos";

function Ampliada({ figura, fechar }: { figura: FiguraInfo; fechar: () => void }) {
  return (
    <div className="figura-ampliada" onClick={fechar}>
      <img src={figura.url} alt={legenda(figura.nome)} />
    </div>
  );
}

export default function Figura({ figura }: { figura: FiguraInfo }) {
  const [ampliada, setAmpliada] = useState(false);
  return (
    <>
      <figure className="figura" onClick={() => setAmpliada(true)}>
        <div className="figura-imagem">
          <img src={figura.url} alt={legenda(figura.nome)} loading="lazy" />
        </div>
        <figcaption>{legenda(figura.nome)}</figcaption>
      </figure>
      {ampliada && <Ampliada figura={figura} fechar={() => setAmpliada(false)} />}
    </>
  );
}

export function GradeFiguras({ figuras }: { figuras: FiguraInfo[] }) {
  if (!figuras.length) return null;
  return (
    <div className="grade-figuras">
      {figuras.map((f) => (
        <Figura key={f.url} figura={f} />
      ))}
    </div>
  );
}

interface InterativaProps<T> {
  etapa: Etapa;
  nome: string; // nome do PNG sem extensão, ex.: "02_matriz_confusao"
  titulo?: string;
  nota?: string;
  controles?: ReactNode; // seletores ao lado do título
  children: (dados: T) => ReactNode;
}

// Gráfico interativo de uma figura: desenhado a partir dos dados que a etapa gravou junto do PNG.
// Sem os dados (etapa rodada antes desta versão), mostra o PNG como antes.
// Embaixo, um link discreto abre a figura da monografia ampliada.
export function FiguraInterativa<T>({ etapa, nome, titulo, nota, controles, children }: InterativaProps<T>) {
  const [ampliada, setAmpliada] = useState(false);
  const figura = etapa.figuras.find((f) => f.nome === nome);
  const dados = etapa.graficos?.[nome] as T | undefined;
  if (!dados) return figura ? <GradeFiguras figuras={[figura]} /> : null;
  return (
    <div className="grafico-figura">
      {(titulo || controles) && (
        <div className="cabecalho-secao">
          <h3 className="tabela-titulo">{titulo ?? legenda(nome)}</h3>
          {controles}
        </div>
      )}
      {nota && <p className="muted section-nota-sub">{nota}</p>}
      {children(dados)}
      {figura && (
        <button className="link-figura" onClick={() => setAmpliada(true)}>
          ver figura da monografia
        </button>
      )}
      {ampliada && figura && <Ampliada figura={figura} fechar={() => setAmpliada(false)} />}
    </div>
  );
}
