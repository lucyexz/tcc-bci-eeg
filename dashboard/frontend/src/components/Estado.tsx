import type { ReactNode } from "react";
import type { EstadoEtapa } from "../api/useEtapa";
import type { Etapa } from "../api/client";

interface Props {
  estado: EstadoEtapa;
  children: (dados: Etapa) => ReactNode;
}

// Carregando / etapa ainda não rodada / erro; com dados, renderiza o conteúdo.
export default function Estado({ estado, children }: Props) {
  if (estado.erro) return <div className="error-msg">Erro: {estado.erro}</div>;
  if (estado.ausente)
    return (
      <div className="aviso">
        <strong>Sem resultados ainda.</strong>
        <p>{estado.ausente}</p>
      </div>
    );
  if (!estado.dados) return <div className="loading">Carregando…</div>;
  return <>{children(estado.dados)}</>;
}
