import { useEffect, useState } from "react";
import { buscarEtapa, EtapaAusente, type Etapa } from "./client";

export interface EstadoEtapa {
  dados: Etapa | null;
  ausente: string | null; // mensagem quando a etapa ainda não foi rodada
  erro: string | null;
}

export function useEtapa(etapa: string): EstadoEtapa {
  const [estado, setEstado] = useState<EstadoEtapa>({ dados: null, ausente: null, erro: null });

  useEffect(() => {
    let ativo = true;
    buscarEtapa(etapa)
      .then((dados) => ativo && setEstado({ dados, ausente: null, erro: null }))
      .catch((e: Error) =>
        ativo &&
        setEstado({
          dados: null,
          ausente: e instanceof EtapaAusente ? e.message : null,
          erro: e instanceof EtapaAusente ? null : e.message,
        }),
      );
    return () => {
      ativo = false;
    };
  }, [etapa]);

  return estado;
}
