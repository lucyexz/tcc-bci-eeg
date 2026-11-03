// Tema claro/escuro. Na primeira visita segue o sistema; depois do botão, a escolha fica salva.
//
// Cores neutras (as mesmas das figuras da monografia): nada de vermelho, verde, amarelo ou laranja,
// que leem como certo/errado ou alerta. Generalista = magenta, especialista = azul.
import { createContext, useContext, useEffect, useState, type ReactNode } from "react";

export type NomeTema = "claro" | "escuro";

export interface CoresGrafico {
  grade: string;
  eixo: string;
  texto: string;
  dicaFundo: string;
  dicaBorda: string;
  dp: string; // barras de desvio-padrão
  acaso: string; // linha dos 33,3%
  modelos: string[]; // uma cor por modelo (CNN, Bi-LSTM, Bi-GRU, Transformer, CNN + Transformer)
  generalista: string;
  generalistaClaro: string;
  especialista: string;
  especialistaClaro: string;
  classes: string[]; // Esquerda, Direita, Neutro (azul, magenta, ocre)
  contexto: string; // pontos de fundo (t-SNE) e linhas de ligação
  treino: string; // curva de treino (grafite)
  sequencial: string[]; // mapas de calor de 0 a máx. (do fundo ao azul forte)
  divergente: string[]; // marrom ↔ neutro ↔ azul
  particoes: Record<string, string>; // divisão: treino, validação, teste
}

const PALETAS: Record<NomeTema, CoresGrafico> = {
  claro: {
    grade: "#e1e0d9",
    eixo: "#52514e",
    texto: "#0b0b0b",
    dicaFundo: "#ffffff",
    dicaBorda: "#c3c2b7",
    dp: "#52514e",
    acaso: "#52514e",
    modelos: ["#2a78d6", "#b0478f", "#a8742f", "#6f55c2", "#184f95"],
    generalista: "#b0478f",
    generalistaClaro: "#e0b3d2",
    especialista: "#2a78d6",
    especialistaClaro: "#9ec5f4",
    classes: ["#2a78d6", "#b0478f", "#a8742f"],
    contexto: "#d6d4cc",
    treino: "#52514e",
    sequencial: ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
    divergente: ["#5c3d14", "#8a5a22", "#b98546", "#dcc29b", "#f0efec", "#9ec5f4", "#3987e5", "#184f95", "#0d366b"],
    particoes: { train: "#c3c2b7", val: "#a8742f", test: "#2a78d6" },
  },
  escuro: {
    grade: "#2d3348",
    eixo: "#a3a19a",
    texto: "#e2e8f0",
    dicaFundo: "#1e2130",
    dicaBorda: "#2d3348",
    dp: "#e2e8f0",
    acaso: "#a3a19a",
    modelos: ["#5b9be8", "#d07ab4", "#c9964f", "#9b85e0", "#9ec5f4"],
    generalista: "#d07ab4",
    generalistaClaro: "#7a4a6d",
    especialista: "#5b9be8",
    especialistaClaro: "#2f5a8f",
    classes: ["#5b9be8", "#d07ab4", "#c9964f"],
    contexto: "#3a4058",
    treino: "#a3a19a",
    // No escuro a escala parte do fundo e clareia: o valor alto é o que mais contrasta
    sequencial: ["#1b2030", "#172a48", "#1a3d6e", "#205496", "#2a6cc0", "#4a8ade", "#7cb0ee", "#b6d3f7"],
    divergente: ["#e8c48f", "#c9964f", "#9a6a2c", "#5c4222", "#2a2e3c", "#1f3f6e", "#2a6cc0", "#5b9be8", "#a9cbf5"],
    particoes: { train: "#5a5f72", val: "#c9964f", test: "#5b9be8" },
  },
};

const CHAVE = "tema";

function temaDoSistema(): NomeTema {
  return window.matchMedia?.("(prefers-color-scheme: light)").matches ? "claro" : "escuro";
}

function temaSalvo(): NomeTema | null {
  try {
    const v = localStorage.getItem(CHAVE);
    return v === "claro" || v === "escuro" ? v : null;
  } catch {
    return null;
  }
}

interface ValorTema {
  tema: NomeTema;
  cores: CoresGrafico;
  alternar: () => void;
}

const Contexto = createContext<ValorTema>({ tema: "escuro", cores: PALETAS.escuro, alternar: () => {} });

export function ProvedorTema({ children }: { children: ReactNode }) {
  const [escolhido, setEscolhido] = useState<NomeTema | null>(temaSalvo);
  const [sistema, setSistema] = useState<NomeTema>(temaDoSistema);
  const tema = escolhido ?? sistema;

  // Sem escolha salva, acompanha a troca de tema do sistema
  useEffect(() => {
    const mq = window.matchMedia?.("(prefers-color-scheme: light)");
    if (!mq) return;
    const ouvir = () => setSistema(mq.matches ? "claro" : "escuro");
    mq.addEventListener("change", ouvir);
    return () => mq.removeEventListener("change", ouvir);
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = tema;
  }, [tema]);

  const alternar = () => {
    const novo: NomeTema = tema === "claro" ? "escuro" : "claro";
    setEscolhido(novo);
    try {
      localStorage.setItem(CHAVE, novo);
    } catch {
      /* navegador sem armazenamento: a escolha vale só nesta aba */
    }
  };

  return <Contexto.Provider value={{ tema, cores: PALETAS[tema], alternar }}>{children}</Contexto.Provider>;
}

// eslint-disable-next-line react-refresh/only-export-components
export const useTema = () => useContext(Contexto);
