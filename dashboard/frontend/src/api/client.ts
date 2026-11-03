// Acesso à API do dashboard (dashboard/backend/app.py). URLs relativas: funcionam
// servidas pelo backend e no "npm run dev" (via proxy do Vite).

export type Valor = string | number | null;
export type Linha = Record<string, Valor>;
export type Tabela = Linha[];

export interface FiguraInfo {
  nome: string;
  url: string;
}

export interface Etapa {
  etapa: string;
  nome: string;
  tabelas: Record<string, Tabela>;
  figuras: FiguraInfo[];
  graficos: Record<string, unknown>; // dados de cada figura (nome do PNG sem extensão)
}

export interface EtapaInfo {
  etapa: string;
  nome: string;
  comando: string;
  disponivel: boolean;
}

export class EtapaAusente extends Error {}

const cache = new Map<string, Promise<unknown>>();

function buscar<T>(url: string): Promise<T> {
  if (!cache.has(url)) {
    cache.set(
      url,
      // no-store: o navegador nunca reaproveita uma resposta antiga (ex.: de antes de reiniciar o backend)
      fetch(url, { cache: "no-store" })
        .then(async (res) => {
          if (res.status === 404) {
            const corpo = await res.json().catch(() => ({}));
            throw new EtapaAusente(corpo.detail ?? "Resultados não encontrados");
          }
          if (!res.ok) throw new Error(`Erro ${res.status} em ${url}`);
          if (!res.headers.get("content-type")?.includes("json"))
            throw new Error(
              "o servidor devolveu uma página em vez dos dados. O backend em execução é de uma versão " +
                "anterior: encerre o dashboard (Ctrl+C) e rode de novo python main.py dashboard.",
            );
          return res.json();
        })
        .catch((e) => {
          cache.delete(url); // em caso de erro, tenta de novo na próxima vez
          throw e;
        }),
    );
  }
  return cache.get(url) as Promise<T>;
}

export const listarEtapas = () => buscar<EtapaInfo[]>("/api/etapas");
export const buscarEtapa = (etapa: string) => buscar<Etapa>(`/api/${etapa}`);

// Simulador no navegador: teste do hold-out de um participante (semente 42), na ordem de gravação
export interface DadosSimulador {
  modelo: string;
  nome: string;
  paradigma: string;
  participante: string;
  semente: number;
  T: number;
  n: number;
  acerto_total: number;
  proba: number[][];
  y: number[];
  blocos: number[];
  registros: number[][];
  atencao: number[][] | null;
}

export const PARADIGMAS = ["especialista", "generalista"] as const;
export const PARTICIPANTES = ["user_a", "user_b", "user_c", "user_d", "user_e"] as const;

export const buscarSimulador = (modelo: string, paradigma: string, participante: string) =>
  buscar<DadosSimulador>(`/api/simulador/${modelo}?paradigma=${paradigma}&participante=${participante}`);

export const MODELOS = ["cnn", "bilstm", "bigru", "transformer", "cnn_transformer"] as const;
export const NOMES: Record<string, string> = {
  cnn: "CNN",
  bilstm: "Bi-LSTM",
  bigru: "Bi-GRU",
  transformer: "Transformer",
  cnn_transformer: "CNN + Transformer",
};

// Quatro cenários das tabelas principais (prefixo das colunas dos .csv)
export const CENARIOS = ["Hold-out generalista", "Hold-out especialista", "CV generalista", "CV especialista"];

// Número com vírgula decimal e casas fixas (ex.: 42 → "42,0").
export function num(v: Valor, casas = 1): string {
  if (v === null || v === undefined || v === "") return "—";
  if (typeof v === "number")
    return v.toLocaleString("pt-BR", { minimumFractionDigits: casas, maximumFractionDigits: casas });
  return v;
}

// Casas decimais de uma coluna de tabela pelo nome: % e pp com 1, ECE com 3, o resto como veio.
export function casasDaColuna(coluna: string, v: number): number {
  if (coluna.includes("(%)") || coluna.includes("(pp)")) return 1;
  if (coluna === "ECE") return 3;
  return Number.isInteger(v) ? 0 : Math.min(3, String(v).split(".")[1]?.length ?? 0);
}
