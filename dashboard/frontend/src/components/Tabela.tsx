import { useState } from "react";
import { casasDaColuna, num, type Tabela as TabelaDados, type Valor } from "../api/client";

const formatar = (coluna: string, v: Valor) => (typeof v === "number" ? num(v, casasDaColuna(coluna, v)) : num(v));

// Quanto mais colunas, mais compacta a tabela, para caber na largura sem rolagem lateral
const densidade = (n: number) => (n > 14 ? "muito-compacta" : n > 7 ? "compacta" : "");

interface Agrupamento {
  grupos: { nome: string; n: number }[];
  sub: string[]; // subtítulo de cada coluna depois da 1ª
}

// Colunas "user_a Alpha", "user_a BetaH", …, "user_b Alpha"… → cabeçalho em dois níveis
// (grupo "user_a" em cima, "Alpha" embaixo), bem mais estreito. Só quando todas as colunas
// depois da 1ª seguem "<grupo> <sub>" e cada grupo tem 2 ou mais colunas seguidas.
function agrupar(colunas: string[]): Agrupamento | null {
  const partes = colunas.slice(1).map((c) => /^(\S+) (\S+)$/.exec(c));
  if (partes.length < 4 || partes.some((p) => !p)) return null;
  const grupos: Agrupamento["grupos"] = [];
  for (const p of partes) {
    const ultimo = grupos[grupos.length - 1];
    if (ultimo && ultimo.nome === p![1]) ultimo.n++;
    else grupos.push({ nome: p![1], n: 1 });
  }
  if (grupos.some((g) => g.n < 2)) return null;
  return { grupos, sub: partes.map((p) => p![2]) };
}

interface Props {
  linhas: TabelaDados | undefined;
  titulo?: string;
  nota?: string;
}

// Ordenação por clique no cabeçalho. Nenhuma linha some: só muda de lugar.
// Coluna numérica: maior → menor ("desc"), menor → maior ("asc"), ordem original.
// Coluna de texto: junta as linhas de mesmo valor (ex.: todas as "Esquerda" seguidas), na ordem em
// que cada valor aparece primeiro ("asc"), depois com os grupos invertidos ("desc"), depois a original.
// A ordenação é estável: dentro de um grupo ou empate, vale a ordem original do .csv.
type Sentido = "asc" | "desc";
interface Ordem {
  coluna: string;
  sentido: Sentido;
}

function ordenar(linhas: TabelaDados, ordem: Ordem | null, numerica: boolean): TabelaDados {
  if (!ordem) return linhas;
  const { coluna, sentido } = ordem;
  const sinal = sentido === "asc" ? 1 : -1;
  const vazio = (v: Valor) => v === null || v === undefined || v === "";
  // Texto: posição da primeira aparição de cada valor (Esquerda, Direita, Neutro…), não ordem alfabética
  const posicao = new Map<string, number>();
  if (!numerica) linhas.forEach((l) => posicao.has(String(l[coluna])) || posicao.set(String(l[coluna]), posicao.size));
  return [...linhas].sort((a, b) => {
    const va = a[coluna];
    const vb = b[coluna];
    if (vazio(va) || vazio(vb)) return Number(vazio(va)) - Number(vazio(vb)); // vazios sempre no fim
    if (numerica && typeof va === "number" && typeof vb === "number") return sinal * (va - vb);
    return sinal * (posicao.get(String(va))! - posicao.get(String(vb))!);
  });
}

const PROXIMO: Record<string, Sentido | null> = { nenhum: "desc", desc: "asc", asc: null };
const PROXIMO_TEXTO: Record<string, Sentido | null> = { nenhum: "asc", asc: "desc", desc: null };

// Tabela genérica: cabeçalho = colunas do .csv; números com vírgula decimal.
// O alinhamento é decidido por coluna (numérica → direita, texto → esquerda) e vale para o
// cabeçalho e para todas as células, para o título ficar exatamente em cima dos valores.
export default function Tabela({ linhas, titulo, nota }: Props) {
  const [ordem, setOrdem] = useState<Ordem | null>(null);
  if (!linhas || !linhas.length) return null;
  const colunas = Object.keys(linhas[0]);
  const numerica = Object.fromEntries(
    colunas.map((c) => [c, linhas.some((l) => typeof l[c] === "number")]),
  ) as Record<string, boolean>;
  const agrupamento = agrupar(colunas);
  // A ordem escolhida continua valendo quando a tabela muda (ex.: troca de cenário), se a coluna existir
  const ativa = ordem && colunas.includes(ordem.coluna) ? ordem : null;
  const visiveis = ordenar(linhas, ativa, ativa ? numerica[ativa.coluna] : false);

  const clicar = (c: string) => {
    const atual = ativa?.coluna === c ? ativa.sentido : "nenhum";
    const proximo = (numerica[c] ? PROXIMO : PROXIMO_TEXTO)[atual];
    setOrdem(proximo ? { coluna: c, sentido: proximo } : null);
  };
  const dica = (c: string) => {
    const atual = ativa?.coluna === c ? ativa.sentido : "nenhum";
    const proximo = (numerica[c] ? PROXIMO : PROXIMO_TEXTO)[atual];
    if (!proximo) return "Voltar à ordem original";
    if (numerica[c]) return proximo === "desc" ? "Ordenar do maior para o menor" : "Ordenar do menor para o maior";
    return proximo === "asc" ? "Agrupar as linhas de mesmo valor" : "Inverter a ordem dos grupos";
  };
  // Cabeçalho clicável: o texto continua o mesmo; a seta só aparece na coluna ordenada (e de leve no hover)
  const cabecalho = (c: string, texto: string) => {
    const sentido = ativa?.coluna === c ? ativa.sentido : null;
    return (
      <button type="button" className={`ordenar${sentido ? " ativa" : ""}`} onClick={() => clicar(c)} title={dica(c)}>
        {texto}
        <span className="seta" aria-hidden="true">
          {sentido === "asc" ? "▲" : sentido === "desc" ? "▼" : "↕"}
        </span>
      </button>
    );
  };
  const ariaSort = (c: string) =>
    ativa?.coluna !== c ? undefined : ativa.sentido === "asc" ? "ascending" : "descending";

  // Colunas que abrem um grupo ganham uma linha vertical à esquerda
  const inicioGrupo = new Set<number>();
  if (agrupamento) {
    let i = 1;
    for (const g of agrupamento.grupos) {
      inicioGrupo.add(i);
      i += g.n;
    }
  }
  const classe = (c: string, i: number) =>
    [numerica[c] ? "num" : "", inicioGrupo.has(i) ? "inicio-grupo" : ""].filter(Boolean).join(" ");

  return (
    <div className="tabela-bloco">
      {titulo && <h3 className="tabela-titulo">{titulo}</h3>}
      {nota && <p className="muted tabela-nota">{nota}</p>}
      <div className="tabela-rolagem">
        <table className={`summary-table ${densidade(colunas.length)}`}>
          <thead>
            {agrupamento ? (
              <>
                <tr>
                  <th rowSpan={2} className={classe(colunas[0], 0)} aria-sort={ariaSort(colunas[0])}>
                    {cabecalho(colunas[0], colunas[0])}
                  </th>
                  {agrupamento.grupos.map((g) => (
                    <th key={g.nome} colSpan={g.n} className="grupo inicio-grupo">
                      {g.nome}
                    </th>
                  ))}
                </tr>
                <tr>
                  {colunas.slice(1).map((c, j) => (
                    <th key={c} className={classe(c, j + 1)} aria-sort={ariaSort(c)}>
                      {cabecalho(c, agrupamento.sub[j])}
                    </th>
                  ))}
                </tr>
              </>
            ) : (
              <tr>
                {colunas.map((c, i) => (
                  <th key={c} className={classe(c, i)} aria-sort={ariaSort(c)}>
                    {cabecalho(c, c)}
                  </th>
                ))}
              </tr>
            )}
          </thead>
          <tbody>
            {visiveis.map((linha, i) => (
              <tr key={i}>
                {colunas.map((c, j) => (
                  <td key={c} className={classe(c, j)}>
                    {formatar(c, linha[c])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
