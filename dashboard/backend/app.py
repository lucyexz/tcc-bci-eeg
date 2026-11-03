"""API do dashboard: serve as tabelas (.csv), as figuras (.png) e os dados das figuras (graficos/*.json)
que as etapas gravam em resultados/.

Nada é recalculado aqui: o dashboard mostra exatamente os números e as figuras
gerados por `python main.py <etapa>`. Também serve o frontend compilado
(dashboard/frontend/dist/) e os dados do simulador no navegador
(src/simulacao/dados_web.py: predições gravadas no treino + grade EEG; nos
Transformers, a atenção do [CLS] é calculada com o modelo salvo).

Uso: python main.py dashboard
"""

import json
import sys
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ))

from src.config import MODELOS, RESULTADOS, USERS  # noqa: E402

DIST = RAIZ / "dashboard" / "frontend" / "dist"
NOMES = {"cnn": "CNN", "bilstm": "Bi-LSTM", "bigru": "Bi-GRU", "transformer": "Transformer",
         "cnn_transformer": "CNN + Transformer"}
ETAPAS = {
    "eda": "Análise exploratória",
    "divisao": "Divisão dos dados",
    **{m: NOMES[m] for m in MODELOS},
    "comparativo": "Comparação",
}
COMANDO = {**{e: e for e in ETAPAS}, "comparativo": "comparar"}

app = FastAPI(title="BCI EEG — dashboard", description=__doc__, version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["GET"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=10_000)  # os dados do simulador têm ~1 MB de JSON


@app.middleware("http")
async def sem_cache(request: Request, call_next):
    """A API e o index.html nunca ficam guardados no navegador: depois de rodar uma etapa ou
    reiniciar o dashboard, a página sempre recebe a versão atual. (Os assets do Vite têm hash no nome.)"""
    resposta = await call_next(request)
    if not request.url.path.startswith("/assets/"):
        resposta.headers["Cache-Control"] = "no-store" if request.url.path.startswith("/api/") else "no-cache"
    return resposta


def _pasta(etapa: str) -> Path:
    if etapa not in ETAPAS:
        raise HTTPException(404, f"Etapa desconhecida: {etapa}")
    pasta = RESULTADOS / etapa
    if not (pasta / "tabelas").exists() and not (pasta / "figuras").exists():
        raise HTTPException(404, f"Sem resultados de '{ETAPAS[etapa]}'. Rode: python main.py {COMANDO[etapa]}")
    return pasta


# Título da 1ª coluna quando o .csv a grava sem nome (índice do pandas); o resto é deduzido
TITULO_INDICE = {"estatisticas_descritivas": "Feature", "qualidade": "Verificação", "silhueta": "Padronização"}


def _titulo_indice(caminho: Path, valores: pd.Series) -> str:
    if valores.astype(str).str.startswith("user_").all():
        return "Usuário"
    return TITULO_INDICE.get(caminho.stem, "")


def _por_classe(df: pd.DataFrame) -> pd.DataFrame:
    """Métricas por classe como na monografia: sensibilidade e precisão das três classes.

    O F1 e a linha Macro ficam no .csv, mas não são exibidos. A sensibilidade vem antes
    da precisão, na ordem em que o texto as cita.
    """
    df = df[df["Classe"].astype(str) != "Macro"]
    chaves = [c for c in df.columns if not c.startswith(("Precisão", "Sensibilidade", "F1"))]
    metricas = [c for c in df.columns if c.startswith("Sensibilidade")] + \
               [c for c in df.columns if c.startswith("Precisão")]
    return df[chaves + metricas]


def _ler_csv(caminho: Path) -> list[dict]:
    df = pd.read_csv(caminho, sep=";", decimal=",", encoding="utf-8-sig")
    # Só na exibição (os .csv não mudam): índice sem nome ganha título; participante → Usuário
    df.columns = [_titulo_indice(caminho, df[c]) if str(c).startswith("Unnamed")
                  else "Usuário" if str(c).lower() == "participante" else c for c in df.columns]
    if "Modelo" in df.columns:  # o baseline linear fica nos .csv, mas não é exibido
        df = df[~df["Modelo"].astype(str).str.endswith("(baseline)")]
    if caminho.stem == "metricas_por_classe":
        df = _por_classe(df)
    # NaN não existe em JSON
    return df.astype(object).where(df.notna(), None).to_dict(orient="records")


@app.get("/api/etapas", summary="Etapas com resultados disponíveis")
def etapas() -> list[dict]:
    return [{"etapa": e, "nome": nome, "comando": f"python main.py {COMANDO[e]}",
             "disponivel": (RESULTADOS / e / "tabelas").exists() or (RESULTADOS / e / "figuras").exists()}
            for e, nome in ETAPAS.items()]


@app.get("/api/simulador/{modelo}", summary="Dados do simulador: teste do hold-out de um participante")
def simulador(modelo: str, paradigma: str = "especialista", participante: str = "user_a") -> dict:
    """As mesmas opções do `python main.py simulador` (semente 42). Declarada antes de /api/{etapa}."""
    from src.simulacao.dados_web import TRACK, SemResultados, dados_simulacao

    if modelo not in MODELOS or paradigma not in TRACK or participante not in USERS:
        raise HTTPException(400, f"Opção inválida: modelo={modelo}, paradigma={paradigma}, "
                                 f"participante={participante}")
    try:
        return dados_simulacao(modelo, paradigma, participante)
    except SemResultados as e:
        raise HTTPException(404, str(e)) from e


@app.get("/api/{etapa}", summary="Tabelas e figuras de uma etapa")
def etapa(etapa: str) -> dict:
    pasta = _pasta(etapa)
    tabelas = {f.stem: _ler_csv(f) for f in sorted((pasta / "tabelas").glob("*.csv"))}
    figuras = [{"nome": f.stem, "url": f"/figuras/{etapa}/{f.name}"}
               for f in sorted((pasta / "figuras").glob("*.png"))]
    # Dados de cada figura (gravados pela etapa junto do PNG): o frontend desenha o gráfico interativo
    graficos = {f.stem: json.loads(f.read_text(encoding="utf-8")) for f in sorted((pasta / "graficos").glob("*.json"))}
    return {"etapa": etapa, "nome": ETAPAS[etapa], "tabelas": tabelas, "figuras": figuras, "graficos": graficos}


@app.get("/figuras/{etapa}/{arquivo}", include_in_schema=False)
def figura(etapa: str, arquivo: str) -> FileResponse:
    caminho = _pasta(etapa) / "figuras" / Path(arquivo).name
    if caminho.suffix != ".png" or not caminho.is_file():
        raise HTTPException(404, "Figura não encontrada")
    return FileResponse(caminho)


@app.get("/{caminho:path}", include_in_schema=False)
def frontend(caminho: str) -> FileResponse:
    """Arquivos do frontend compilado; qualquer outra rota devolve o index.html (rotas do React)."""
    if caminho.startswith("api/"):  # rota da API inexistente: erro em JSON, nunca o index.html
        raise HTTPException(404, f"Rota desconhecida: /{caminho}")
    if not DIST.exists():
        raise HTTPException(404, "Frontend não compilado. Rode: python main.py dashboard")
    arquivo = (DIST / caminho).resolve()
    if caminho and arquivo.is_file() and DIST.resolve() in arquivo.parents:
        return FileResponse(arquivo)
    return FileResponse(DIST / "index.html")
