"""Classificação de intenção motora em EEG com deep learning: ponto de entrada do projeto.

Uso:
    python main.py <etapa> [opções]
    python main.py <etapa> --help      opções de cada etapa

Etapas, na ordem da metodologia:
    eda               análise exploratória do dataset (só dados)
    divisao           divisão por blocos, purge, janelas causais e checagens anti-vazamento
    baseline          LDA e regressão logística na média da janela (~1,5 min)
    cnn               CNN
    bilstm            Bi-LSTM
    bigru             Bi-GRU
    transformer       Transformer
    cnn_transformer   CNN + Transformer (EEG Conformer adaptado)
    comparar          tabelas e figuras comparando os cinco modelos e o baseline
    tudo              todas as etapas acima, em ordem (~3,5 h em CPU); "tudo --estimar" só estima o tempo

Apresentação (depois das etapas):
    dashboard         abre o dashboard com os resultados em http://127.0.0.1:8000
    simulador         simulação em Pygame: um modelo treinado dirige o carro com o EEG de teste

Tudo é gerado em resultados/. Treinos já gravados são pulados, então dá para
interromper e continuar depois.
"""

import argparse
import runpy
import shutil
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ))

REDES = ["cnn", "bilstm", "bigru", "transformer", "cnn_transformer"]
ETAPAS = ["eda", "divisao", "baseline", *REDES, "comparar"]
# Ordem do "tudo": as redes mais rápidas primeiro
ORDEM_TUDO = ["eda", "divisao", "baseline", "cnn", "cnn_transformer", "transformer", "bilstm", "bigru", "comparar"]


def tudo(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="python main.py tudo",
                                     description="Roda todas as etapas em ordem; para no primeiro erro.")
    parser.add_argument("--estimar", action="store_true", help="só mostra a estimativa de tempo")
    args = parser.parse_args(argv)

    from src.config import PROTOCOLOS
    from src.modelos.treino import estimar, formatar_duracao, raiz_modelo

    total = 0.0
    for modelo in REDES:
        print(f"\n{modelo}")
        total += estimar(modelo, PROTOCOLOS, raiz_modelo(modelo))
    print(f"\nTreino das redes: ~{formatar_duracao(total)} (mais alguns minutos de EDA, baseline e análises)")
    if args.estimar:
        return

    # Cada etapa num processo próprio, como se fosse rodada sozinha (mesmo estado inicial e mesmo resultado).
    inicio = time.perf_counter()
    for etapa in ORDEM_TUDO:
        print(f"\n{'#' * 78}\n# {etapa}\n{'#' * 78}", flush=True)
        t0 = time.perf_counter()
        codigo = subprocess.call([sys.executable, str(RAIZ / "main.py"), etapa], cwd=RAIZ)
        if codigo != 0:
            print(f"\n{etapa} terminou com erro (código {codigo}). Parando.")
            sys.exit(codigo)
        print(f"\n{etapa} concluída em {formatar_duracao(time.perf_counter() - t0)}", flush=True)
    print(f"\nTudo concluído em {formatar_duracao(time.perf_counter() - inicio)}. Resultados em resultados/")


def dashboard(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="python main.py dashboard",
                                     description="Compila o frontend (só na primeira vez) e abre o dashboard.")
    parser.add_argument("--porta", type=int, default=8000)
    parser.add_argument("--sem-navegador", action="store_true", help="não abre o navegador")
    parser.add_argument("--recompilar", action="store_true", help="compila o frontend de novo")
    args = parser.parse_args(argv)

    frontend = RAIZ / "dashboard" / "frontend"
    if args.recompilar or not (frontend / "dist" / "index.html").exists():
        npm = shutil.which("npm")
        if npm is None:
            sys.exit("O frontend ainda não foi compilado e o Node.js não foi encontrado. "
                     "Instale o Node 18+ (https://nodejs.org) e rode de novo.")
        if not (frontend / "node_modules").exists():
            print("Instalando as dependências do frontend (npm install)...", flush=True)
            subprocess.run([npm, "install"], cwd=frontend, check=True)
        print("Compilando o frontend (npm run build)...", flush=True)
        subprocess.run([npm, "run", "build"], cwd=frontend, check=True)

    import uvicorn

    url = f"http://127.0.0.1:{args.porta}"
    print(f"\nDashboard em {url}  (Ctrl+C para encerrar)\n", flush=True)
    if not args.sem_navegador:
        webbrowser.open(url)
    uvicorn.run("app:app", app_dir=str(RAIZ / "dashboard" / "backend"), host="127.0.0.1", port=args.porta,
                log_level="warning")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return
    etapa, argv = sys.argv[1], sys.argv[2:]
    if etapa == "tudo":
        tudo(argv)
    elif etapa == "dashboard":
        dashboard(argv)
    elif etapa == "simulador":
        from src.simulacao.pygame_simulator import main as simulador
        simulador(argv)
    elif etapa == "eda":
        runpy.run_module("src.etapas.eda", run_name="__main__")
    elif etapa in REDES:
        from src.etapas import redes
        redes.main(etapa, argv)
    elif etapa in ETAPAS:
        import importlib
        importlib.import_module(f"src.etapas.{etapa}").main(argv)
    else:
        print(f"Etapa desconhecida: {etapa}\n{__doc__}")
        sys.exit(2)


if __name__ == "__main__":
    main()
