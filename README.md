# BCI EEG — Classificação de Sinais EEG com Deep Learning

Projeto de TCC que compara cinco modelos de deep learning (CNN, Bi-LSTM, Bi-GRU, Transformer e CNN + Transformer) para classificar a intenção motora (Esquerda, Direita, Neutro) a partir da potência espectral de 5 eletrodos EEG (AF3, T7, Pz, T8, AF4).

---

## Requisitos

| Ferramenta | Versão mínima |
|---|---|
| Python | 3.10 a 3.12 (testado com 3.11) |
| Node.js | 18+ (só para compilar o dashboard na primeira vez; ver seção 7) |
| Git | qualquer |

---

## 1. Clonar o repositório

    git clone https://github.com/tiagotlg/tcc-bci-eeg.git
    cd tcc-bci-eeg

---

## 2. Criar e ativar o ambiente virtual

**Windows (PowerShell)**

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1

**Linux / macOS**

    python -m venv .venv
    source .venv/bin/activate

O prompt do terminal muda para `(.venv)` confirmando a ativação.

---

## 3. Instalar dependências Python

    pip install -r requirements.txt

> Inclui TensorFlow, NumPy, pandas, scikit-learn, scipy, matplotlib, pygame (simulação) e FastAPI + uvicorn (dashboard). O TensorFlow é um download grande: instale com internet antes de precisar.

---

## 4. Download do dataset

1. Acesse o dataset no Kaggle: [Brain Wave Data from Hands Movement of EEG](https://www.kaggle.com/datasets/fabriciotorquato/brain-wave-data-from-hands-movement-of-eeg)
2. Faça o download e extraia os arquivos CSV
3. Coloque os arquivos no diretório `data/raw/` com os nomes exatos:

```
data/raw/
├── user_a.csv
├── user_b.csv
├── user_c.csv
├── user_d.csv
└── user_e.csv
```

---

## 5. Estrutura do projeto

```
main.py                ponto de entrada: python main.py <etapa>
src/
├── constantes.py      colunas, eletrodos, bandas e classes do dataset
├── config.py          configuração final: protocolos, sementes, janela (T) e dropout de cada modelo
├── dados/             carregamento, pré-processamento (log10, divisão por blocos, normalização),
│                      janelas causais e checagens anti-vazamento
├── modelos/           CNN, Bi-LSTM/Bi-GRU, Transformer, CNN + Transformer e o laço de treino
├── avaliacao/         leitura dos resultados, figuras e explicabilidade (Integrated Gradients, atenção)
├── etapas/            o que o main.py executa: eda, divisao, baseline, redes, comparar
└── simulacao/         simulador em Pygame e os dados do simulador do dashboard
dashboard/
├── backend/app.py     API (FastAPI): serve as tabelas, figuras e dados de resultados/
└── frontend/          site (React + Vite); dist/ é a versão compilada (não versionada)
data/raw/              dataset (não versionado)
resultados/            gerado ao rodar (não versionado)
```

---

## 6. Executar

Nenhum resultado vem pronto no repositório: tudo é gerado do zero a partir de `data/raw/`, em `resultados/`.

Para rodar tudo, na ordem certa (mostra a estimativa de tempo antes; ~3,5 h em CPU, quase todo no treino da Bi-LSTM e da Bi-GRU):

    python main.py tudo
    python main.py tudo --estimar   # só a estimativa

Ou uma etapa de cada vez (`python main.py --help` lista todas):

Comando | O que faz | Saída em `resultados/`
--- | --- | ---
`python main.py eda` | Análise exploratória do dataset (só dados) | `eda/`
`python main.py divisao` | Divisão por blocos, purge, janelas causais e checagens anti-vazamento | `divisao/`
`python main.py baseline` | LDA e regressão logística na média da janela (~1,5 min) | `baseline/`
`python main.py cnn` | CNN | `cnn/`
`python main.py bilstm` | Bi-LSTM | `bilstm/`
`python main.py bigru` | Bi-GRU | `bigru/`
`python main.py transformer` | Transformer | `transformer/`
`python main.py cnn_transformer` | CNN + Transformer (EEG Conformer adaptado) | `cnn_transformer/`
`python main.py comparar` | Tabelas e figuras comparando todos os modelos | `comparativo/`

Cada etapa mostra as tabelas no terminal e grava em `tabelas/` (`.csv`, separador `;`, abre direto no Excel), as figuras em `figuras/` e os dados de cada figura em `graficos/` (`.json`, usados pelos gráficos interativos do dashboard). Cada modelo treina o que ainda não existe (hold-out por blocos com 3 sementes, validação cruzada por blocos com 6 folds e controle de permutação de rótulos) e depois gera tabelas e figuras (resultado, matriz de confusão, curvas de treino, permutação, calibração e explicabilidade). Treinos já gravados são pulados, então dá para interromper e continuar depois. Opções úteis:

    python main.py cnn --estimar       # tempo previsto
    python main.py cnn --so-analisar   # refaz só a análise
    python main.py cnn --smoke         # teste rápido do código (2 épocas)
    python main.py cnn --help          # todas as opções

O treino é determinístico (sementes fixas, 4 threads, operações determinísticas do TensorFlow): rodar de novo dá os mesmos números. A configuração final fica em `src/config.py`; as arquiteturas, em `src/modelos/`.

---

## 7. Dashboard

Mostra os resultados numa página web, com a navegação numa barra lateral (em tela estreita ela vai para o topo) e tema claro/escuro. Rode depois das etapas (ou do `python main.py tudo`):

    python main.py dashboard

Abre `http://127.0.0.1:8000` no navegador. Opções: `--porta 8080`, `--sem-navegador`, `--recompilar`.

Na primeira vez, o comando compila o site em `dashboard/frontend/dist/` (roda `npm install` e `npm run build`, então precisa do Node.js 18+ e de internet). Depois disso não precisa mais do Node. Use `--recompilar` depois de mudar o código do frontend (ou de dar `git pull` com mudanças nele).

Páginas:
- **Visão geral:** os cinco modelos lado a lado: resultado principal, sensibilidade e precisão por classe, controle de permutação, acurácia de cada participante, curvas de treino e calibração.
- **Uma página por modelo:** no topo, o **simulador** (seção 8); depois configuração, resultado (com matriz de confusão), treino e controles (curvas, permutação, calibração) e explicabilidade (Integrated Gradients e, nos Transformers, atenção do [CLS]).
- **Dados:** a análise exploratória, em seções na ordem do `python main.py eda`, e a divisão dos registros (hold-out e cada fold da validação cruzada). `/dados?aba=divisao` abre direto na divisão.

O dashboard não recalcula nada: lê os `.csv`, as figuras e os dados das figuras (`graficos/*.json`) gravados pelas etapas. As figuras aparecem como gráficos interativos (passe o mouse para ver os valores), com um link "ver figura da monografia" que abre o PNG; resultados gerados antes de existir `graficos/` aparecem como PNG. Se uma etapa ainda não rodou, a página diz qual comando falta.

O que o dashboard lê:
- `resultados/` — tudo o que aparece nas páginas;
- `data/raw/` — só o simulador, para montar a grade EEG de cada janela.

Para mexer no frontend com recarregamento automático: deixe `python main.py dashboard --sem-navegador` rodando e, em outro terminal, `cd dashboard/frontend`, `npm install` e `npm run dev` (abre em `http://localhost:5173`).

---

## 8. Simulador

Um modelo treinado dirige um carro 2D com o EEG de teste de um participante. Os dados são o teste do hold-out por blocos (o fim da sessão, que o modelo nunca viu), tocado na ordem em que foi gravado, com o modelo da semente 42 desse hold-out. As predições são as mesmas gravadas no treino. Há duas versões, com a mesma tela:

- **No dashboard (navegador):** no topo da página de cada modelo. Escolha paradigma, participante, velocidade (janelas/s), FPS e "sem modelo", e clique em **Iniciar**; só então os dados são carregados (nos Transformers, a primeira vez leva alguns segundos porque carrega o TensorFlow para calcular a atenção). Teclas `SPACE`, `F` e `R` depois de clicar na simulação; a animação para quando sai da tela.
- **Em Pygame (janela 1280×720):** pelo terminal, como abaixo.

Exemplos da versão Pygame:

    python main.py simulador                                          # CNN, especialista, user_a
    python main.py simulador --modelo transformer --participante user_c
    python main.py simulador --modelo bigru --paradigma generalista --participante user_e
    python main.py simulador --velocidade 16                          # 2× mais rápido que o tempo real
    python main.py simulador --mock                                   # sem modelo: predições sorteadas

Opção | Valores | Padrão
--- | --- | ---
`--modelo` | `cnn`, `bilstm`, `bigru`, `transformer`, `cnn_transformer` | `cnn`
`--paradigma` | `especialista` (modelo do participante), `generalista` (modelo único) | `especialista`
`--participante` | `user_a` … `user_e` | `user_a`
`--velocidade` | janelas por segundo (8 = o ritmo da gravação) | `8`

Na tela:
- **Painel da direita:** classe prevista e probabilidades, a **classe verdadeira** daquele momento (acertou/errou), o **acerto acumulado** (proporção simples de acertos desde o início do trecho) e o histórico das últimas predições, com a classe verdadeira numa faixa embaixo.
- **Painel da esquerda:** o registro mais recente da janela (log da potência padronizado, eletrodo × banda) e, no Transformer e no CNN + Transformer, a atenção do token [CLS] ao longo da janela.

O modelo precisa ter sido treinado antes (`python main.py <modelo>`), e as duas versões precisam de `data/raw/`.

### Controles (Pygame)

Tecla | Ação
--- | ---
`SPACE` | Pausar / retomar (congela tudo)
`F` | Congelar dados EEG — o veículo continua se movendo com o último comando; permite analisar por que ele fez o que fez
`R` | Resetar veículo ao centro
`ESC` | Sair

---

## Resultados

Os resultados são gerados em `resultados/` (ver seção 6). A tabela geral fica em `resultados/comparativo/tabelas/resultado_principal.csv`.
