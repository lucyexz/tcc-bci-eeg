# BCI EEG — Classificação de Sinais EEG com Deep Learning

Projeto de TCC que compara modelos de deep learning (CNN, Bi-LSTM, Bi-GRU e Transformer) para classificação de intenção motora a partir de sinais EEG de 5 eletrodos (AF3, T7, Pz, T8, AF4).

---

## Requisitos

| Ferramenta | Versão mínima |
|---|---|
| Python | 3.10+ |
| Node.js | 18+ |
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

> Inclui TensorFlow, NumPy, pandas, scikit-learn, scipy, matplotlib, seaborn, JupyterLab e pygame (necessário para a simulação).

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

## 5. Executar os notebooks

Inicie o Jupyter:

    jupyter lab
    # ou
    jupyter notebook

Execute os notebooks **na ordem abaixo** — cada um depende do anterior:

Notebook | O que faz | Outputs gerados
--- | --- | ---
`01_analise_dataset.ipynb` | Análise exploratória (distribuições, PCA, t-SNE, correlação) | Figuras em `outputs/figures/`
`02_preprocessamento_base.ipynb` | Pipeline de preprocessing (IQR clip, StandardScaler, split 80/20) | `.npy` em `data/processed/`
`03_cnn.ipynb` | Treina CNN — learning curves, confusion matrix, classification report | `outputs/models/cnn_model.keras`, `outputs/metrics/cnn.json`
`04_lstm.ipynb` | Treina Bi-LSTM e Bi-GRU | `outputs/models/lstm_model.keras`, `gru_model.keras`, `outputs/metrics/lstm.json`, `gru.json`
`05_transformer.ipynb` | Treina Transformer — inclui visualização de pesos de atenção | `outputs/models/transformer_model.keras`, `outputs/metrics/transformer.json`

> O notebook `02` deve ser rodado antes de qualquer notebook de modelo — ele gera os arrays processados que os modelos consomem.

---

## 6. Dashboard

O dashboard visualiza as métricas de todos os modelos treinados.

### Backend (FastAPI)

Certifique-se de que o ambiente virtual está ativo, depois:

    cd dashboard\backend
    pip install -r requirements.txt
    python main.py

O backend sobe em:
- `http://localhost:8000/docs` — Swagger UI com todos os endpoints documentados
- `http://localhost:8000/api/models` — lista os modelos em JSON

### Frontend (React + Vite)

Em outro terminal (não precisa do venv):

    cd dashboard\frontend
    npm install
    npm run dev

Acesse: `http://localhost:5173`

---

## 7. Simulação Pygame

Demonstração interativa onde um modelo BCI controla um veículo 2D em tempo real (janela 1280×720).

### Modo mock (sem modelo, abre imediatamente)

    python -m src.simulation.pygame_simulator

### Modo real (requer `outputs/models/` e `data/processed/`)

    python -m src.simulation.pygame_simulator --model cnn --no-mock
    python -m src.simulation.pygame_simulator --model lstm --no-mock
    python -m src.simulation.pygame_simulator --model gru --no-mock
    python -m src.simulation.pygame_simulator --model transformer --no-mock

Use `--data-path <caminho>` para apontar para um diretório `data/processed/` alternativo.

### Controles

Tecla | Ação
--- | ---
`SPACE` | Pausar / retomar (congela tudo)
`F` | Congelar dados EEG — o veículo continua se movendo com o último comando; permite analisar por que ele fez o que fez
`R` | Resetar veículo ao centro
`ESC` | Sair

---

## Resultados

Modelo | Test Accuracy | Test Loss
--- | --- | ---
CNN | **72.21%** | 0.6253
Bi-GRU | 69.58% | 0.6702
Bi-LSTM | 67.32% | 0.7024
Transformer | 66.46% | 0.7195
