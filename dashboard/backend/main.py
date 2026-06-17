from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from routers.models import router as models_router

app = FastAPI(
    title="BCI EEG Dashboard API",
    description=(
        "API de métricas dos modelos CNN, Bi-LSTM, Bi-GRU e Transformer "
        "treinados para classificação de sinais EEG motores (Esquerda / Direita / Neutro). "
        "Dataset: EMOTIV Insight+ 5 canais, 5 sujeitos, ~85k amostras."
    ),
    version="1.0.0",
    contact={"name": "Tiago Taligon", "email": "tiagotaligon@gmail.com"},
    openapi_tags=[
        {
            "name": "models",
            "description": "Métricas e comparação dos modelos treinados.",
        }
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

FIGURES_DIR = Path(__file__).parent.parent.parent / "outputs" / "figures"
app.mount("/figures", StaticFiles(directory=str(FIGURES_DIR)), name="figures")

app.include_router(models_router)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
