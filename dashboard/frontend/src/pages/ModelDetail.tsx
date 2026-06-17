import { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { fetchModel, figureUrl, type ModelDetail as ModelDetailType } from "../api/client";
import LearningCurves from "../components/LearningCurves";
import ConfusionMatrix from "../components/ConfusionMatrix";
import ClassReport from "../components/ClassReport";
import PerUserChart from "../components/PerUserChart";

export default function ModelDetail() {
  const { name } = useParams<{ name: string }>();
  const [model, setModel] = useState<ModelDetailType | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!name) return;
    fetchModel(name)
      .then(setModel)
      .catch((e) => setError(e.message));
  }, [name]);

  if (error) return <div className="error-msg">Erro: {error}</div>;
  if (!model) return <div className="loading">Carregando modelo…</div>;

  const cfg = model.config;

  return (
    <div className="page">
      <div className="page-header">
        <Link to="/" className="back-link">← Voltar</Link>
        <div className="model-title-row">
          <h1>{model.display_name}</h1>
          <span className="accuracy-badge accuracy-badge-lg">
            {(model.test_accuracy * 100).toFixed(2)}%
          </span>
        </div>
        <p className="muted">
          Test Loss: {model.test_loss.toFixed(4)} &nbsp;·&nbsp;
          Treinado em: {new Date(model.trained_at).toLocaleString("pt-BR", { dateStyle: "short", timeStyle: "short" })}
        </p>
      </div>

      {/* Config */}
      <div className="section">
        <h2>Configuração</h2>
        <div className="config-grid">
          {Object.entries(cfg).map(([k, v]) => (
            <div key={k} className="config-item">
              <span className="config-key">{k}</span>
              <span className="config-val">{String(v)}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Learning Curves */}
      <div className="section">
        <h2>Curvas de Aprendizado</h2>
        <LearningCurves history={model.history} />
      </div>

      {/* Confusion Matrix + Class Report */}
      <div className="section">
        <h2>Avaliação</h2>
        <div className="eval-grid">
          <ConfusionMatrix matrix={model.confusion_matrix} />
          <ClassReport report={model.classification_report} />
        </div>
      </div>

      {/* Per-user */}
      <div className="section">
        <h2>Acurácia por Usuário</h2>
        <PerUserChart perUser={model.per_user_accuracy} />
      </div>

      {/* Transformer attention weights */}
      {name === "transformer" && (
        <div className="section">
          <h2>Pesos de Atenção</h2>
          <img
            src={figureUrl("transformer_attention_weights.png")}
            alt="Transformer attention weights"
            className="figure-img"
          />
        </div>
      )}
    </div>
  );
}
