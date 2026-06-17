import { useNavigate } from "react-router-dom";
import type { ModelSummary } from "../api/client";

interface Props {
  model: ModelSummary;
}

export default function MetricCard({ model }: Props) {
  const navigate = useNavigate();
  const pct = (model.test_accuracy * 100).toFixed(2);
  const date = model.trained_at
    ? new Date(model.trained_at).toLocaleString("pt-BR", { dateStyle: "short", timeStyle: "short" })
    : "—";

  return (
    <div className="card card-clickable" onClick={() => navigate(`/model/${model.name}`)}>
      <div className="card-header">
        <span className="model-name">{model.display_name}</span>
        <span className="accuracy-badge">{pct}%</span>
      </div>
      <div className="card-body">
        <div className="metric-row">
          <span className="metric-label">Test Accuracy</span>
          <span className="metric-value">{pct}%</span>
        </div>
        {model.test_loss != null && (
          <div className="metric-row">
            <span className="metric-label">Test Loss</span>
            <span className="metric-value">{model.test_loss.toFixed(4)}</span>
          </div>
        )}
        <div className="metric-row">
          <span className="metric-label">Treinado em</span>
          <span className="metric-value muted">{date}</span>
        </div>
      </div>
      <div className="card-footer">Ver detalhes →</div>
    </div>
  );
}
