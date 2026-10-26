import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { fetchModels, type ModelSummary } from "../api/client";
import MetricCard from "../components/MetricCard";

export default function Overview() {
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchModels()
      .then(setModels)
      .catch((e) => setError(e.message));
  }, []);

  if (error) return <div className="error-msg">Erro: {error}</div>;
  if (!models.length) return <div className="loading">Carregando modelos…</div>;

  return (
    <div className="page">
      <div className="page-header">
        <h1>BCI EEG Dashboard</h1>
        <p className="muted">Comparação de modelos de classificação de sinais EEG — 3 classes (Esquerda, Direita, Neutro)</p>
      </div>

      <div className="cards-grid">
        {models.map((m) => (
          <MetricCard key={m.name} model={m} />
        ))}
      </div>

      <div className="section">
        <div className="section-header">
          <h2>Resumo Comparativo</h2>
          <Link to="/comparison" className="link-btn">Ver comparação completa →</Link>
        </div>
        <table className="summary-table">
          <thead>
            <tr>
              <th>Modelo</th>
              <th>Test Accuracy</th>
              <th>Test Loss</th>
              <th>Treinado em</th>
            </tr>
          </thead>
          <tbody>
            {models.map((m) => (
              <tr key={m.name}>
                <td>
                  <Link to={`/model/${m.name}`} className="table-link">{m.display_name}</Link>
                </td>
                <td><span className="accuracy-badge">{(m.test_accuracy * 100).toFixed(2)}%</span></td>
                <td>{m.test_loss != null ? m.test_loss.toFixed(4) : "—"}</td>
                <td className="muted">
                  {m.trained_at
                    ? new Date(m.trained_at).toLocaleString("pt-BR", { dateStyle: "short", timeStyle: "short" })
                    : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
