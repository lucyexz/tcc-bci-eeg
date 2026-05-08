import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell,
} from "recharts";
import { fetchComparison, fetchModel, type ComparisonModel } from "../api/client";

const MODEL_COLORS: Record<string, string> = {
  cnn: "#6c7bff",
  lstm: "#ff6b6b",
  gru: "#22c55e",
  transformer: "#f59e0b",
};

const CLASS_LABELS: Record<string, string> = {
  "0": "Esquerda",
  "1": "Direita",
  "2": "Neutro",
};

const USER_LABELS: Record<string, string> = {
  user_a: "A",
  user_b: "B",
  user_c: "C",
  user_d: "D",
  user_e: "E",
};

export default function Comparison() {
  const [models, setModels] = useState<ComparisonModel[]>([]);
  const [histories, setHistories] = useState<Record<string, number[]>>({});
  const [selectedModels, setSelectedModels] = useState<Set<string>>(new Set());
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchComparison()
      .then(async ({ models: ms }) => {
        setModels(ms);
        const results = await Promise.all(ms.map((m) => fetchModel(m.name)));
        const hist: Record<string, number[]> = {};
        results.forEach((r) => {
          hist[r.model_name] = r.history["val_accuracy"] ?? [];
        });
        setHistories(hist);
      })
      .catch((e) => setError(e.message));
  }, []);

  useEffect(() => {
    if (models.length) setSelectedModels(new Set(models.map((m) => m.name)));
  }, [models]);

  function toggleModel(name: string) {
    setSelectedModels((prev) => {
      const next = new Set(prev);
      next.has(name) ? next.delete(name) : next.add(name);
      return next;
    });
  }

  if (error) return <div className="error-msg">Erro: {error}</div>;
  if (!models.length) return <div className="loading">Carregando comparação…</div>;

  // Build overlapping val_accuracy chart data
  const maxEpochs = Math.max(...Object.values(histories).map((h) => h.length));
  const curveData = Array.from({ length: maxEpochs }, (_, i) => {
    const point: Record<string, number | null> = { epoch: i + 1 };
    Object.entries(histories).forEach(([name, hist]) => {
      point[name] = hist[i] ?? null;
    });
    return point;
  });

  // Per-user multi-model bar chart
  const users = Object.keys(models[0]?.per_user_accuracy ?? {});
  const userBarData = users.map((u) => {
    const point: Record<string, number | string> = { user: USER_LABELS[u] ?? u };
    models.forEach((m) => {
      point[m.name] = m.per_user_accuracy[u] ?? 0;
    });
    return point;
  });

  return (
    <div className="page">
      <div className="page-header">
        <Link to="/" className="back-link">← Voltar</Link>
        <h1>Comparação de Modelos</h1>
        <p className="muted">Val. accuracy por época + métricas por classe e por usuário</p>
      </div>

      {/* Overlapping val_accuracy curves */}
      <div className="section">
        <h2>Val. Accuracy por Época</h2>
        <div style={{ display: "flex", gap: 16, marginBottom: 12, flexWrap: "wrap" }}>
          {models.map((m) => (
            <label key={m.name} style={{ display: "flex", alignItems: "center", gap: 6, cursor: "pointer", userSelect: "none" }}>
              <input
                type="checkbox"
                checked={selectedModels.has(m.name)}
                onChange={() => toggleModel(m.name)}
                style={{ accentColor: MODEL_COLORS[m.name] ?? "#aaa" }}
              />
              <span style={{ color: MODEL_COLORS[m.name] ?? "#aaa", fontWeight: 500 }}>{m.display_name}</span>
            </label>
          ))}
        </div>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={curveData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
            <CartesianGrid stroke="#2d3348" strokeDasharray="3 3" />
            <XAxis dataKey="epoch" tick={{ fill: "#64748b", fontSize: 12 }} />
            <YAxis
              tick={{ fill: "#64748b", fontSize: 12 }}
              domain={[0, 1]}
              tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
            />
            <Tooltip
              formatter={(v) => typeof v === "number" ? `${(v * 100).toFixed(2)}%` : v}
              contentStyle={{ background: "#1e2130", border: "1px solid #2d3348", borderRadius: 6 }}
            />
            <Legend />
            {models.filter((m) => selectedModels.has(m.name)).map((m) => (
              <Line
                key={m.name}
                type="monotone"
                dataKey={m.name}
                name={m.display_name}
                stroke={MODEL_COLORS[m.name] ?? "#aaa"}
                dot={false}
                strokeWidth={2}
                connectNulls
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Comparison table */}
      <div className="section">
        <h2>Métricas por Classe</h2>
        <table className="summary-table">
          <thead>
            <tr>
              <th>Modelo</th>
              <th>Accuracy</th>
              <th>F1 Macro</th>
              {Object.keys(models[0].per_class_f1).map((k) => (
                <th key={k}>F1 {CLASS_LABELS[k] ?? k}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {models.map((m) => (
              <tr key={m.name}>
                <td>
                  <Link to={`/model/${m.name}`} className="table-link">{m.display_name}</Link>
                </td>
                <td>
                  <span className="accuracy-badge">{(m.test_accuracy * 100).toFixed(2)}%</span>
                </td>
                <td>{m.macro_f1 != null ? m.macro_f1.toFixed(3) : "—"}</td>
                {Object.values(m.per_class_f1).map((f, i) => (
                  <td key={i}>{f.toFixed(3)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Per-user multi-model bar chart */}
      <div className="section">
        <h2>Acurácia por Usuário</h2>
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={userBarData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
            <CartesianGrid stroke="#2d3348" strokeDasharray="3 3" />
            <XAxis dataKey="user" tick={{ fill: "#64748b", fontSize: 12 }} />
            <YAxis
              tick={{ fill: "#64748b", fontSize: 12 }}
              domain={[0, 1]}
              tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
            />
            <Tooltip
              formatter={(v) => typeof v === "number" ? `${(v * 100).toFixed(2)}%` : v}
              contentStyle={{ background: "#1e2130", border: "1px solid #2d3348", borderRadius: 6 }}
            />
            <Legend />
            {models.map((m) => (
              <Bar key={m.name} dataKey={m.name} name={m.display_name} fill={MODEL_COLORS[m.name] ?? "#aaa"} radius={[3, 3, 0, 0]}>
                {userBarData.map((_, i) => (
                  <Cell key={i} fill={MODEL_COLORS[m.name] ?? "#aaa"} />
                ))}
              </Bar>
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
