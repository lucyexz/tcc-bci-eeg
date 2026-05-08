import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

interface Props {
  history: Record<string, number[]>;
}

export default function LearningCurves({ history }: Props) {
  const epochs = history["accuracy"]?.length ?? 0;
  const data = Array.from({ length: epochs }, (_, i) => ({
    epoch: i + 1,
    accuracy: history["accuracy"]?.[i],
    val_accuracy: history["val_accuracy"]?.[i],
    loss: history["loss"]?.[i],
    val_loss: history["val_loss"]?.[i],
  }));

  const chartProps = {
    data,
    margin: { top: 5, right: 20, left: 0, bottom: 5 },
  };

  const axisStyle = { fill: "#64748b", fontSize: 12 };
  const gridStyle = { stroke: "#2d3348" };

  return (
    <div className="learning-curves">
      <div className="chart-wrap">
        <h4 className="chart-title">Accuracy</h4>
        <ResponsiveContainer width="100%" height={220}>
          <LineChart {...chartProps}>
            <CartesianGrid {...gridStyle} strokeDasharray="3 3" />
            <XAxis dataKey="epoch" tick={axisStyle} label={{ value: "Época", position: "insideBottom", offset: -2, fill: "#64748b", fontSize: 12 }} />
            <YAxis tick={axisStyle} domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
            <Tooltip formatter={(v) => typeof v === "number" ? `${(v * 100).toFixed(2)}%` : v} contentStyle={{ background: "#1e2130", border: "1px solid #2d3348", borderRadius: 6 }} />
            <Legend />
            <Line type="monotone" dataKey="accuracy" name="Train" stroke="#6c7bff" dot={false} strokeWidth={2} />
            <Line type="monotone" dataKey="val_accuracy" name="Validação" stroke="#ff6b6b" dot={false} strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="chart-wrap">
        <h4 className="chart-title">Loss</h4>
        <ResponsiveContainer width="100%" height={220}>
          <LineChart {...chartProps}>
            <CartesianGrid {...gridStyle} strokeDasharray="3 3" />
            <XAxis dataKey="epoch" tick={axisStyle} label={{ value: "Época", position: "insideBottom", offset: -2, fill: "#64748b", fontSize: 12 }} />
            <YAxis tick={axisStyle} tickFormatter={(v) => v.toFixed(2)} />
            <Tooltip formatter={(v) => typeof v === "number" ? v.toFixed(4) : v} contentStyle={{ background: "#1e2130", border: "1px solid #2d3348", borderRadius: 6 }} />
            <Legend />
            <Line type="monotone" dataKey="loss" name="Train" stroke="#6c7bff" dot={false} strokeWidth={2} />
            <Line type="monotone" dataKey="val_loss" name="Validação" stroke="#ff6b6b" dot={false} strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
