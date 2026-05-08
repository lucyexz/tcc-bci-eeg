import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

interface Props {
  perUser: Record<string, number>;
  color?: string;
}

const USER_LABELS: Record<string, string> = {
  user_a: "A",
  user_b: "B",
  user_c: "C",
  user_d: "D",
  user_e: "E",
};

export default function PerUserChart({ perUser, color = "#6c7bff" }: Props) {
  const data = Object.entries(perUser).map(([key, val]) => ({
    user: USER_LABELS[key] ?? key,
    accuracy: val,
  }));

  return (
    <div className="chart-wrap">
      <h4 className="chart-title">Acurácia por Usuário</h4>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
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
            labelStyle={{ color: "#ffffff" }}
            itemStyle={{ color: "#ffffff" }}
          />
          <Bar dataKey="accuracy" radius={[4, 4, 0, 0]}>
            {data.map((_, i) => (
              <Cell key={i} fill={color} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
