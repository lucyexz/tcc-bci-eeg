import type { ClassificationReport } from "../api/client";

const CLASS_NAMES: Record<string, string> = {
  "0": "Esquerda",
  "1": "Direita",
  "2": "Neutro",
};

interface Props {
  report: ClassificationReport;
}

export default function ClassReport({ report }: Props) {
  const classKeys = Object.keys(report).filter(
    (k) => !["accuracy", "macro avg", "weighted avg"].includes(k)
  );
  const summaryKeys = ["macro avg", "weighted avg"].filter((k) => k in report);

  const fmt = (v: unknown) =>
    typeof v === "number" ? v.toFixed(3) : "—";

  const row = (key: string, label: string, highlight = false) => {
    const metrics = report[key] as { precision: number; recall: number; "f1-score": number; support: number } | undefined;
    if (!metrics) return null;
    return (
      <tr key={key} className={highlight ? "report-summary-row" : ""}>
        <td>{label}</td>
        <td>{fmt(metrics.precision)}</td>
        <td>{fmt(metrics.recall)}</td>
        <td>{fmt(metrics["f1-score"])}</td>
        <td>{metrics.support ?? "—"}</td>
      </tr>
    );
  };

  return (
    <div className="class-report-wrap">
      <h4 className="chart-title">Classification Report</h4>
      <table className="report-table">
        <thead>
          <tr>
            <th>Classe</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1</th>
            <th>Support</th>
          </tr>
        </thead>
        <tbody>
          {classKeys.map((k) => row(k, CLASS_NAMES[k] ?? k))}
          <tr className="report-divider">
            <td colSpan={5} />
          </tr>
          {summaryKeys.map((k) => row(k, k, true))}
        </tbody>
      </table>
    </div>
  );
}
