const CLASS_NAMES = ["Esquerda", "Direita", "Neutro"];

interface Props {
  matrix: number[][];
}

export default function ConfusionMatrix({ matrix }: Props) {
  const rowMaxes = matrix.map((row) => Math.max(...row));

  return (
    <div className="confusion-matrix-wrap">
      <h4 className="chart-title">Confusion Matrix</h4>
      <div className="cm-grid">
        {/* header row */}
        <div className="cm-cell cm-header-corner" />
        {CLASS_NAMES.map((name) => (
          <div key={name} className="cm-cell cm-header">{name}</div>
        ))}
        {/* data rows */}
        {matrix.map((row, i) => (
          <>
            <div key={`row-${i}`} className="cm-cell cm-row-label">{CLASS_NAMES[i]}</div>
            {row.map((val, j) => {
              const opacity = rowMaxes[i] > 0 ? val / rowMaxes[i] : 0;
              const isDiag = i === j;
              return (
                <div
                  key={`${i}-${j}`}
                  className={`cm-cell cm-data ${isDiag ? "cm-diag" : ""}`}
                  style={{ background: `rgba(108, 123, 255, ${opacity * 0.85 + 0.05})` }}
                >
                  <span className="cm-value">{val}</span>
                </div>
              );
            })}
          </>
        ))}
      </div>
      <div className="cm-legend">
        <span className="muted">Predito →</span>
        <span className="muted" style={{ marginLeft: "auto" }}>Real ↓</span>
      </div>
    </div>
  );
}
