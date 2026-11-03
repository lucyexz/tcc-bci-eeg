// Legenda simples em HTML, para gráficos em que a legenda do recharts não serve.

export default function LegendaItens({ itens }: { itens: { rotulo: string; cor: string; forma?: string; tracejado?: boolean }[] }) {
  return (
    <div className="legenda-itens">
      {itens.map((it) => (
        <span key={it.rotulo} className="legenda-item">
          <svg width="16" height="12" aria-hidden="true">
            {it.forma === "linha" || it.tracejado ? (
              <line x1="0" y1="6" x2="16" y2="6" stroke={it.cor} strokeWidth="2" strokeDasharray={it.tracejado ? "4 3" : undefined} />
            ) : it.forma === "square" ? (
              <rect x="3" y="1" width="10" height="10" fill={it.cor} />
            ) : it.forma === "vazado" ? (
              <circle cx="8" cy="6" r="4.5" fill="none" stroke={it.cor} strokeWidth="1.6" />
            ) : it.forma === "triangle" ? (
              <polygon points="8,0 2,11 14,11" fill={it.cor} />
            ) : (
              <circle cx="8" cy="6" r="5" fill={it.cor} />
            )}
          </svg>
          {it.rotulo}
        </span>
      ))}
    </div>
  );
}
