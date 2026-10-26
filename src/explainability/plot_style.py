"""Estilo único das figuras da v2 (matplotlib, PNG para a monografia).

Paleta categórica validada (ordem fixa, nunca reciclada): azul, laranja,
água, amarelo, magenta. As cores seguem a entidade: um modelo, sujeito ou
classe tem sempre a mesma cor em todas as figuras.

- Modelos  → slots 1–4 (CNN, Bi-LSTM, Bi-GRU, Transformer)
- Sujeitos → slots 1–5 (user_a … user_e)
- Classes  → slots 1–3 (Esquerda, Direita, Neutro)
- v1 × v2  → cinza (antes) × azul (depois)
- Magnitude (heatmaps) → rampa sequencial de um só tom (azul)
- Ganho/perda (Δ)      → divergente azul ↔ vermelho com meio cinza
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

MODEL_ORDER = ["cnn", "lstm", "gru", "transformer"]
MODEL_LABELS = {"cnn": "CNN", "lstm": "Bi-LSTM", "gru": "Bi-GRU", "transformer": "Transformer"}
MODEL_COLORS = dict(zip(MODEL_ORDER, SERIES))

USERS = ["user_a", "user_b", "user_c", "user_d", "user_e"]
USER_COLORS = dict(zip(USERS, SERIES))

CLASS_LABELS = ["Esquerda", "Direita", "Neutro"]
CLASS_COLORS = dict(zip(range(3), SERIES[:3]))

V1_COLOR = "#a9a7a0"
V2_COLOR = SERIES[0]

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"
NEUTRAL = "#f0efec"
NEGATIVE = "#e34948"

SEQUENTIAL = LinearSegmentedColormap.from_list(
    "seq_blue",
    ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)
DIVERGING = LinearSegmentedColormap.from_list(
    "div_red_blue",
    ["#b1302f", "#e34948", "#f2a3a2", NEUTRAL, "#9ec5f4", "#3987e5", "#184f95"],
)

DPI = 300


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.labelsize": 9,
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "text.color": INK,
        "lines.linewidth": 1.6,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    })


def no_grid(ax) -> None:
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def percent_axis(ax, axis: str = "y") -> None:
    from matplotlib.ticker import PercentFormatter

    formatter = PercentFormatter(xmax=1.0, decimals=0)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(formatter)


def save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {path}")
    return path


apply_style()
