"""Análise exploratória do dataset de EEG (só dados: nada de modelos).

Descreve o dataset antes de qualquer preparação para modelagem:

    1. carregamento e descrição da coleta
    2. qualidade dos dados
    3. estrutura temporal da coleta (blocos e ciclo das classes)
    4. distribuição dos valores e correlação entre features
    5. valores extremos
    6. dependência temporal e tamanho amostral efetivo
    7. variação entre blocos × variação entre classes
    8. onde há diferença entre as classes
    9. participante × classe

Autocontida: lê data/raw/ diretamente, sem usar o resto de src/.

Saída: resultados/eda/ (tabelas/*.csv e figuras/eda_01 … eda_12)
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter, ScalarFormatter  # noqa: E402
from scipy import optimize, stats  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

SAIDA = Path(__file__).resolve().parents[2] / "resultados" / "eda"
DATA_DIR = SAIDA.parents[1] / "data" / "raw"
FIG_DIR = SAIDA / "figuras"
TAB_DIR = SAIDA / "tabelas"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)


def tabela(df, nome):
    # Mostra a tabela no terminal e grava em resultados/eda/tabelas/<nome>.csv
    # (colunas de dois níveis viram uma só, ex.: "user_a Alpha").
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [" ".join(map(str, c)) for c in df.columns]
    print(f"\n{df.to_string()}\n  → {TAB_DIR / nome}.csv")
    df.to_csv(TAB_DIR / f"{nome}.csv", sep=";", decimal=",", encoding="utf-8-sig")


RANDOM_STATE = 42

pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 160)
pd.set_option("display.float_format", lambda x: f"{x:,.3f}")


# ---------------------------------------------------------------------------
CLASS_NAMES = {0: "Esquerda", 1: "Direita", 2: "Neutro"}
CLASS_COLORS = {0: "#2a78d6", 1: "#b0478f", 2: "#a8742f"}
CLASS_MARKERS = {0: "o", 1: "s", 2: "^"}

ACCENT = "#2a78d6"     # série em foco
SECOND = "#52514e"     # série de referência (grafite)
CONTEXT = "#d6d4cc"    # pontos de contexto
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"

SEQUENTIAL = LinearSegmentedColormap.from_list(
    "azul", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list(
    "marrom_azul", ["#5c3d14", "#8a5a22", "#b98546", "#dcc29b", "#f0efec",
                    "#9ec5f4", "#3987e5", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 9,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 10, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "legend.frameon": False, "lines.linewidth": 2,
})


def num(v, casas=2):
    # Número com vírgula decimal (padrão do texto em português).
    return f"{v:.{casas}f}".replace(".", ",")


_virgula = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "−"))


def salvar(fig, nome):
    # Vírgula decimal nos eixos numéricos e eixos log sem notação científica;
    # depois salva em resultados/eda/figuras/.
    for ax in fig.axes:
        for eixo in (ax.xaxis, ax.yaxis):
            if eixo.get_scale() == "log":
                eixo.set_major_formatter(_virgula)
                eixo.set_minor_formatter(NullFormatter())
            elif isinstance(eixo.get_major_formatter(), ScalarFormatter):
                eixo.set_major_formatter(_virgula)
    fig.savefig(FIG_DIR / f"{nome}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [figura] {FIG_DIR / nome}.png")


GRAF_DIR = SAIDA / "graficos"


def _jsonavel(obj):
    # numpy → listas e números do Python, com 4 casas; NaN vira null.
    if isinstance(obj, dict):
        return {str(k): _jsonavel(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray, pd.Index, pd.Series)):
        return [_jsonavel(v) for v in (obj.tolist() if hasattr(obj, "tolist") else obj)]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if np.isnan(obj) else round(float(obj), 4)
    return obj


def dados(nome, obj):
    # Os dados da figura <nome>, para o dashboard desenhar o gráfico interativo
    # (resultados/eda/graficos/<nome>.json, com os mesmos números do PNG).
    GRAF_DIR.mkdir(parents=True, exist_ok=True)
    (GRAF_DIR / f"{nome}.json").write_text(json.dumps(_jsonavel(obj), ensure_ascii=False), encoding="utf-8")


def legenda_classes(ax_or_fig, **kwargs):
    handles = [Line2D([], [], color=CLASS_COLORS[k], marker=CLASS_MARKERS[k], linestyle="",
                      markersize=6, label=CLASS_NAMES[k]) for k in range(3)]
    ax_or_fig.legend(handles=handles, **kwargs)

# 1. Carregamento e descrição da coleta

# ---------------------------------------------------------------------------
USERS = [f"user_{c}" for c in "abcde"]
CHANNELS = ["AF3", "T7", "Pz", "T8", "AF4"]
BANDS = ["Theta", "Alpha", "BetaL", "BetaH", "Gamma"]
RAW_COLS = [f"POW.{ch}.{b}" for ch in CHANNELS for b in BANDS]
RECORDS_PER_SECOND = 8  # Leite (2023)

frames = []
for user in USERS:
    part = pd.read_csv(DATA_DIR / f"{user}.csv")
    part["participante"] = user
    frames.append(part)
df_raw = pd.concat(frames, ignore_index=True)

print(f"Registros: {len(df_raw):,}  |  colunas: {df_raw.shape[1]} (25 potências + classe + participante)")
print(df_raw.head().to_string())

# ---------------------------------------------------------------------------
contagem = pd.crosstab(df_raw["participante"], df_raw["Values"].map(CLASS_NAMES))[list(CLASS_NAMES.values())]
contagem["total"] = contagem.sum(axis=1)
contagem["duração (min)"] = contagem["total"] / RECORDS_PER_SECOND / 60
tabela(contagem, "registros_por_classe")

# 2. Qualidade dos dados

# ---------------------------------------------------------------------------
qualidade = pd.DataFrame({
    "valores ausentes": df_raw.isna().sum().sum(),
    "linhas duplicadas": df_raw.duplicated().sum(),
    "classes encontradas": [sorted(df_raw["Values"].unique().tolist())],
    "valores <= 0 (potência)": (df_raw[RAW_COLS] <= 0).sum().sum(),
}, index=["dataset"]).T
tabela(qualidade, "qualidade")

nao_positivos = (df_raw[RAW_COLS] <= 0).groupby(df_raw["participante"]).sum()
nao_positivos = nao_positivos.loc[:, nao_positivos.sum() > 0]
print("Valores <= 0 por participante e coluna:")
tabela(nao_positivos, "valores_nao_positivos")

df = df_raw.dropna().drop_duplicates().reset_index(drop=True)
df["Values"] = df["Values"].astype(int)

# ---------------------------------------------------------------------------
tabela(df[RAW_COLS].describe().T.round(3), "estatisticas_descritivas")


# 3. Estrutura temporal da coleta

# ---------------------------------------------------------------------------
novo_bloco = (df["Values"] != df["Values"].shift()) | (df["participante"] != df["participante"].shift())
df["bloco"] = novo_bloco.groupby(df["participante"]).cumsum().astype(int) - 1
df["posicao"] = df.groupby("participante").cumcount()

blocos = (df.groupby(["participante", "bloco"])
            .agg(classe=("Values", "first"), registros=("Values", "size"))
            .reset_index())
blocos["segundos"] = blocos["registros"] / RECORDS_PER_SECOND

sequencia = blocos.groupby("participante")["classe"].apply(lambda s: "".join("EDN"[v] for v in s))
print("Sequência de classes dos blocos (E = Esquerda, D = Direita, N = Neutro):")
for user, seq in sequencia.items():
    print(f"  {user}: {seq}  ({len(seq)} blocos, ciclo E→D→N fixo: {seq == 'EDN' * 12})")

# ---------------------------------------------------------------------------
resumo_blocos = blocos.groupby("participante")["registros"].describe()[["count", "min", "50%", "max"]]
resumo_blocos.columns = ["blocos", "mín.", "mediana", "máx."]
resumo_blocos["blocos > 450 registros"] = blocos[blocos["registros"] > 450].groupby("participante").size()
resumo_blocos = resumo_blocos.fillna(0).astype(int)
resumo_blocos["mediana (s)"] = resumo_blocos["mediana"] / RECORDS_PER_SECOND
tabela(resumo_blocos, "duracao_blocos")

por_classe = pd.DataFrame({
    "blocos por classe": blocos.groupby(["participante", "classe"]).size().unstack().rename(columns=CLASS_NAMES).astype(str).agg(" / ".join, axis=1),
    "registros por classe (%)": (pd.crosstab(df["participante"], df["Values"], normalize="index") * 100).round(1).astype(str).agg(" / ".join, axis=1),
})
por_classe.columns = ["blocos E / D / N", "registros E / D / N (%)"]
tabela(por_classe, "balanco_classes")

# ---------------------------------------------------------------------------
fig, axes = plt.subplots(5, 1, figsize=(7.2, 7.4), sharex=True, sharey=True)
for ax, user in zip(axes, USERS):
    b = blocos[blocos["participante"] == user]
    ax.plot(b["bloco"], b["segundos"], color=AXIS, linewidth=1, zorder=1)
    for k in range(3):
        sel = b[b["classe"] == k]
        ax.scatter(sel["bloco"], sel["segundos"], color=CLASS_COLORS[k], marker=CLASS_MARKERS[k],
                   s=26, edgecolor=SURFACE, linewidth=1, zorder=2)
    ax.set_ylabel(user, rotation=0, ha="right", va="center", color=INK)
    ax.set_yscale("log")
    ax.set_yticks([50, 100, 200, 300])
axes[0].set_title("Duração de cada bloco, na ordem de gravação (segundos, escala log)")
axes[-1].set_xlabel("Bloco (ordem de gravação)")
legenda_classes(fig, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.03, 1, 1))
salvar(fig, "eda_01_duracao_blocos")
dados("eda_01_duracao_blocos", {"classes": list(CLASS_NAMES.values()), "participantes": [
    {"participante": u, "blocos": blocos.loc[blocos["participante"] == u, ["bloco", "classe", "segundos"]]
     .to_dict(orient="records")} for u in USERS]})


# 4. Distribuição dos valores e correlação entre features

# ---------------------------------------------------------------------------
LOG_COLS = [c.replace("POW.", "LOG.") for c in RAW_COLS]
POWER_FLOOR = 1e-3
for raw, log in zip(RAW_COLS, LOG_COLS):
    df[log] = np.log10(np.maximum(df[raw].to_numpy(dtype=float), POWER_FLOOR))

# Assimetria calculada dentro de cada participante (os níveis diferem muito entre pessoas,
# ver a tabela de medianas abaixo) e resumida pela mediana entre participantes.
grupos = df.groupby("participante")
assimetria = pd.DataFrame({
    "bruta": [grupos[c].apply(stats.skew).median() for c in RAW_COLS],
    "log": [grupos[c].apply(stats.skew).median() for c in LOG_COLS],
}, index=[c.replace("POW.", "") for c in RAW_COLS])
print(f"Assimetria (skewness) mediana das 25 features: bruta {assimetria['bruta'].median():.2f} | log {assimetria['log'].median():.2f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1, 1.6]})
exemplo = "T7.Gamma"
bins = np.linspace(df[f"LOG.{exemplo}"].min(), df[f"LOG.{exemplo}"].max(), 81)
destaque = df["participante"] == "user_a"
ax1.hist([df.loc[~destaque, f"LOG.{exemplo}"], df.loc[destaque, f"LOG.{exemplo}"]], bins=bins, stacked=True,
         color=[CONTEXT, ACCENT], edgecolor=SURFACE, linewidth=0.3, label=["user_b a user_e", "user_a"])
ax1.legend(loc="upper right")
ax1.set_title(f"Distribuição de {exemplo} (log10 da potência)")
ax1.set_xlabel("log10(potência)")
ax1.set_ylabel("Registros")

ordem = np.arange(len(assimetria))
ax2.scatter(assimetria["bruta"], ordem, color=SECOND, marker="s", s=22, label="bruta")
ax2.scatter(assimetria["log"], ordem, color=ACCENT, marker="o", s=22, label="log")
ax2.axvline(0, color=AXIS, linewidth=1)
ax2.set_yticks(ordem)
ax2.set_yticklabels(assimetria.index, fontsize=7)
ax2.invert_yaxis()
ax2.set_title("Assimetria de cada feature (mediana dos participantes)")
ax2.set_xlabel("Assimetria (0 = simétrica)")
ax2.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.11))
fig.tight_layout()
salvar(fig, "eda_02_distribuicao_log")
dados("eda_02_distribuicao_log", {
    "exemplo": exemplo, "bordas": bins,
    "contagens": {"user_b a user_e": np.histogram(df.loc[~destaque, f"LOG.{exemplo}"], bins=bins)[0],
                  "user_a": np.histogram(df.loc[destaque, f"LOG.{exemplo}"], bins=bins)[0]},
    "assimetria": [{"feature": f, "bruta": r["bruta"], "log": r["log"]} for f, r in assimetria.iterrows()]})

# ---------------------------------------------------------------------------
mediana_log = (df.groupby("participante")[LOG_COLS].median().T)
mediana_log.index = pd.MultiIndex.from_tuples([tuple(c.split(".")[1:]) for c in mediana_log.index], names=["eletrodo", "banda"])
print("Mediana do log10 da potência por eletrodo × banda (colunas = participantes):")
tabela(mediana_log.unstack("banda").round(2), "mediana_log")

# ---------------------------------------------------------------------------
# Correlação entre as 25 features (log), calculada dentro de cada participante e depois
# promediada, para que a diferença de nível entre pessoas não crie correlação artificial.
corr = np.mean([df.loc[df["participante"] == u, LOG_COLS].corr().to_numpy() for u in USERS], axis=0)
labels = [c.replace("LOG.", "") for c in LOG_COLS]

same_channel, same_band, other = [], [], []
for i in range(25):
    for j in range(i + 1, 25):
        (ci, bi), (cj, bj) = divmod(i, 5), divmod(j, 5)
        (same_channel if ci == cj else same_band if bi == bj else other).append(corr[i, j])
print(f"Correlação média | mesmo eletrodo, bandas diferentes: {np.mean(same_channel):.2f}"
      f" | mesma banda, eletrodos diferentes: {np.mean(same_band):.2f} | demais pares: {np.mean(other):.2f}")

fig, ax = plt.subplots(figsize=(7.2, 6.2))
im = ax.imshow(corr, cmap=DIVERGING, vmin=-1, vmax=1)
ax.grid(False)
ax.set_xticks(range(25)); ax.set_xticklabels(labels, rotation=90, fontsize=6.5)
ax.set_yticks(range(25)); ax.set_yticklabels(labels, fontsize=6.5)
for k in range(1, 5):
    ax.axhline(k * 5 - 0.5, color=SURFACE, linewidth=1.5)
    ax.axvline(k * 5 - 0.5, color=SURFACE, linewidth=1.5)
cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cbar.set_label("Correlação de Pearson (log), média dos participantes")
cbar.outline.set_visible(False)
ax.set_title("Correlação entre as 25 features (blocos = eletrodos)")
salvar(fig, "eda_03_correlacao")
dados("eda_03_correlacao", {"rotulos": labels, "matriz": corr})


# 5. Valores extremos

# ---------------------------------------------------------------------------
def fora_iqr(X, m):
    q1, q3 = np.percentile(X, 25, axis=0), np.percentile(X, 75, axis=0)
    iqr = q3 - q1
    return (X < q1 - m * iqr) | (X > q3 + m * iqr)

linhas, por_feature, por_terco = [], {}, {}
for m in (1.5, 3.0):
    feats = []
    for user in USERS:
        ud = df[df["participante"] == user]
        out = fora_iqr(ud[LOG_COLS].to_numpy(), m)
        reg = out.any(axis=1)
        terco = ud["bloco"].to_numpy() // 12
        classe = ud["Values"].to_numpy()
        feats.append(out.mean(axis=0))
        linha = {"m": m, "participante": user, "valores (%)": out.mean() * 100, "registros (%)": reg.mean() * 100}
        for t, nome in enumerate(["1º terço", "2º terço", "3º terço"]):
            linha[f"registros {nome} (%)"] = reg[terco == t].mean() * 100
        for k in range(3):
            linha[f"registros {CLASS_NAMES[k]} (%)"] = reg[classe == k].mean() * 100
        linhas.append(linha)
    por_feature[m] = np.mean(feats, axis=0)

extremos = pd.DataFrame(linhas).set_index(["m", "participante"])
tabela(extremos.round(1), "valores_extremos")

# ---------------------------------------------------------------------------
top = pd.Series(por_feature[1.5], index=labels).sort_values(ascending=False)
print("Features com mais valores fora de 1,5×IQR (média dos participantes, %):")
print((top.head(8) * 100).round(1).to_string())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={"width_ratios": [1, 1.25]})
grid = por_feature[1.5].reshape(5, 5) * 100
im = ax1.imshow(grid, cmap=SEQUENTIAL, vmin=0)
ax1.grid(False)
ax1.set_xticks(range(5)); ax1.set_xticklabels(BANDS)
ax1.set_yticks(range(5)); ax1.set_yticklabels(CHANNELS)
for (i, j), v in np.ndenumerate(grid):
    ax1.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7.5,
             color=SURFACE if v > grid.max() * 0.6 else INK)
ax1.set_title("Valores fora de 1,5×IQR por feature (%)")

tercos = extremos.loc[1.5][[f"registros {t} (%)" for t in ["1º terço", "2º terço", "3º terço"]]].to_numpy()
im2 = ax2.imshow(tercos, cmap=SEQUENTIAL, vmin=0, aspect="auto")
ax2.grid(False)
ax2.set_xticks(range(3)); ax2.set_xticklabels(["blocos 0–11", "blocos 12–23", "blocos 24–35"])
ax2.set_yticks(range(5)); ax2.set_yticklabels(USERS)
for (i, j), v in np.ndenumerate(tercos):
    ax2.text(j, i, f"{v:.0f}%", ha="center", va="center", fontsize=8,
             color=SURFACE if v > tercos.max() * 0.6 else INK)
ax2.set_title("Registros com algum valor fora de 1,5×IQR, por terço da sessão")
fig.tight_layout()
salvar(fig, "eda_04_valores_extremos")
dados("eda_04_valores_extremos", {"eletrodos": CHANNELS, "bandas": BANDS, "por_feature": grid,
                                  "participantes": USERS, "tercos_rotulos": ["blocos 0–11", "blocos 12–23", "blocos 24–35"],
                                  "tercos": tercos})


# 6. Dependência temporal e tamanho amostral efetivo

# ---------------------------------------------------------------------------
def acf(x, max_lag):
    x = x - x.mean()
    size = 1 << (2 * len(x) - 1).bit_length()
    spec = np.fft.rfft(x, size)
    acov = np.fft.irfft(spec * np.conj(spec), size)[: max_lag + 1]
    return acov / acov[0]

MAX_LAG = 3000
BLOCK_LEN = int(blocos["registros"].median())
acfs = {u: np.array([acf(df.loc[df["participante"] == u, c].to_numpy(), MAX_LAG) for c in LOG_COLS]) for u in USERS}

tabela_acf = pd.DataFrame({
    u: {"distância 1": a.mean(axis=0)[1], "distância 8 (~1 s)": a.mean(axis=0)[8],
        f"distância {BLOCK_LEN} (~1 bloco)": a.mean(axis=0)[BLOCK_LEN]}
    for u, a in acfs.items()}).T
tabela(tabela_acf.round(2), "autocorrelacao")

lags = np.arange(1, 1201)
fig, axes = plt.subplots(1, 5, figsize=(11, 2.8), sharey=True)
for ax, user in zip(axes, USERS):
    ax.plot(lags, acfs[user].mean(axis=0)[1:1201], color=ACCENT)
    ax.axvline(BLOCK_LEN, color=MUTED, linewidth=1, linestyle="--")
    ax.axhline(0, color=AXIS, linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_title(user, fontsize=9)
    ax.set_xlabel("Distância (registros)")
axes[0].set_ylabel("Autocorrelação média")
axes[0].set_ylim(-0.1, 1.0)
axes[0].text(BLOCK_LEN * 0.8, 0.95, "1 bloco", color=INK2, fontsize=7.5, ha="right", va="top")
fig.suptitle("Registros vizinhos são quase cópias (autocorrelação média das 25 features, log)",
             x=0.01, ha="left", fontsize=10, fontweight="semibold")
fig.tight_layout()
salvar(fig, "eda_05_autocorrelacao")
dados("eda_05_autocorrelacao", {"bloco": BLOCK_LEN, "lags": lags,
                                "participantes": {u: acfs[u].mean(axis=0)[1:1201] for u in USERS}})

# Tamanho amostral efetivo

# ---------------------------------------------------------------------------
def tempo_integrado(r):
    tau = 1.0
    for M in range(1, len(r) - 1):
        tau = 1 + 2 * r[1:M + 1].sum()
        if M >= 5 * tau:
            break
    return tau

linhas = []
for user in USERS:
    n = int((df["participante"] == user).sum())
    taus = [tempo_integrado(r) for r in acfs[user]]
    ar1 = [n * (1 - r[1]) / (1 + r[1]) for r in acfs[user]]
    linhas.append({"participante": user, "registros (N)": n, "τ mediano (registros)": np.median(taus),
                   "τ mediano (s)": np.median(taus) / RECORDS_PER_SECOND,
                   "n efetivo (N/τ)": n / np.median(taus), "n efetivo AR(1)": np.median(ar1),
                   "blocos": 36, "blocos por classe": 12})
n_efetivo = pd.DataFrame(linhas).set_index("participante")
tabela(n_efetivo.round(1), "tamanho_amostral_efetivo")

# O que 12 blocos por classe permitem detectar

# ---------------------------------------------------------------------------
def poder_anova(f, n_total=36, k=3, alpha=0.05):
    df1, df2 = k - 1, n_total - k
    critico = stats.f.ppf(1 - alpha, df1, df2)
    return 1 - stats.ncf.cdf(critico, df1, df2, f ** 2 * n_total)

f80 = optimize.brentq(lambda f: poder_anova(f) - 0.8, 0.01, 3)
eta2_min = f80 ** 2 / (1 + f80 ** 2)
print(f"Menor efeito detectável com poder 0,8 (36 blocos, 3 classes): f = {f80:.2f}  →  η² = {eta2_min:.3f}")


def eta2(values, groups):
    total = ((values - values.mean()) ** 2).sum()
    entre = sum((groups == g).sum() * (values[groups == g].mean() - values.mean()) ** 2 for g in np.unique(groups))
    return entre / total if total > 0 else 0.0

eta2_blocos = {}
for user in USERS:
    ud = df[df["participante"] == user]
    medias = ud.groupby("bloco")[LOG_COLS].mean()
    classes = ud.groupby("bloco")["Values"].first().to_numpy()
    eta2_blocos[user] = np.array([eta2(medias[c].to_numpy(), classes) for c in LOG_COLS])

efeito = pd.DataFrame({u: {"η² mediano": np.median(e), "η² máximo": e.max(),
                           f"features com η² ≥ {eta2_min:.2f}": int((e >= eta2_min).sum())}
                       for u, e in eta2_blocos.items()}).T
tabela(efeito.round(3), "efeito_classe")

grade = np.linspace(0, 0.6, 301)
f_grade = np.sqrt(grade / (1 - grade))
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1.6], "hspace": 0.12})
ax.plot(grade, [poder_anova(f) for f in f_grade], color=ACCENT)
ax.axhline(0.8, color=MUTED, linewidth=1, linestyle="--")
for a in (ax, ax2):
    a.axvline(eta2_min, color=MUTED, linewidth=1, linestyle="--")
ax.text(eta2_min + 0.008, 0.08, f"η² = {num(eta2_min)}: menor efeito\ndetectável com poder 0,8", color=INK2, fontsize=8)
ax.set_ylabel("Poder da ANOVA\n(12 blocos por classe)")
ax.set_ylim(0, 1.02)
ax.set_title("Efeito de classe observado × efeito detectável")
for i, user in enumerate(USERS):
    e = eta2_blocos[user]
    ax2.plot([e.min(), e.max()], [i, i], color=SECOND, linewidth=1.2)
    ax2.plot([np.median(e)], [i], marker="o", color=SECOND, markersize=5.5, linestyle="")
ax2.set_yticks(range(5)); ax2.set_yticklabels(USERS)
ax2.set_ylim(4.6, -0.6)
ax2.grid(axis="y", visible=False)
ax2.set_xlim(0, 0.6)
ax2.set_xlabel("Efeito da classe (η² nas médias por bloco); embaixo: mediana (●) e faixa das 25 features")
salvar(fig, "eda_06_poder_estatistico")
dados("eda_06_poder_estatistico", {
    "eta2_min": eta2_min, "eta2": grade, "poder": [poder_anova(f) for f in f_grade],
    "participantes": [{"participante": u, "min": eta2_blocos[u].min(), "mediana": np.median(eta2_blocos[u]),
                       "max": eta2_blocos[u].max()} for u in USERS]})


# 7. Variação entre blocos × variação entre classes

# ---------------------------------------------------------------------------
linhas = []
for user in USERS:
    ud = df[df["participante"] == user]
    e_cls = np.array([eta2(ud[c].to_numpy(), ud["Values"].to_numpy()) for c in LOG_COLS])
    e_blk = np.array([eta2(ud[c].to_numpy(), ud["bloco"].to_numpy()) for c in LOG_COLS])
    linhas.append({"participante": user, "η² classe (%)": np.median(e_cls) * 100,
                   "η² bloco além da classe (%)": np.median(e_blk - e_cls) * 100})
variancia = pd.DataFrame(linhas).set_index("participante")
variancia["razão bloco / classe"] = variancia["η² bloco além da classe (%)"] / variancia["η² classe (%)"]
tabela(variancia.round(1), "variancia_bloco_classe")

fig, ax = plt.subplots(figsize=(7.2, 2.9))
y = np.arange(5)
ax.scatter(variancia["η² classe (%)"], y, color=SECOND, marker="s", s=34, label="η² classe", zorder=3)
ax.scatter(variancia["η² bloco além da classe (%)"], y, color=ACCENT, marker="o", s=34,
           label="η² bloco além da classe", zorder=3)
for i, (a, b) in enumerate(zip(variancia["η² classe (%)"], variancia["η² bloco além da classe (%)"])):
    ax.plot([a, b], [i, i], color=AXIS, linewidth=1, zorder=1)
ax.set_xscale("log")
ax.set_xticks([0.5, 1, 2, 5, 10, 20, 50])
ax.set_yticks(y); ax.set_yticklabels(USERS)
ax.invert_yaxis()
ax.set_xlabel("Fração da variância explicada (%, escala log), mediana das 25 features")
ax.set_title("O sinal varia mais entre blocos do que entre classes")
ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.22))
salvar(fig, "eda_07_variancia_bloco_classe")
dados("eda_07_variancia_bloco_classe", {"participantes": [
    {"participante": u, "classe": r["η² classe (%)"], "bloco": r["η² bloco além da classe (%)"]}
    for u, r in variancia.iterrows()]})

# ---------------------------------------------------------------------------
# Deriva ao longo da sessão: nível médio do sinal (média das 25 features, log) em cada bloco.
fig, axes = plt.subplots(5, 1, figsize=(7.2, 8.0), sharex=True)
deriva, deriva_graf = {}, []
for ax, user in zip(axes, USERS):
    ud = df[df["participante"] == user]
    nivel = ud[LOG_COLS].mean(axis=1).groupby(ud["bloco"]).mean()
    classe = ud.groupby("bloco")["Values"].first()
    rho, p = stats.spearmanr(nivel.index, nivel.values)
    deriva[user] = {"ρ de Spearman (nível × ordem do bloco)": rho, "p": p}
    deriva_graf.append({"participante": user, "rho": rho, "blocos": [
        {"bloco": b, "classe": classe.loc[b], "nivel": nivel.loc[b]} for b in nivel.index]})
    ax.plot(nivel.index, nivel.values, color=AXIS, linewidth=1, zorder=1)
    for k in range(3):
        idx = classe.index[classe == k]
        ax.scatter(idx, nivel.loc[idx], color=CLASS_COLORS[k], marker=CLASS_MARKERS[k], s=26,
                   edgecolor=SURFACE, linewidth=1, zorder=2)
    ax.set_ylabel(user, rotation=0, ha="right", va="center", color=INK)
    ax.text(1.0, 0.92, f"ρ = {num(rho)}".replace("-", "−"), transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK2)
axes[0].set_title("Nível médio do sinal em cada bloco (log da potência)")
axes[-1].set_xlabel("Bloco (ordem de gravação)")
legenda_classes(fig, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.03, 1, 1))
salvar(fig, "eda_08_deriva_sessao")
dados("eda_08_deriva_sessao", {"classes": list(CLASS_NAMES.values()), "participantes": deriva_graf})
tabela(pd.DataFrame(deriva).T.round(3), "deriva_sessao")


# 8. Onde há diferença entre as classes

# ---------------------------------------------------------------------------
ASY_COLS = []
for left, right in [("AF3", "AF4"), ("T7", "T8")]:
    for b in BANDS:
        col = f"ASY.{left}-{right}.{b}"
        df[col] = df[f"LOG.{left}.{b}"] - df[f"LOG.{right}.{b}"]
        ASY_COLS.append(col)


def anova_blocos(ud, cols):
    medias = ud.groupby("bloco")[cols].mean()
    classes = ud.groupby("bloco")["Values"].first()
    linhas = []
    for c in cols:
        grupos = [medias.loc[classes == k, c].to_numpy() for k in range(3)]
        f, p = stats.f_oneway(*grupos)
        linhas.append({"feature": c, "F": f, "p": p})
    return pd.DataFrame(linhas).set_index("feature")

anova, resumo = {}, []
for user in USERS:
    ud = df[df["participante"] == user]
    r_log, r_asy = anova_blocos(ud, LOG_COLS), anova_blocos(ud, ASY_COLS)
    anova[user] = (r_log, r_asy)
    top3 = pd.concat([r_log, r_asy]).sort_values("p").head(3).index.str.replace("LOG.", "").tolist()
    resumo.append({"participante": user, "eletrodo × banda com p < 0,05": f"{(r_log['p'] < 0.05).sum()}/25",
                   "assimetrias com p < 0,05": f"{(r_asy['p'] < 0.05).sum()}/10",
                   "3 features com menor p": ", ".join(top3)})
tabela(pd.DataFrame(resumo).set_index("participante"), "anova_classes")

# ---------------------------------------------------------------------------
grades = {u: -np.log10(anova[u][0]["p"].to_numpy()).reshape(5, 5) for u in USERS}
grades_asy = {u: -np.log10(anova[u][1]["p"].to_numpy()).reshape(2, 5) for u in USERS}
vmax = max(4.0, max(g.max() for g in grades.values()), max(g.max() for g in grades_asy.values()))
limiar = -np.log10(0.05)

fig, axes = plt.subplots(2, 5, figsize=(11, 4.6),
                         gridspec_kw={"height_ratios": [5, 2.2], "hspace": 0.35, "wspace": 0.12})
for col, user in enumerate(USERS):
    for row, (grid, ylabels) in enumerate([(grades[user], CHANNELS), (grades_asy[user], ["AF3−AF4", "T7−T8"])]):
        ax = axes[row, col]
        im = ax.imshow(grid, cmap=SEQUENTIAL, vmin=0, vmax=vmax, aspect="auto")
        ax.grid(False)
        for (i, j), v in np.ndenumerate(grid):
            if v >= limiar:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, linewidth=1.4))
        ax.set_xticks(range(5))
        ax.set_xticklabels(BANDS if row == 1 else [], rotation=45, fontsize=7)
        ax.set_yticks(range(len(ylabels)))
        ax.set_yticklabels(ylabels if col == 0 else [], fontsize=7)
        if row == 0:
            ax.set_title(f"{user}  ({int((anova[user][0]['p'] < 0.05).sum())}/25)", fontsize=9)
cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
cbar.set_label("−log10(p), ANOVA entre classes\n(contorno = p < 0,05)", fontsize=8)
cbar.outline.set_visible(False)
fig.suptitle("Diferença entre classes por eletrodo × banda (acima) e assimetria hemisférica (abaixo)",
             x=0.06, y=1.0, ha="left", fontsize=10, fontweight="semibold")
salvar(fig, "eda_09_anova_classes")
dados("eda_09_anova_classes", {
    "eletrodos": CHANNELS, "bandas": BANDS, "assimetrias": ["AF3−AF4", "T7−T8"], "vmax": vmax, "limiar": limiar,
    "participantes": [{"participante": u, "grade": grades[u], "grade_asy": grades_asy[u],
                       "significativos": int((anova[u][0]["p"] < 0.05).sum())} for u in USERS]})

# ---------------------------------------------------------------------------
# Espectro médio por classe: log padronizado dentro de cada participante, média dos blocos ± IC 95%.
fig, axes = plt.subplots(5, 5, figsize=(10, 9.2), sharex=True, sharey=True)
x = np.arange(5)
espectro_media = np.zeros((5, 5, 3, 5))  # participante × eletrodo × classe × banda
espectro_ic = np.zeros((5, 5, 3, 5))
for row, user in enumerate(USERS):
    ud = df[df["participante"] == user].copy()
    ud[LOG_COLS] = StandardScaler().fit_transform(ud[LOG_COLS])
    medias = ud.groupby("bloco")[LOG_COLS].mean()
    classes = ud.groupby("bloco")["Values"].first()
    for col, ch in enumerate(CHANNELS):
        ax = axes[row, col]
        cols = [f"LOG.{ch}.{b}" for b in BANDS]
        for k in range(3):
            v = medias.loc[classes == k, cols].to_numpy()
            m = v.mean(axis=0)
            ic = 1.96 * v.std(axis=0, ddof=1) / np.sqrt(len(v))
            espectro_media[row, col, k], espectro_ic[row, col, k] = m, ic
            ax.plot(x, m, color=CLASS_COLORS[k], marker=CLASS_MARKERS[k], markersize=3.5, linewidth=1.4)
            ax.fill_between(x, m - ic, m + ic, color=CLASS_COLORS[k], alpha=0.10, linewidth=0)
        ax.axhline(0, color=AXIS, linewidth=0.8)
        ax.tick_params(labelsize=7)
        if row == 0:
            ax.set_title(ch, fontsize=9)
        if col == 0:
            ax.set_ylabel(user, rotation=0, ha="right", va="center", color=INK)
        if row == 4:
            ax.set_xticks(x)
            ax.set_xticklabels(BANDS, rotation=45, fontsize=7)
legenda_classes(fig, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.005))
fig.suptitle("Espectro médio por classe (log padronizado por participante; faixa = IC 95% entre blocos)",
             x=0.02, ha="left", fontsize=10, fontweight="semibold")
fig.tight_layout(rect=(0, 0.03, 1, 0.97))
salvar(fig, "eda_10_espectro_classes")
dados("eda_10_espectro_classes", {"participantes": USERS, "eletrodos": CHANNELS, "bandas": BANDS,
                                  "classes": list(CLASS_NAMES.values()), "media": espectro_media, "ic": espectro_ic})


# 9. Participante × classe

# ---------------------------------------------------------------------------
rng = np.random.default_rng(RANDOM_STATE)
idx = np.concatenate([rng.choice(np.flatnonzero(df["participante"].to_numpy() == u), 600, replace=False) for u in USERS])
amostra = df.iloc[idx]
grupo_part = amostra["participante"].to_numpy()
grupo_cls = amostra["Values"].to_numpy()

X_global = StandardScaler().fit_transform(amostra[LOG_COLS])
X_part = np.empty_like(X_global)
for u in USERS:
    sel = grupo_part == u
    X_part[sel] = StandardScaler().fit_transform(amostra.loc[sel, LOG_COLS])

silhuetas, embeddings = {}, {}
for nome, X in [("global", X_global), ("por participante", X_part)]:
    silhuetas[nome] = {"silhueta por participante": silhouette_score(X, grupo_part),
                       "silhueta por classe": silhouette_score(X, grupo_cls)}
    embeddings[nome] = TSNE(n_components=2, perplexity=30, random_state=RANDOM_STATE, init="pca").fit_transform(X)
tabela(pd.DataFrame(silhuetas).T.round(3), "silhueta")

# ---------------------------------------------------------------------------
def painel_tsne(emb, titulo, nome):
    fig, axes = plt.subplots(2, 5, figsize=(11, 4.9))
    for ax in axes.flat:
        ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
    for ax, user in zip(axes[0], USERS):
        sel = grupo_part == user
        ax.scatter(emb[~sel, 0], emb[~sel, 1], s=2, color=CONTEXT, linewidth=0)
        ax.scatter(emb[sel, 0], emb[sel, 1], s=3, color=ACCENT, linewidth=0)
        ax.set_title(user, fontsize=9)
    for ax, k in zip(axes[1], range(3)):
        sel = grupo_cls == k
        ax.scatter(emb[~sel, 0], emb[~sel, 1], s=2, color=CONTEXT, linewidth=0)
        ax.scatter(emb[sel, 0], emb[sel, 1], s=4, color=CLASS_COLORS[k], marker=CLASS_MARKERS[k], linewidth=0)
        ax.set_title(CLASS_NAMES[k], fontsize=9)
    axes[1, 3].set_visible(False); axes[1, 4].set_visible(False)
    fig.text(0.01, 0.73, "Por participante", rotation=90, va="center", fontsize=9, color=INK2)
    fig.text(0.01, 0.28, "Por classe", rotation=90, va="center", fontsize=9, color=INK2)
    fig.suptitle(titulo, x=0.03, ha="left", fontsize=10, fontweight="semibold")
    fig.tight_layout(rect=(0.02, 0, 1, 0.95))
    salvar(fig, nome)
    dados(nome, {"participantes": USERS, "classes": list(CLASS_NAMES.values()),
                 "x": np.round(emb[:, 0], 2), "y": np.round(emb[:, 1], 2),
                 "participante": [USERS.index(u) for u in grupo_part], "classe": grupo_cls})

painel_tsne(embeddings["global"], "t-SNE com padronização global: os grupos são os participantes", "eda_11_tsne_global")
painel_tsne(embeddings["por participante"], "t-SNE com padronização por participante", "eda_12_tsne_por_participante")


print(f"\nTabelas em {TAB_DIR} e figuras em {FIG_DIR}")
