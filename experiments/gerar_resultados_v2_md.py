"""Gera docs/resultados_v2.md — iteração v2 comparada com a v1.

Lê a tabela longa montada por `_v2_results.collect_all()` e os resumos de
explicabilidade em outputs/iterations/v2/analysis/, e escreve um documento em
português pronto para a monografia. Não treina nada; pode ser reexecutado à
vontade. O docs/resultados_tcc2.md (v1) não é tocado.

Uso:
    python experiments/gerar_resultados_v2_md.py
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from _v2_results import MODELS, USERS, V2_ROOT, collect_all
from gerar_resultados_md import DISPLAY, pct, points

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ITER_ROOT = V2_ROOT.parent
ANALYSIS_ROOT = ITER_ROOT / "analysis"
FIG_REL = "../outputs/iterations/v2/figures"
OUTPUT_PATH = PROJECT_ROOT / "docs" / "resultados_v2.md"

METRIC = "balanced_accuracy"
CHANCE = 1 / 3

POOLED_VARIANTS = ["raw", "log", "log_subjnorm", "log_rel_subjnorm"]
PERSONAL_VARIANTS = ["raw", "log", "log_rel"]
VARIANT_LABELS = {
    "raw": "raw (v1)",
    "log": "log",
    "log_subjnorm": "log + norm. por sujeito",
    "log_rel": "log + potência relativa",
    "log_rel_subjnorm": "log + relativa + norm. por sujeito",
}
PROTOCOL_LABELS = {"chronological": "hold-out cronológico", "blocked_cv": "CV por blocos"}
TRACK_LABELS = {"pooled": "agrupado", "personalized": "personalizado"}


def pm(values) -> str:
    values = [v for v in values if pd.notna(v)]
    if not values:
        return "—"
    if len(values) == 1:
        return pct(values[0])
    return f"{pct(np.mean(values))} ± {np.std(values, ddof=1) * 100:.1f}".replace(".", ",") + " pp"


def num(value, digits=2) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines + [""]


def fig(name: str, caption: str) -> list[str]:
    return [f"![{caption}]({FIG_REL}/{name})", f"*{caption}*", ""]


class Data:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.v2 = df[df.iteration == "v2"]
        self.v1 = df[df.iteration == "v1"]

    def subj(self, protocol, track, variants=None):
        """Média por (variante, modelo, sujeito) sobre as repetições."""
        sub = self.v2[(self.v2.protocol == protocol) & (self.v2.track == track) & (self.v2.subject != "ALL")]
        if variants:
            sub = sub[sub.variant.isin(variants)]
        return sub.groupby(["variant", "model", "subject"], as_index=False).mean(numeric_only=True)

    def has(self, protocol, track=None):
        sub = self.v2[self.v2.protocol == protocol]
        if track:
            sub = sub[sub.track == track]
        return not sub.empty

    def best_variant(self, protocol, track):
        variants = POOLED_VARIANTS if track == "pooled" else PERSONAL_VARIANTS
        sl = self.subj(protocol, track, [v for v in variants if v != "raw"])
        return sl.groupby("variant")[METRIC].mean().idxmax() if not sl.empty else None

    def seeds(self, protocol="chronological"):
        sub = self.v2[self.v2.protocol == protocol]
        return sorted(sub.rep.unique())


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


# ---------------------------------------------------------------------------
# Seções
# ---------------------------------------------------------------------------

def section_header(d: Data) -> list[str]:
    now = datetime.now().strftime("%d/%m/%Y às %H:%M")
    return [
        "# Resultados v2 — Novas representações de features, avaliação robusta e explicabilidade",
        "",
        f"*Documento gerado automaticamente por `experiments/gerar_resultados_v2_md.py` em {now}. "
        "Os resultados da v1 estão congelados em `outputs/iterations/v1/` "
        "(com SHA-256 de cada arquivo em `MANIFEST.json`) e o relatório anterior, "
        "`docs/resultados_tcc2.md`, permanece inalterado.*",
        "",
    ]


def section_summary(d: Data) -> list[str]:
    lines = ["## Resumo", ""]
    lines.append(
        "A v1 mostrou que, sob avaliação honesta (split cronológico), o modelo agrupado ficava no "
        "nível do acaso e que só 2 dos 5 sujeitos obtinham modelos personalizados úteis. Também "
        "mostrou um sintoma importante: em quase todos os treinos, a menor perda de validação "
        "acontecia na 1ª ou 2ª época. A v2 ataca esse problema **apenas pela representação dos "
        "dados** — arquiteturas e hiperparâmetros são exatamente os da v1 — e adiciona um "
        "protocolo de avaliação mais robusto e uma análise de explicabilidade."
    )
    lines.append("")

    bullets = []
    for protocol in ["blocked_cv", "chronological"]:
        for track in ["pooled", "personalized"]:
            if not d.has(protocol, track):
                continue
            best = d.best_variant(protocol, track)
            if best is None:
                continue
            sl = d.subj(protocol, track, ["raw", best])
            raw = sl[sl.variant == "raw"][METRIC].mean()
            new = sl[sl.variant == best][METRIC].mean()
            bullets.append(
                f"- **{TRACK_LABELS[track].capitalize()}, {PROTOCOL_LABELS[protocol]}:** acurácia "
                f"balanceada média de {pct(raw)} (raw/v1) para **{pct(new)}** com "
                f"*{VARIANT_LABELS[best]}* ({points(new - raw)})."
            )
    lines += bullets + [""]
    lines += [
        "**Leitura honesta dos números:**",
        "",
        "1. **A normalização por sujeito é o único ganho robusto.** No modelo agrupado, *log + "
        "normalização por sujeito* melhora o resultado nos dois protocolos e é estatisticamente "
        "significativa nos dois (Wilcoxon, p < 0,01). O tamanho do ganho, porém, depende muito do "
        "protocolo: +8,6 pp no hold-out cronológico e +1,8 pp na CV por blocos.",
        "2. **Nos modelos personalizados, as novas features não mudam o resultado** de forma "
        "significativa em nenhum protocolo (ganhos de 0 a 1,5 pp, dentro do ruído entre sementes).",
        "3. **O hold-out cronológico superestima os ganhos da v2.** Um diagnóstico dedicado mostrou "
        "que, para os sujeitos com sinal, o resultado depende de a validação usada no "
        "EarlyStopping estar temporalmente próxima do teste. Com a validação adjacente (como no "
        "hold-out), `user_a` chega a ~58–62%; com a validação longe do teste, cai para ~42–46% no "
        "mesmo trecho. A deriva do sinal entre trials faz a escolha dos pesos depender de dados "
        "recentes.",
        "4. **O limite está nos dados, não nas redes.** Só `user_c` fica acima do acaso em todos os "
        "protocolos, e só ele tem várias features que diferem entre as classes. `user_d` fica no "
        "acaso em todos, e nenhuma das suas 25 features canal × banda difere entre as classes. Os "
        "outros três mudam de lado conforme o trecho da sessão avaliado (`user_a` vai bem no fim da "
        "gravação e mal no resto; `user_b`, o contrário): o sinal discriminativo existe em parte da "
        "sessão e não em outra. Onde há sinal, os modelos usam exatamente as features que a "
        "estatística aponta — majoritariamente a banda **Gamma** em eletrodos temporais, o que "
        "levanta a hipótese de **artefato muscular** (ver Explicabilidade II).",
        "",
    ]
    return lines


def section_changes() -> list[str]:
    return [
        "## O que mudou da v1 para a v2",
        "",
        "### 1. Representação das features",
        "",
        "As 25 features de entrada continuam sendo a potência espectral de 5 eletrodos (AF3, T7, Pz, "
        "T8, AF4) em 5 bandas (Theta, Alpha, BetaL, BetaH, Gamma). O que muda é como esses valores "
        "são apresentados às redes:",
        "",
        *table(["Variante", "O que é", "Por quê"], [
            ["raw (v1)", "Potência bruta", "Referência. Distribuição muito assimétrica (assimetria mediana 6,7)."],
            ["log", "log10 da potência", "Potência espectral é aproximadamente log-normal: o log reduz a assimetria (mediana 1,2), transforma razões entre bandas em diferenças e faz o clipping de IQR deixar de achatar a cauda."],
            ["log + potência relativa", "log10 + log10(banda ÷ soma das bandas do canal), 2 mapas 5×5", "A potência relativa cancela o \"ganho\" de cada eletrodo, que varia com contato e impedância entre sessões."],
            ["+ norm. por sujeito", "IQR e StandardScaler ajustados no treino de cada sujeito", "Remove o deslocamento entre pessoas. Só se aplica ao modelo agrupado; equivale a uma sessão de calibração por usuário."],
        ]),
        "Todas as transformações são **por amostra e sem estado** (não olham para outras amostras), "
        "portanto não há risco de vazamento. A normalização por sujeito é ajustada apenas no treino "
        "de cada sujeito — exatamente como o StandardScaler global da v1. A única mudança no código "
        "dos modelos é a largura da entrada: com 2 mapas, a CNN recebe uma imagem 5×5 com 2 canais, "
        "e a LSTM/GRU/Transformer recebem 10 em vez de 5 valores por passo/token.",
        "",
        "### 2. Protocolo de avaliação",
        "",
        "- **Hold-out cronológico 80/20** — idêntico ao da v1 (mesmas amostras de teste), agora com "
        "3 sementes de inicialização (42, 43, 44) para medir a variância do treino.",
        "- **Validação cruzada por blocos de trials (novo).** Os 36 trials de cada sujeito são "
        "divididos em 6 folds de 6 trials consecutivos — 2 ciclos completos Esquerda → Direita → "
        "Neutro, o que mantém as classes balanceadas. Em cada rodada, um fold é teste, o seguinte é "
        "validação (para o EarlyStopping) e os outros 4 são treino. Nas fronteiras com o teste, 40 "
        "amostras de treino/validação são descartadas (*purge*), porque amostras vizinhas são quase "
        "cópias (ver Figura D1). Assim **todos os 36 trials de cada sujeito são testados exatamente "
        "uma vez**, contra ~7 no hold-out.",
        "",
        "### 3. Métricas",
        "",
        "A métrica principal passa a ser a **acurácia balanceada** (média do recall das três "
        "classes). O motivo: no teste cronológico as classes não ficam balanceadas, e a acurácia "
        "simples precisava ser comparada com a classe majoritária de cada sujeito. Na acurácia "
        "balanceada, o acaso é **33,3% para qualquer sujeito e qualquer split**, o que torna as "
        "comparações diretas. Também são reportados F1 macro, kappa de Cohen e a **acurácia por "
        "trial** (uma decisão por trial, pela média das probabilidades das suas amostras), que é "
        "como uma interface de BCI decide na prática.",
        "",
    ]


def section_equivalence(d: Data) -> list[str]:
    raw = d.v2[(d.v2.protocol == "chronological") & (d.v2.variant == "raw") & (d.v2.rep == "seed_42")]
    if raw.empty or d.v1.empty:
        return []
    rows = []
    diffs = []
    for track in ["pooled", "personalized"]:
        for model in MODELS:
            if track == "pooled":
                a = d.v1[(d.v1.track == track) & (d.v1.model == model) & (d.v1.subject == "ALL")].accuracy
                b = raw[(raw.track == track) & (raw.model == model) & (raw.subject == "ALL")].accuracy
            else:
                a = d.v1[(d.v1.track == track) & (d.v1.model == model)].accuracy
                b = raw[(raw.track == track) & (raw.model == model)].accuracy
            if a.empty or b.empty:
                continue
            diffs.append(b.mean() - a.mean())
            rows.append([TRACK_LABELS[track], DISPLAY[model], pct(a.mean()), pct(b.mean()), points(b.mean() - a.mean())])
    max_diff = max(abs(x) for x in diffs) if diffs else 0
    return [
        "## Verificação: a v2 reproduz a v1",
        "",
        "Antes de comparar, é preciso garantir que o código novo não mudou o pipeline. A variante "
        "*raw* da v2, com a semente 42 e o hold-out cronológico, é por construção o mesmo experimento "
        "da v1: os arrays de treino, validação e teste foram verificados como **idênticos bit a bit** "
        "aos da v1, para o agrupado e para cada sujeito.",
        "",
        *table(["Track", "Modelo", "Acurácia v1", "Acurácia v2 (raw, semente 42)", "Diferença"], rows),
        f"A diferença máxima é de {num(max_diff * 100, 2)} pp. Como os dados de entrada são "
        "idênticos, ela só pode vir do próprio treino — o TensorFlow em CPU não é determinístico "
        "(a ordem das operações em paralelo varia) — e tem a mesma ordem de grandeza do desvio "
        "entre sementes medido adiante (~2 pp). Portanto, **ganhos da v2 claramente maiores que "
        "essa faixa podem ser atribuídos às features**; ganhos menores, não.",
        "",
    ]


def _variant_model_table(d: Data, protocol: str, track: str) -> list[str]:
    variants = POOLED_VARIANTS if track == "pooled" else PERSONAL_VARIANTS
    sl = d.subj(protocol, track, variants)
    if sl.empty:
        return []
    header = ["Variante"] + [DISPLAY[m] for m in MODELS] + ["Média"]
    rows = []
    for v in variants:
        sv = sl[sl.variant == v]
        if sv.empty:
            continue
        row = [VARIANT_LABELS[v]]
        for m in MODELS:
            row.append(pm(sv[sv.model == m][METRIC]))
        row.append(f"**{pct(sv[METRIC].mean())}**")
        rows.append(row)
    return table(header, rows)


def section_chronological(d: Data) -> list[str]:
    if not d.has("chronological"):
        return []
    seeds = d.seeds("chronological")
    lines = [
        "## Resultados — hold-out cronológico (comparável à v1)",
        "",
        f"Cada célula é a acurácia balanceada média entre os 5 sujeitos ± desvio entre sujeitos. "
        f"Sementes usadas: {', '.join(s.replace('seed_', '') for s in seeds)} (média entre elas). "
        "No agrupado, a acurácia de cada sujeito é medida no recorte das suas amostras de teste.",
        "",
        "### Modelo agrupado",
        "",
        *_variant_model_table(d, "chronological", "pooled"),
        *fig("resultados/res_ablacao_pooled_chronological.png",
             "Figura R1 — Efeito de cada variante de features no modelo agrupado (hold-out cronológico)."),
        *fig("resultados/res_por_sujeito_pooled_chronological.png",
             "Figura R2 — Modelo agrupado, por sujeito."),
        "### Modelos personalizados",
        "",
        *_variant_model_table(d, "chronological", "personalized"),
        *fig("resultados/res_ablacao_personalized_chronological.png",
             "Figura R3 — Efeito de cada variante de features nos modelos personalizados (hold-out cronológico)."),
        *fig("resultados/res_heatmap_delta_chronological.png",
             "Figura R4 — Personalizado: raw × melhor variante e ganho por sujeito e modelo."),
    ]

    # v1 × v2 com as duas métricas
    rows = []
    for track in ["pooled", "personalized"]:
        best = d.best_variant("chronological", track)
        if best is None:
            continue
        v1 = d.v1[(d.v1.track == track) & (d.v1.subject != "ALL")]
        v2 = d.subj("chronological", track, [best])
        for model in MODELS:
            a1 = v1[v1.model == model].accuracy.mean()
            a2 = v2[v2.model == model].accuracy.mean()
            b1 = v1[v1.model == model][METRIC].mean() if track == "personalized" else np.nan
            b2 = v2[v2.model == model][METRIC].mean()
            rows.append([TRACK_LABELS[track], DISPLAY[model], VARIANT_LABELS[best], pct(a1), pct(a2),
                         points(a2 - a1), pct(b1) if pd.notna(b1) else "—", pct(b2),
                         points(b2 - b1) if pd.notna(b1) else "—"])
    lines += [
        "### v1 × v2 (mesmo teste)",
        "",
        "Médias entre os 5 sujeitos. A v2 usa a melhor variante de cada track. No agrupado da v1 só "
        "existe a acurácia simples por sujeito (os JSONs da v1 não guardavam as predições).",
        "",
        *table(["Track", "Modelo", "Variante v2", "Acurácia v1", "Acurácia v2", "Δ",
                "Acur. bal. v1", "Acur. bal. v2", "Δ"], rows),
        *fig("resultados/res_v1_vs_v2.png", "Figura R5 — v1 × v2 no mesmo hold-out cronológico."),
    ]

    # variância entre sementes
    chrono = d.v2[(d.v2.protocol == "chronological") & (d.v2.subject != "ALL")]
    if chrono.rep.nunique() > 1:
        spread = chrono.groupby(["variant", "track", "model", "subject"])[METRIC].std().mean()
        lines += [
            f"**Variância entre sementes.** Para a mesma configuração, o desvio padrão médio da "
            f"acurácia balanceada entre as {chrono.rep.nunique()} sementes é de "
            f"{num(spread * 100, 1)} pp. Diferenças entre arquiteturas menores que isso não devem "
            "ser interpretadas.",
            "",
        ]
    return lines


def section_cv(d: Data) -> list[str]:
    if not d.has("blocked_cv"):
        return []
    lines = [
        "## Resultados — validação cruzada por blocos (estimativa principal)",
        "",
        "Cada célula é a acurácia balanceada média entre os 5 sujeitos (cada sujeito resumido "
        "pela média dos seus 6 folds) ± desvio entre sujeitos. É a estimativa mais confiável deste "
        "documento: cobre os 36 trials de cada sujeito, com classes balanceadas no teste.",
        "",
        "### Modelo agrupado",
        "",
        *_variant_model_table(d, "blocked_cv", "pooled"),
        *fig("resultados/res_ablacao_pooled_blocked_cv.png",
             "Figura R6 — Efeito das variantes no modelo agrupado (CV por blocos)."),
        *fig("resultados/res_por_sujeito_pooled_blocked_cv.png",
             "Figura R7 — Modelo agrupado por sujeito (CV por blocos)."),
        "### Modelos personalizados",
        "",
        *_variant_model_table(d, "blocked_cv", "personalized"),
        *fig("resultados/res_ablacao_personalized_blocked_cv.png",
             "Figura R8 — Efeito das variantes nos modelos personalizados (CV por blocos)."),
        *fig("resultados/res_por_sujeito_personalized_blocked_cv.png",
             "Figura R9 — Personalizado por sujeito (CV por blocos)."),
        *fig("resultados/res_heatmap_delta_blocked_cv.png",
             "Figura R10 — Personalizado: raw × melhor variante, por sujeito e modelo (CV)."),
        *fig("resultados/res_cv_folds_personalizado.png",
             "Figura R11 — Dispersão entre folds: cada ponto é um grupo de 6 trials."),
    ]

    # tabela por sujeito com a melhor variante de cada track
    rows = []
    best_p = d.best_variant("blocked_cv", "personalized")
    best_g = d.best_variant("blocked_cv", "pooled")
    if best_p is None or best_g is None:
        return lines
    per = d.subj("blocked_cv", "personalized", ["raw", best_p])
    poo = d.subj("blocked_cv", "pooled", [best_g])
    for user in USERS:
        raw_u = per[(per.variant == "raw") & (per.subject == user)]
        new_u = per[(per.variant == best_p) & (per.subject == user)]
        poo_u = poo[poo.subject == user]
        if new_u.empty:
            continue
        best_model = new_u.loc[new_u[METRIC].idxmax()]
        rows.append([
            user, pct(raw_u[METRIC].mean()), pct(new_u[METRIC].mean()),
            f"{DISPLAY[best_model.model]} {pct(best_model[METRIC])}",
            pct(new_u.trial_accuracy.mean()) if "trial_accuracy" in new_u else "—",
            pct(poo_u[METRIC].mean()) if not poo_u.empty else "—",
            "**sim**" if new_u[METRIC].mean() > CHANCE + 0.05 else "não",
        ])
    lines += [
        "### Por sujeito",
        "",
        f"Personalizado com *{VARIANT_LABELS[best_p]}* e agrupado com *{VARIANT_LABELS[best_g]}*. "
        "Médias dos 4 modelos, exceto na coluna do melhor modelo. A última coluna adota um critério "
        "simples de utilidade: ao menos 5 pp acima do acaso na média dos modelos.",
        "",
        *table(["Sujeito", "Personalizado raw", "Personalizado v2", "Melhor modelo (v2)",
                "Acurácia por trial (v2)", "Agrupado v2", "Útil?"], rows),
        *fig("resultados/res_trial_vs_amostra.png",
             "Figura R12 — Acurácia por amostra × por trial (decisão sobre o trial inteiro)."),
        *fig("resultados/res_agrupado_vs_personalizado.png",
             "Figura R13 — Agrupado × personalizado por sujeito (melhor variante de cada track, CV)."),
        *fig("resultados/res_matrizes_confusao.png",
             "Figura R14 — Matrizes de confusão (recall por classe), personalizado, CV."),
    ]
    return lines


def section_diagnostic() -> list[str]:
    path = ANALYSIS_ROOT / "diagnostico_validacao.csv"
    if not path.exists():
        return []
    t = pd.read_csv(path)
    labels = {("pooled", "raw"): "agrupado, raw", ("pooled", "log_subjnorm"): "agrupado, log + norm. sujeito",
              ("personalized", "log"): "personalizado, log"}
    rows = []
    for (track, variant), g in t.groupby(["track", "variant"]):
        rows.append([labels.get((track, variant), f"{track}, {variant}"),
                     pct(g.next.mean()), pct(g.prev.mean()), pct(g.chrono.mean())])
    subj = t[(t.track == "pooled") & (t.variant == "log_subjnorm")].groupby("subject")[["next", "prev", "chrono"]].mean()
    subj_rows = [[u, pct(r.next), pct(r.prev), pct(r.chrono)] for u, r in subj.iterrows()]
    return [
        "## Por que o hold-out e a CV discordam? Um diagnóstico",
        "",
        "Os ganhos da v2 são bem maiores no hold-out cronológico do que na CV, e o fold 5 da CV "
        "testa praticamente o mesmo trecho que o hold-out (os últimos ~6–7 trials). Comparando os "
        "dois no mesmo trecho, a diferença relevante está na **validação usada pelo "
        "EarlyStopping**: no hold-out ela fica imediatamente antes do teste; no fold 5 da CV, como "
        "o fold de validação é o *seguinte* de forma circular, ela cai no fold 0 — os primeiros "
        "trials da gravação, longe do teste.",
        "",
        "Para testar essa hipótese, o fold 5 foi treinado de novo com a validação no fold "
        "**anterior** (adjacente ao teste), mantendo todo o resto igual (script "
        "`experiments/diagnostico_validacao.py`). Acurácia balanceada média dos 4 modelos e dos 5 "
        "sujeitos, sempre no mesmo trecho de teste:",
        "",
        *table(["Configuração", "Fold 5, validação distante (CV padrão)",
                "Fold 5, validação adjacente", "Hold-out cronológico"], rows),
        "Por sujeito, no agrupado com *log + norm. por sujeito*:",
        "",
        *table(["Sujeito", "Validação distante", "Validação adjacente", "Hold-out"], subj_rows),
        "**Conclusão.** Com a validação adjacente, o fold 5 reproduz o hold-out para os sujeitos "
        "com sinal (`user_a` e `user_c`), enquanto para os sujeitos sem sinal nada muda além do "
        "ruído. Ou seja, o hold-out não está errado: ele mede um cenário em que os pesos do modelo "
        "são escolhidos com dados **recentes**, o que é realista numa BCI calibrada logo antes do "
        "uso. A CV padrão mede um cenário mais pessimista. As duas leituras são consistentes com a "
        "deriva entre trials (Figura D2): o padrão que distingue as classes muda ao longo da sessão, "
        "e um modelo só funciona bem perto do momento em que foi ajustado. Isso aponta para "
        "**recalibração frequente ou adaptação contínua** como direção mais promissora do que "
        "redes maiores.",
        "",
    ]


def section_stats(d: Data) -> list[str]:
    rows = []
    for protocol in ["chronological", "blocked_cv"]:
        for track in ["pooled", "personalized"]:
            if not d.has(protocol, track):
                continue
            variants = POOLED_VARIANTS if track == "pooled" else PERSONAL_VARIANTS
            sl = d.subj(protocol, track, variants)
            raw = sl[sl.variant == "raw"].set_index(["model", "subject"])[METRIC]
            for v in variants[1:]:
                new = sl[sl.variant == v].set_index(["model", "subject"])[METRIC]
                pairs = pd.concat([raw, new], axis=1, keys=["raw", "new"]).dropna()
                if len(pairs) < 6:
                    continue
                diff = pairs.new - pairs.raw
                p = stats.wilcoxon(pairs.new, pairs.raw).pvalue if (diff != 0).any() else 1.0
                rows.append([PROTOCOL_LABELS[protocol], TRACK_LABELS[track], VARIANT_LABELS[v],
                             str(len(pairs)), points(diff.mean()), f"{int((diff > 0).sum())}/{len(pairs)}",
                             f"{p:.4f}".replace(".", ","), "**sim**" if p < 0.05 else "não"])
    if not rows:
        return []
    return [
        "## Os ganhos são estatisticamente significativos?",
        "",
        "Teste de Wilcoxon pareado (bicaudal) entre cada variante e o raw. Cada par é uma combinação "
        "sujeito × arquitetura (até 20 pares), resumida pela média das repetições, de modo que as "
        "amostras de EEG, autocorrelacionadas, não são tratadas como independentes.",
        "",
        *table(["Protocolo", "Track", "Variante", "Pares", "Ganho médio", "Pares com ganho",
                "p-valor", "p < 0,05?"], rows),
        "Os 20 pares não são totalmente independentes (os 4 modelos de um mesmo sujeito "
        "compartilham os dados), então os p-valores devem ser lidos como indicativos.",
        "",
    ]


def section_epochs(d: Data) -> list[str]:
    sub = d.df[d.df.best_epoch.notna() & ((d.df.iteration == "v1") | (d.df.rep == "seed_42"))
               & (d.df.protocol == "chronological")]
    sub = sub.drop_duplicates(["iteration", "variant", "track", "model", "subject", "rep"])
    if sub.empty:
        return []
    rows = []
    for (it, v), g in sub.groupby(["iteration", "variant"]):
        rows.append([it, VARIANT_LABELS.get(v, v), num(g.best_epoch.median(), 0),
                     pct((g.best_epoch <= 2).mean())])
    return [
        "## O sintoma \"melhor época = 1\" mudou?",
        "",
        "Época em que a perda de validação foi mínima (hold-out cronológico, semente 42).",
        "",
        *table(["Iteração", "Variante", "Mediana da melhor época", "Treinos com melhor época ≤ 2"], rows),
        "**Praticamente não mudou.** A variação entre as linhas é da mesma ordem da variação entre "
        "execuções do próprio raw (v1 × v2 raw). Mudar só a representação dos dados não resolve o "
        "overfitting precoce: a "
        "rede continua aprendendo, já na primeira passada, padrões específicos dos trials de "
        "treino (a Figura D2 explica por quê). Esse sintoma é um problema do regime de treino — "
        "regularização, augmentation, tamanho do modelo — e é o alvo natural da frente 3 dos "
        "próximos passos.",
        "",
        *fig("resultados/res_melhor_epoca.png", "Figura R15 — Distribuição da melhor época."),
    ]


def section_data_explain() -> list[str]:
    s = load_json(ANALYSIS_ROOT / "data_analysis.json")
    if not s:
        return []
    ac = s["autocorrelation"]
    lag1 = np.mean([v["lag_1"] for v in ac["per_user"].values()])
    lag8 = np.mean([v["lag_8"] for v in ac["per_user"].values()])
    drift = s["temporal_drift"]
    sep = s["separability"]
    svc = s["subject_vs_class"]

    drift_rows = [[u, num(v["eta2_class_median"] * 100, 1) + "%", num(v["eta2_block_not_class_median"] * 100, 1) + "%"]
                  for u, v in drift.items()]
    sep_rows = [[u, f"{v['n_significant_p05']}/25", f"{v['n_significant_asym_p05']}/10",
                 ", ".join(f"`{t['feature']}`" for t in v["top_features"][:3])]
                for u, v in sep.items() if not u.startswith("_")]

    return [
        "## Explicabilidade I — o que o dataset oferece (independente de modelo)",
        "",
        "Antes de olhar para as redes, vale perguntar o que o próprio sinal permite. As análises "
        "abaixo usam o log da potência e, nos testes estatísticos, **médias por trial** (36 por "
        "sujeito) em vez de amostras individuais, já que amostras vizinhas não são independentes.",
        "",
        "### D1. Amostras vizinhas são quase cópias",
        "",
        f"A autocorrelação média das features é {num(lag1)} entre amostras consecutivas e ainda "
        f"{num(lag8)} a 8 amostras de distância. Um trial dura ~{ac['median_block_length']} amostras. "
        "Isso explica por que o split aleatório da v1 inflava a acurácia: quase toda amostra de "
        "teste tinha uma \"gêmea\" no treino. Também justifica o *purge* de 40 amostras na CV.",
        "",
        *fig("dados/dados_01_autocorrelacao.png", "Figura D1 — Autocorrelação das features em função da distância."),
        "### D2. O sinal muda mais de um trial para o outro do que entre classes",
        "",
        "Para cada feature, mede-se a fração da variância explicada pela classe (η² classe) e a "
        "fração explicada pelo trial, descontada a classe (η² trial além da classe), isto é, a "
        "variação de um trial para outro da mesma classe. Valores medianos entre as 25 features:",
        "",
        *table(["Sujeito", "η² classe", "η² trial além da classe"], drift_rows),
        "**Este é o achado central para entender os resultados.** A variação entre trials que nada "
        "tem a ver com a classe (cansaço, atenção, contato dos eletrodos, deriva do equipamento) é "
        "de 10 a 60 vezes maior que a diferença entre Esquerda, Direita e Neutro. Uma rede treinada "
        "nesse sinal aprende facilmente a reconhecer *trials* (o que explica a acurácia de treino "
        "subindo e a melhor época ser a 1ª), mas o que distingue as *classes* é um efeito pequeno, "
        "enterrado sob essa variação.",
        "",
        *fig("dados/dados_02_deriva_temporal.png",
             "Figura D2 — Nível médio do sinal em cada trial, colorido pela classe do trial."),
        "### D3. Onde existe informação de classe",
        "",
        "ANOVA de 1 via entre as três classes, sobre as médias por trial (12 trials por classe), "
        "para cada canal × banda e para a assimetria hemisférica log(esquerdo) − log(direito). "
        "Sem correção para múltiplas comparações: com 25 testes, ~1,25 resultado \"significativo\" "
        "é esperado só pelo acaso.",
        "",
        *table(["Sujeito", "Canal×banda com p<0,05", "Assimetrias com p<0,05", "Features mais discriminativas"], sep_rows),
        "Os extremos coincidem com o desempenho dos modelos: `user_c`, com 6 features "
        "significativas, é o único sujeito acima do acaso em todos os protocolos, e `user_d`, sem "
        "nenhuma feature canal × banda significativa (a única assimetria marcada é compatível com o "
        "acaso), fica no acaso em todos. Isso dá uma explicação **independente de modelo** para o "
        "fenômeno de *BCI illiteracy* observado na v1: para esse sujeito, o problema não está nas "
        "redes, está na ausência de sinal discriminativo nas features disponíveis. Para `user_a`, "
        "`user_b` e `user_e` a relação é mais fraca. Uma ANOVA sobre a sessão inteira não detecta "
        "um padrão que só existe em parte dela, e é isso que a CV sugere (por exemplo, `user_b` "
        "passa de 50% nos primeiros trials e fica no acaso no fim). A assimetria hemisférica — o "
        "marcador clássico de imaginação motora esquerda × direita — aparece entre as features mais "
        "informativas de `user_a` e `user_e`, embora em BetaH/Gamma, e não nas bandas Alpha/Mu "
        "típicas da imaginação motora.",
        "",
        *fig("dados/dados_03_separabilidade.png",
             "Figura D3 — Significância da diferença entre classes por canal × banda e por assimetria."),
        *fig("dados/dados_05_espectro_por_classe.png",
             "Figura D4 — Espectro médio por classe, canal e sujeito."),
        "### D4. Os dados se agrupam por sujeito, não por classe",
        "",
        f"Com normalização global, a silhueta por sujeito é {num(svc['global']['silhouette_by_subject'])} "
        f"e a silhueta por classe é {num(svc['global']['silhouette_by_class'])}. Ou seja, há "
        "grupos claros por pessoa e nenhuma estrutura por classe. Após normalizar cada sujeito "
        f"separadamente, a silhueta por sujeito cai para {num(svc['por_sujeito']['silhouette_by_subject'])}: "
        "a normalização por sujeito apaga a identidade da pessoa. Isso explica por que ela é a "
        "única mudança com ganho significativo no modelo agrupado: sem ela, a rede gasta parte da "
        "capacidade separando sujeitos. Note, porém, que a silhueta por classe continua ~0 nos dois "
        "casos — remover a identidade do sujeito não faz as classes aparecerem.",
        "",
        *fig("dados/dados_04_tsne_global.png", "Figura D5 — t-SNE com normalização global, por sujeito e por classe."),
        *fig("dados/dados_04_tsne_por_sujeito.png", "Figura D6 — t-SNE com normalização por sujeito."),
    ]


def section_model_explain() -> list[str]:
    s = load_json(ANALYSIS_ROOT / "model_explain.json")
    if not s:
        return []
    arch = DISPLAY.get(s.get("architecture", ""), s.get("architecture", ""))
    variant = VARIANT_LABELS.get(s.get("personalized_variant", ""), "")
    lines = [
        "## Explicabilidade II — o que os modelos usam para decidir",
        "",
        f"As análises desta seção usam os modelos **personalizados** com *{variant}*, treinados no "
        f"hold-out cronológico (semente 42). A arquitetura analisada em detalhe é a **{arch}**, a "
        "de melhor desempenho médio na CV.",
        "",
    ]

    perm = s.get("permutation", {})
    if perm:
        rows = []
        for user, res in perm.items():
            test = res["test"]
            train = res["train"]
            top_c = max(test["by_channel"], key=test["by_channel"].get)
            top_b = max(test["by_band"], key=test["by_band"].get)
            rows.append([user, pct(test["baseline_balanced_accuracy"]), pct(train["baseline_balanced_accuracy"]),
                         f"{top_c} ({num(test['by_channel'][top_c] * 100, 1)} pp)",
                         f"{top_b} ({num(test['by_band'][top_b] * 100, 1)} pp)"])
        lines += [
            "### M1. Importância por permutação",
            "",
            "Mede quanto a acurácia balanceada cai quando os valores de um canal × banda são "
            "embaralhados entre as amostras, o que destrói a informação daquela feature e mantém o "
            "resto intacto. Calculada no **treino** (o que o modelo aprendeu a usar) e no **teste** "
            "(o que de fato generaliza). A diferença entre as duas figuras é a própria definição de "
            "overfitting: features importantes no treino e irrelevantes no teste foram decoradas.",
            "",
            *table(["Sujeito", "Acur. bal. teste", "Acur. bal. treino", "Canal mais importante (teste)",
                    "Banda mais importante (teste)"], rows),
            *fig("explicabilidade/exp_01_permutacao_teste.png", "Figura M1 — Importância por permutação no teste."),
            *fig("explicabilidade/exp_02_permutacao_treino.png", "Figura M2 — Importância por permutação no treino."),
        ]

    lines += [
        "### M2. Integrated Gradients por classe",
        "",
        "Para cada classe, mostra quais canais × bandas empurram a probabilidade daquela classe "
        "para cima (azul) ou para baixo (vermelho), em média sobre as amostras de teste da classe. "
        "O método satisfaz a propriedade de completude: a soma das atribuições é igual à diferença "
        "entre a probabilidade prevista e a probabilidade no ponto de referência (a média do "
        "treino). Isso foi verificado numericamente na implementação.",
        "",
        *fig("explicabilidade/exp_03_integrated_gradients.png", "Figura M3 — Integrated Gradients por classe e sujeito."),
        "### M3. Atenção do Transformer",
        "",
        "O Transformer decide a classe pelo token [CLS]. A figura mostra quanto esse token olha "
        "para cada eletrodo, em cada camada, separado pela classe verdadeira. 20% significa "
        "atenção uniforme. Atenção indica onde o modelo busca informação, não necessariamente o "
        "que é decisivo; por isso ela é apresentada junto da permutação e do IG.",
        "",
        *fig("explicabilidade/exp_04_atencao_transformer.png", "Figura M4 — Atenção [CLS] → canal."),
    ]

    emb = s.get("embeddings", {})
    if emb:
        rows = [[name, num(v["silhouette_by_class"]), num(v["silhouette_by_subject"])] for name, v in emb.items()]
        lines += [
            "### M4. Representação interna do modelo agrupado",
            "",
            "t-SNE da penúltima camada do modelo agrupado nas amostras de teste. Uma boa "
            "representação para a tarefa agrupa por classe, não por sujeito.",
            "",
            *table(["Modelo", "Silhueta por classe", "Silhueta por sujeito"], rows),
            *fig("explicabilidade/exp_05_embeddings_agrupado.png", "Figura M5 — Embeddings do modelo agrupado."),
        ]

    lines += [
        "### M5. O que o modelo prevê ao longo do tempo",
        "",
        "A figura mais direta para entender o comportamento: a probabilidade de cada classe ao "
        "longo do trecho de teste, com o rótulo verdadeiro na faixa superior. Um bom modelo "
        "mostra a curva da classe correta subindo dentro de cada trial. Um modelo no acaso mostra "
        "três curvas emboladas perto de 33%, ou uma classe dominando independentemente do trial.",
        "",
        *fig("explicabilidade/exp_06_timeline_predicoes.png", "Figura M6 — Timeline de predições no teste."),
    ]

    lines += [
        "### Síntese da explicabilidade: três métodos, a mesma resposta",
        "",
        "Para `user_c`, o sujeito com melhor desempenho, três métodos independentes convergem: a "
        "ANOVA nos dados brutos aponta **T8·Gamma** como a feature mais discriminativa (Figura D3); "
        "a importância por permutação do Bi-GRU mostra que embaralhar T8·Gamma derruba a acurácia "
        "em ~16 pp, mais que qualquer outra feature (Figura M1); e o Transformer dedica ~44% da "
        "atenção do [CLS] ao eletrodo T8 (Figura M4). Para `user_a`, a importância é mais difusa: "
        "nenhuma feature isolada passa de ~3 pp, mas embaralhar a banda Gamma inteira custa ~12 pp. Os modelos não são caixas-pretas aleatórias: eles encontram e usam "
        "o sinal que existe.",
        "",
        "Para `user_b` e `user_d`, o contraste entre treino e teste é a assinatura do "
        "overfitting: as mesmas redes acertam 62–72% (acurácia balanceada) no treino e ficam no "
        "acaso no teste do hold-out, com importâncias de teste próximas de zero ou negativas. O caso de "
        "`user_b` também ilustra por que atenção não é explicação: o Transformer concentra a "
        "atenção da 2ª camada em T7 e mesmo assim fica no acaso.",
        "",
        "> **Ressalva importante — possível artefato muscular.** A banda Gamma (acima de ~30 Hz) "
        "em eletrodos temporais (T7, T8) e frontais (AF3, AF4) é a região do espectro mais "
        "contaminada por atividade elétrica muscular (EMG) — contração da mandíbula, dos músculos "
        "temporais e da face. A imaginação motora clássica se manifesta em Alpha/Mu e Beta sobre o "
        "córtex sensório-motor (C3/C4), eletrodos que o EMOTIV Insight não possui. Portanto, a "
        "hipótese de que parte do desempenho de `user_c` e `user_a` venha de tensão muscular "
        "associada à tarefa — e não de atividade cortical — não pode ser descartada com estes "
        "dados. Isso não invalida o resultado para uma interface prática, mas muda o que ele "
        "significa e deve ser declarado na monografia.",
        "",
    ]

    cal = s.get("calibration", {})
    if cal:
        rows = [[DISPLAY[m], pct(c["mean_confidence"]), pct(c["accuracy_overall"]), num(c["ece"], 3)]
                for m, c in cal.items()]
        lines += [
            "### M6. Calibração — o modelo sabe quando está errando?",
            "",
            "Compara a confiança média do modelo com a acurácia real. O ECE (*expected calibration "
            "error*) resume a distância entre as duas. Em BCI isso importa: um modelo calibrado "
            "permite que a interface fique em \"neutro\" quando a confiança é baixa, em vez de "
            "executar um comando errado.",
            "",
            *table(["Modelo", "Confiança média", "Acurácia", "ECE"], rows),
            "Todos os modelos são **excessivamente confiantes**: a confiança média supera a "
            "acurácia real, reflexo do overfitting precoce. A CNN é a mais bem calibrada. Para uso "
            "prático, uma calibração pós-treino (*temperature scaling*) com a validação seria "
            "um passo barato.",
            "",
            *fig("explicabilidade/exp_07_calibracao.png", "Figura M7 — Diagrama de calibração."),
        ]
    return lines


def section_limitations() -> list[str]:
    return [
        "## Limitações",
        "",
        "1. **Poucos trials independentes.** Mesmo na CV, cada sujeito tem 36 trials, ou 12 por "
        "classe. Um trial a mais ou a menos acertado move a acurácia por trial em ~2,8 pp.",
        "2. **Os dois protocolos medem cenários diferentes.** Na CV por blocos, o modelo às vezes "
        "é treinado com trials gravados depois do teste (padrão em avaliação offline de BCI; o "
        "purge remove a contaminação por vizinhança), e no fold 5 a validação fica longe do teste. "
        "O hold-out simula o uso real (calibrar antes, usar logo depois), mas avalia só ~7 trials "
        "por sujeito. Como mostra o diagnóstico, a diferença entre eles é grande para os sujeitos "
        "com sinal. Por isso os dois são reportados, e nenhuma conclusão deste documento se apoia "
        "em só um deles.",
        "3. **A normalização por sujeito exige calibração.** Para um usuário novo, seria preciso "
        "coletar alguns trials antes do uso para estimar média e desvio. É uma exigência comum em "
        "BCI, mas deve ser declarada.",
        "4. **A seleção da \"melhor variante\" usa os mesmos dados que a avaliam.** Com 3–4 "
        "variantes, o viés de seleção é pequeno, mas existe. A tabela de ablação mostra todas as "
        "variantes, não só a vencedora.",
        "5. **Explicabilidade é local ao modelo e ao sujeito.** Permutação e IG descrevem o que um "
        "modelo específico usa, não uma verdade neurofisiológica. Onde os modelos estão no acaso, "
        "os mapas de importância são ruído — o que, por si só, é um resultado.",
        "",
    ]


def section_next() -> list[str]:
    return [
        "## Próximos passos (v3) — frentes congeladas nesta iteração",
        "",
        "Por decisão de escopo, a v2 mexeu **apenas na representação dos dados**. Ficam registradas, "
        "para a próxima iteração e cada uma medida isoladamente:",
        "",
        "1. **Janelas temporais causais**: contexto de N amostras passadas, sem cruzar trials, mais "
        "suavização das predições. É o que daria sentido real às arquiteturas sequenciais (LSTM, "
        "GRU, Transformer), que hoje tratam bandas ou canais como \"tempo\". Os resultados por "
        "trial (Figura R12) indicam o tamanho do ganho potencial.",
        "2. **Modelos clássicos de BCI** (LDA com shrinkage, Regressão Logística, SVM, Random "
        "Forest) como linha de base e para explicabilidade direta pelos coeficientes.",
        "3. **Ajuste das redes**: mais regularização, label smoothing, data augmentation e busca de "
        "hiperparâmetros feita só na validação. É a resposta direta ao sintoma \"melhor época ≈ 1\".",
        "4. **Features de assimetria explícitas** como entrada adicional, dado o seu destaque na "
        "Figura D3. Isso exige mudar a entrada das redes além da grade 5×5.",
        "",
    ]


def main() -> None:
    df = collect_all()
    d = Data(df)
    lines = []
    for section in [
        section_header(d), section_summary(d), section_changes(), section_equivalence(d),
        section_chronological(d), section_cv(d), section_diagnostic(), section_stats(d), section_epochs(d),
        section_data_explain(), section_model_explain(), section_limitations(), section_next(),
    ]:
        lines += section
        if section and lines[-1] != "---":
            lines += ["---", ""]
    OUTPUT_PATH.write_text("\n".join(lines).rstrip("-\n ") + "\n", encoding="utf-8")
    print(f"[ok] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
