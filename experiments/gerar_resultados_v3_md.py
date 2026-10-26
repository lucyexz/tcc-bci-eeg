"""Gera docs/resultados_v3.md — LOSO e LOTO do modelo agrupado.

Lê a tabela longa de `_v3_results.collect_all()` (v3 + agrupado da v2) e
escreve um documento em português pronto para a monografia. Não treina nada;
pode ser reexecutado à vontade. Os relatórios da v1 e da v2 não são tocados.

Uso:
    python experiments/gerar_resultados_v3_md.py
"""

from datetime import datetime

import pandas as pd
from scipy import stats

from _v3_results import MODELS, PROJECT_ROOT, USERS, collect_all
from gerar_resultados_md import DISPLAY, pct, points
from gerar_resultados_v2_md import POOLED_VARIANTS, VARIANT_LABELS, num, pm, table

FIG_REL = "../outputs/iterations/v3/figures/resultados"
OUTPUT_PATH = PROJECT_ROOT / "docs" / "resultados_v3.md"

METRIC = "balanced_accuracy"
CHANCE = 1 / 3

PROTOCOLS = ["chronological", "blocked_cv", "loto", "loso"]
PROTOCOL_LABELS = {
    "chronological": "Hold-out cronológico",
    "blocked_cv": "CV por blocos (6 folds)",
    "loto": "LOTO (12 folds)",
    "loso": "LOSO",
}


def fig(name: str, caption: str) -> list[str]:
    return [f"![{caption}]({FIG_REL}/{name})", f"*{caption}*", ""]


class Data:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def subj(self, protocol, variants=POOLED_VARIANTS):
        """Média por (variante, modelo, sujeito) sobre as repetições."""
        sub = self.df[(self.df.protocol == protocol) & (self.df.subject != "ALL")
                      & self.df.variant.isin(variants)]
        return sub.groupby(["variant", "model", "subject"], as_index=False).mean(numeric_only=True)

    def has(self, protocol):
        return bool((self.df.protocol == protocol).any())

    def best_variant(self, protocol):
        sl = self.subj(protocol)
        return sl.groupby("variant")[METRIC].mean().idxmax() if not sl.empty else None

    def reps(self, protocol):
        return sorted(self.df[self.df.protocol == protocol].rep.unique())


def variant_model_table(d: Data, protocol: str) -> list[str]:
    sl = d.subj(protocol)
    rows = []
    for v in POOLED_VARIANTS:
        sv = sl[sl.variant == v]
        if sv.empty:
            continue
        rows.append([VARIANT_LABELS[v]] + [pm(sv[sv.model == m][METRIC]) for m in MODELS]
                    + [f"**{pct(sv[METRIC].mean())}**"])
    return table(["Variante"] + [DISPLAY[m] for m in MODELS] + ["Média"], rows)


def subject_table(d: Data, protocol: str) -> list[str]:
    sl = d.subj(protocol)
    rows = []
    for user in USERS:
        su = sl[sl.subject == user]
        if su.empty:
            continue
        row = [user]
        for v in POOLED_VARIANTS:
            row.append(pct(su[su.variant == v][METRIC].mean()))
        best = su.loc[su[METRIC].idxmax()]
        row.append(f"{DISPLAY[best.model]}, {VARIANT_LABELS[best.variant]}: {pct(best[METRIC])}")
        rows.append(row)
    return table(["Sujeito"] + [VARIANT_LABELS[v] for v in POOLED_VARIANTS]
                 + ["Melhor combinação"], rows)


# ---------------------------------------------------------------------------
# Seções
# ---------------------------------------------------------------------------

def section_header() -> list[str]:
    now = datetime.now().strftime("%d/%m/%Y às %H:%M")
    return [
        "# Resultados v3 — LOSO e LOTO para o modelo agrupado (generalista)",
        "",
        f"*Documento gerado automaticamente por `experiments/gerar_resultados_v3_md.py` em {now}. "
        "As iterações anteriores estão congeladas em `outputs/iterations/v1/` e "
        "`outputs/iterations/v2/` (SHA-256 de cada arquivo em `MANIFEST.json`); "
        "`docs/resultados_tcc2.md` e `docs/resultados_v2.md` permanecem inalterados.*",
        "",
    ]


def section_summary(d: Data) -> list[str]:
    lines = [
        "## Resumo",
        "",
        "Nas iterações anteriores, o modelo agrupado foi sempre avaliado em sujeitos que também "
        "estavam no treino. A v3 mantém features, arquiteturas e hiperparâmetros da v2 e muda "
        "**apenas o protocolo de avaliação do modelo agrupado**, com dois protocolos clássicos de "
        "BCI: **LOSO** (*leave-one-subject-out*: o modelo serve para uma pessoa nova?) e **LOTO** "
        "(*leave-one-trial-out*: o modelo serve para um trial novo de pessoas conhecidas?). Os "
        "modelos personalizados não entram: com um único sujeito não existe sujeito a deixar de fora.",
        "",
        "Acurácia balanceada média (4 modelos, 5 sujeitos); o acaso é 33,3%:",
        "",
    ]
    rows = []
    for v in POOLED_VARIANTS:
        row = [VARIANT_LABELS[v]]
        for protocol in PROTOCOLS:
            sl = d.subj(protocol, [v])
            row.append(pct(sl[METRIC].mean()) if not sl.empty else "—")
        rows.append(row)
    lines += table(["Variante"] + [PROTOCOL_LABELS[p] for p in PROTOCOLS], rows)
    lines += fig("res_protocolos.png",
                 "Figura P1 — O mesmo modelo agrupado sob os quatro protocolos de avaliação.")

    loso, loto = d.subj("loso"), d.subj("loto")
    if loso.empty or loto.empty:
        return lines
    loso_by_variant = loso.groupby("variant")[METRIC].mean()
    loto_by_variant = loto.groupby("variant")[METRIC].mean()
    loto_by_subject = loto.groupby("subject")[METRIC].mean()
    top = loto_by_subject.idxmax()
    lines += [
        "**Leitura honesta dos números:**",
        "",
        f"1. **Para uma pessoa nova, o modelo agrupado fica praticamente no acaso.** No LOSO as "
        f"quatro variantes ficam entre {pct(loso_by_variant.min())} e {pct(loso_by_variant.max())}, "
        f"ou seja, no máximo ~{points(loso_by_variant.max() - CHANCE).lstrip('+')} acima de 33,3%. Sem "
        "calibração (*raw* e *log*) não há diferença detectável para o acaso. Com calibração não "
        "supervisionada o resultado fica acima do acaso e também acima do *raw* (Wilcoxon, "
        "p < 0,05, ganho de ~2 pp), e esse ganho se repete nas três sementes; é o único efeito "
        "de feature estatisticamente significativo da v3, mas ainda pequeno demais para uma "
        "interface utilizável.",
        f"2. **Com o sujeito no treino, o ganho é modesto e concentrado em `{top}`.** No LOTO as "
        f"variantes ficam entre {pct(loto_by_variant.min())} e {pct(loto_by_variant.max())}, e "
        f"`{top}` chega a {pct(loto_by_subject.max())} na média das variantes, enquanto os demais "
        "ficam poucos pontos acima do acaso. É o mesmo padrão da v2: o que o modelo agrupado "
        "aprende é, em boa parte, específico de um sujeito.",
        "3. **O LOTO confirma a CV por blocos da v2, não o hold-out.** Com 12 folds e mais dados "
        "de treino, os números ficam ~1 pp acima da CV de 6 folds e bem abaixo dos ~42% do "
        "hold-out cronológico com normalização por sujeito, reforçando o diagnóstico da v2 de que "
        "aquele ganho dependia de a validação estar colada no teste. No LOTO a normalização por "
        "sujeito não acrescenta nada sobre o *log* sozinho, e nenhuma variante é "
        "significativamente melhor que o *raw*.",
        "4. **O sintoma \"melhor época ≈ 1\" continua.** Em 58% a 100% dos treinos a menor perda "
        "de validação acontece até a 2ª época, nos dois protocolos: as redes não aprendem um "
        "padrão de classe que se transfira, nem entre trials nem entre pessoas.",
        "",
        "Sementes: o LOSO foi rodado com as sementes 42, 43 e 44 e cada valor é a média das três. "
        "A média por variante varia no máximo ~1 pp entre sementes; as diferenças entre "
        "arquiteturas (~1–2 pp) continuam dentro do ruído e não devem ser interpretadas.",
        "",
    ]
    return lines


def section_changes() -> list[str]:
    return [
        "## O que mudou da v2 para a v3",
        "",
        "Só o protocolo de avaliação. As quatro variantes de features do agrupado, as quatro "
        "arquiteturas e todos os hiperparâmetros são os da v2.",
        "",
        "### LOSO — *leave-one-subject-out*",
        "",
        "Cinco rodadas. Em cada uma, **um sujeito inteiro é o teste e nunca aparece no treino**. "
        "Outro sujeito inteiro (o seguinte, de forma circular) é a validação usada pelo "
        "EarlyStopping, de modo que os pesos sejam escolhidos pelo que generaliza para uma pessoa "
        "não vista; os três restantes são o treino. Todos os 36 trials do sujeito de teste são "
        "avaliados.",
        "",
        "Como o sujeito de teste não tem dados de treino, a normalização tem dois cenários, e os "
        "dois são reportados:",
        "",
        "- **Sem calibração** (variantes *raw* e *log*): IQR e StandardScaler ajustados só nos "
        "sujeitos de treino. É o usuário que coloca o headset e usa, sem nenhuma coleta prévia.",
        "- **Calibração não supervisionada** (variantes *+ norm. por sujeito*): média, desvio e "
        "limites de IQR do sujeito de teste são estimados **nas próprias amostras dele, sem usar "
        "os rótulos**. Corresponde a um usuário novo que fornece alguns minutos de sinal sem "
        "precisar executar a tarefa rotulada. Nenhum rótulo de teste é usado, mas as estatísticas "
        "vêm da sessão inteira de teste (normalização transdutiva), o que é mais favorável do que "
        "estimar só com os primeiros minutos.",
        "",
        "### LOTO — *leave-one-trial-out*",
        "",
        "Doze rodadas. Em cada uma, **um ciclo Esquerda → Direita → Neutro (um trial de cada "
        "classe) de cada sujeito é o teste**; o ciclo seguinte é a validação e os outros dez são "
        "treino. Deixar o ciclo inteiro de fora, em vez de um único trial, mantém as três classes "
        "no teste de cada rodada. Como na CV da v2, 40 amostras são descartadas em cada fronteira "
        "com o teste (*purge*). É a mesma validação cruzada por blocos da v2 com 12 folds em vez "
        "de 6: mais dados de treino por rodada (10/12 contra 4/6) e todos os 36 trials de cada "
        "sujeito testados exatamente uma vez.",
        "",
        "### Guardrails",
        "",
        "Antes de cada treino o script confere: no LOSO, que os sujeitos de treino, validação e "
        "teste são disjuntos; no LOTO, que cada sujeito tem exatamente um trial de cada classe no "
        "teste e que o purge foi respeitado; em ambos, que as partições somam o total de linhas. "
        "O pré-processamento dos protocolos da v2 foi verificado como idêntico bit a bit ao de "
        "antes da mudança.",
        "",
    ]


def section_loso(d: Data) -> list[str]:
    if not d.has("loso"):
        return []
    seeds = [r.replace("seed_", "") for r in d.reps("loso")]
    return [
        "## Resultados — LOSO (pessoa nova)",
        "",
        "Cada célula é a acurácia balanceada média entre os 5 sujeitos de teste ± desvio entre "
        f"sujeitos. Semente(s): {', '.join(seeds)}.",
        "",
        *variant_model_table(d, "loso"),
        *fig("res_ablacao_loso.png", "Figura L1 — Efeito das variantes no LOSO."),
        "### Por sujeito de teste",
        "",
        "Média dos 4 modelos em cada variante. As duas primeiras colunas são o cenário sem "
        "calibração; as duas seguintes, com calibração não supervisionada.",
        "",
        *subject_table(d, "loso"),
        *fig("res_por_sujeito_loso.png", "Figura L2 — LOSO por sujeito de teste."),
        *fig("res_matrizes_confusao_loso.png", "Figura L3 — Matrizes de confusão no LOSO."),
    ]


def section_loto(d: Data) -> list[str]:
    if not d.has("loto"):
        return []
    return [
        "## Resultados — LOTO (trial novo de pessoas conhecidas)",
        "",
        "Cada célula é a acurácia balanceada média entre os 5 sujeitos (cada sujeito resumido "
        "pela média dos seus 12 folds) ± desvio entre sujeitos.",
        "",
        *variant_model_table(d, "loto"),
        *fig("res_ablacao_loto.png", "Figura T1 — Efeito das variantes no LOTO."),
        "### Por sujeito",
        "",
        *subject_table(d, "loto"),
        *fig("res_por_sujeito_loto.png", "Figura T2 — LOTO por sujeito."),
        *fig("res_matrizes_confusao_loto.png", "Figura T3 — Matrizes de confusão no LOTO."),
    ]


def section_gap(d: Data) -> list[str]:
    if not (d.has("loso") and d.has("loto")):
        return []
    variant = d.best_variant("loso")
    seen = d.subj("loto", [variant]).groupby("subject")[METRIC].mean()
    unseen = d.subj("loso", [variant]).groupby("subject")[METRIC].mean()
    rows = [[u, pct(seen.get(u)), pct(unseen.get(u)), points(unseen.get(u) - seen.get(u))]
            for u in USERS if u in seen.index and u in unseen.index]
    rows.append(["**Média**", f"**{pct(seen.mean())}**", f"**{pct(unseen.mean())}**",
                 f"**{points(unseen.mean() - seen.mean())}**"])
    return [
        "## O custo de ser um usuário novo",
        "",
        f"Mesma variante (*{VARIANT_LABELS[variant]}*, a melhor no LOSO) e mesmas redes; muda só "
        "se o sujeito avaliado estava no treino (LOTO) ou não (LOSO). Média dos 4 modelos.",
        "",
        *table(["Sujeito", "No treino (LOTO)", "Fora do treino (LOSO)", "Diferença"], rows),
        *fig("res_loso_vs_loto.png", "Figura P2 — Sujeito no treino × fora do treino."),
    ]


def section_stats(d: Data) -> list[str]:
    rows = []
    for protocol in ["loso", "loto"]:
        sl = d.subj(protocol)
        if sl.empty:
            continue
        raw = sl[sl.variant == "raw"].set_index(["model", "subject"])[METRIC]
        for v in POOLED_VARIANTS[1:]:
            new = sl[sl.variant == v].set_index(["model", "subject"])[METRIC]
            pairs = pd.concat([raw, new], axis=1, keys=["raw", "new"]).dropna()
            if len(pairs) < 6:
                continue
            diff = pairs.new - pairs.raw
            p = stats.wilcoxon(pairs.new, pairs.raw).pvalue if (diff != 0).any() else 1.0
            rows.append([PROTOCOL_LABELS[protocol], VARIANT_LABELS[v], str(len(pairs)),
                         points(diff.mean()), f"{int((diff > 0).sum())}/{len(pairs)}",
                         f"{p:.4f}".replace(".", ","), "**sim**" if p < 0.05 else "não"])

    # Acima do acaso? Teste de sinal unilateral sobre os pares sujeito × modelo.
    chance_rows = []
    for protocol in ["loso", "loto"]:
        sl = d.subj(protocol)
        for v in POOLED_VARIANTS:
            values = sl[sl.variant == v][METRIC]
            if len(values) < 6:
                continue
            p = stats.wilcoxon(values - CHANCE, alternative="greater").pvalue
            chance_rows.append([PROTOCOL_LABELS[protocol], VARIANT_LABELS[v], pct(values.mean()),
                                f"{int((values > CHANCE).sum())}/{len(values)}",
                                f"{p:.4f}".replace(".", ","), "**sim**" if p < 0.05 else "não"])
    if not rows and not chance_rows:
        return []
    return [
        "## Significância estatística",
        "",
        "Cada par é uma combinação sujeito × arquitetura (20 pares), resumida pela média das "
        "repetições. Os 4 modelos de um mesmo sujeito compartilham os dados, então os p-valores "
        "devem ser lidos como indicativos.",
        "",
        "**O modelo fica acima do acaso?** Wilcoxon unilateral contra 33,3%:",
        "",
        *table(["Protocolo", "Variante", "Acur. bal. média", "Pares acima do acaso",
                "p-valor", "p < 0,05?"], chance_rows),
        "**A variante melhora em relação ao raw?** Wilcoxon pareado bicaudal:",
        "",
        *table(["Protocolo", "Variante", "Pares", "Ganho médio", "Pares com ganho",
                "p-valor", "p < 0,05?"], rows),
    ]


def section_epochs(d: Data) -> list[str]:
    # Uma linha por treino: no LOTO, a linha agregada (as por sujeito repetem a época).
    sub = d.df[(d.df.iteration == "v3") & d.df.best_epoch.notna()]
    sub = sub[(sub.protocol == "loso") | (sub.subject == "ALL")]
    if sub.empty:
        return []
    rows = [[PROTOCOL_LABELS[p], VARIANT_LABELS[v], num(g.best_epoch.median(), 0),
             pct((g.best_epoch <= 2).mean())]
            for (p, v), g in sub.groupby(["protocol", "variant"])]
    return [
        "## Melhor época",
        "",
        "Época em que a perda de validação foi mínima. No LOSO a validação é um sujeito não visto "
        "no treino; no LOTO, um ciclo vizinho ao teste dos mesmos sujeitos.",
        "",
        *table(["Protocolo", "Variante", "Mediana da melhor época",
                "Treinos com melhor época ≤ 2"], rows),
    ]


def section_limitations() -> list[str]:
    return [
        "## Limitações",
        "",
        "1. **LOSO com 5 sujeitos.** Cada rodada treina com apenas 3 pessoas e testa em 1. É "
        "pouco para aprender o que é comum entre pessoas, e a média de 5 valores tem variância "
        "alta: um único sujeito move a média em vários pontos.",
        "2. **A calibração não supervisionada usa a sessão de teste inteira.** As variantes com "
        "normalização por sujeito estimam média e desvio do usuário novo em todas as amostras "
        "dele (sem rótulos). Num uso real essas estatísticas viriam só dos primeiros minutos; o "
        "resultado reportado é, portanto, um limite superior para esse cenário.",
        "3. **A validação do LOSO é um único sujeito.** O ponto de parada do treino depende de "
        "qual pessoa caiu na validação; com outra escolha os números mudariam dentro do ruído "
        "entre sementes.",
        "4. **No LOTO a validação é um ciclo de 3 trials por sujeito**, bem menor que na CV de 6 "
        "folds, o que deixa o EarlyStopping mais ruidoso. Em compensação o treino é maior.",
        "5. **LOTO não é teste de pessoa nova.** O sujeito avaliado está no treino, inclusive com "
        "trials gravados depois do trecho de teste. Ele mede generalização entre trials, não "
        "entre pessoas.",
        "",
    ]


def main() -> None:
    df = collect_all()
    if df.empty or not (df.iteration == "v3").any():
        raise SystemExit("Nenhum resultado da v3 encontrado.")
    d = Data(df)
    lines = []
    for section in [
        section_header(), section_summary(d), section_changes(), section_loso(d),
        section_loto(d), section_gap(d), section_stats(d), section_epochs(d),
        section_limitations(),
    ]:
        lines += section
        if section and lines[-1] != "---":
            lines += ["---", ""]
    OUTPUT_PATH.write_text("\n".join(lines).rstrip("-\n ") + "\n", encoding="utf-8")
    print(f"[ok] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
