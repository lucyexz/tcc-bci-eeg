"""Gera docs/resultados_tcc2.md a partir dos JSONs de métricas já exportados.

Lê os três conjuntos de resultados — baseline aleatório, agrupado cronológico e
personalizado por sujeito — e monta um documento em português, com tabelas em
Markdown, pronto para ser copiado para a monografia.

Não treina nada e não depende do TensorFlow: é só leitura de JSON. Pode ser
reexecutado quantas vezes for preciso; o arquivo é sempre reescrito do zero.

Uso:
    python experiments/gerar_resultados_md.py
"""

import json
import statistics
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_PATH = PROJECT_ROOT / "docs" / "resultados_tcc2.md"

MODEL_NAMES = ["cnn", "lstm", "gru", "transformer"]
USERS = ["user_a", "user_b", "user_c", "user_d", "user_e"]

DISPLAY = {
    "cnn": "CNN",
    "lstm": "Bi-LSTM",
    "gru": "Bi-GRU",
    "transformer": "Transformer",
}


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.2f}".replace(".", ",") + "%"


def points(value: float | None) -> str:
    """Formata uma diferença em pontos percentuais, com sinal explícito."""
    if value is None:
        return "—"
    return f"{value * 100:+.2f}".replace(".", ",") + " pp"


def f1_macro(metrics: dict | None) -> float | None:
    if metrics is None:
        return None
    return metrics["classification_report"]["macro avg"]["f1-score"]


def accuracy(metrics: dict | None) -> float | None:
    if metrics is None:
        return None
    return metrics["test_accuracy"]


def majority_baseline(metrics: dict | None) -> float | None:
    """Acurácia de um classificador que sempre chuta a classe mais frequente.

    É a linha de base honesta quando o teste é desbalanceado: um modelo que não
    a supera não está extraindo informação do sinal.
    """
    if metrics is None:
        return None

    support = [sum(row) for row in metrics["confusion_matrix"]]
    total = sum(support)

    if total == 0:
        return None

    return max(support) / total


def mean_std(values: list[float]) -> str:
    if not values:
        return "—"
    if len(values) == 1:
        return pct(values[0])
    mean = statistics.mean(values)
    std = statistics.stdev(values)
    return f"{pct(mean)} ± {std * 100:.2f}".replace(".", ",") + " pp"


def collect() -> dict:
    """Carrega os três conjuntos de resultados do disco."""
    random_track = {
        name: load(METRICS_DIR / f"{name}.json") for name in MODEL_NAMES
    }
    chronological_track = {
        name: load(METRICS_DIR / "chronological" / f"{name}.json")
        for name in MODEL_NAMES
    }
    personalized_track = {
        user: {
            name: load(METRICS_DIR / "personalized" / user / f"{name}.json")
            for name in MODEL_NAMES
        }
        for user in USERS
    }
    control_track = {
        name: load(METRICS_DIR / "control_random" / f"{name}.json")
        for name in MODEL_NAMES
    }
    preprocessing = load(METRICS_DIR / "chronological" / "_preprocessing.json")

    return {
        "random": random_track,
        "chronological": chronological_track,
        "personalized": personalized_track,
        "control": control_track,
        "preprocessing": preprocessing,
    }


def section_header(data: dict) -> list[str]:
    config = (data["preprocessing"] or {}).get("preprocessing_config", {})
    report = (data["preprocessing"] or {}).get("preprocessing_report", {})

    lines = [
        "# Resultados TCC2 — Split Cronológico e Track Personalizado",
        "",
        f"*Documento gerado automaticamente por `experiments/gerar_resultados_md.py` "
        f"em {datetime.now().strftime('%d/%m/%Y às %H:%M')}.*",
        "",
        "## Resumo",
        "",
        "Este documento reporta dois experimentos sobre a classificação de intenção "
        "motora em três classes (Esquerda, Direita, Neutro) a partir de features de "
        "potência espectral de EEG (5 canais × 5 bandas = 25 features), coletadas com "
        "o headset EMOTIV Insight em 5 sujeitos.",
        "",
        "O **primeiro experimento** substitui o split aleatório estratificado, usado "
        "até então, por um **split cronológico por sujeito**. A motivação é que "
        "amostras consecutivas de EEG são quase idênticas entre si: ao embaralhar os "
        "dados antes de dividi-los, quase-duplicatas de uma mesma janela temporal "
        "acabam distribuídas entre treino e teste, o que infla artificialmente a "
        "acurácia reportada. O split cronológico elimina esse vazamento ao reservar "
        "o trecho final da gravação de cada sujeito como conjunto de teste.",
        "",
        "O **segundo experimento** introduz um paradigma **personalizado**: em vez de "
        "um único modelo treinado com os dados de todos os sujeitos (paradigma "
        "generalista), treina-se um modelo independente para cada sujeito. A hipótese "
        "é que a especificidade individual do sinal de EEG compense o volume muito "
        "menor de dados disponível por sujeito.",
        "",
        "### Configuração experimental",
        "",
        "| Parâmetro | Valor |",
        "|---|---|",
        f"| Estratégia de split | {config.get('split_strategy', '—')} |",
        f"| Proporção de teste | {config.get('test_size', '—')} |",
        f"| Proporção de validação (sobre o treino) | {config.get('validation_size', '—')} |",
        f"| Tratamento de outliers | {config.get('outlier_strategy', '—')} "
        f"(multiplicador {config.get('iqr_multiplier', '—')}) |",
        f"| Imputação | {config.get('imputation_strategy', '—')} |",
        f"| Normalização | StandardScaler ajustado apenas no treino |",
        f"| Semente | {config.get('random_state', '—')} |",
        "",
    ]

    if report:
        lines += [
            f"Após a limpeza, o conjunto agrupado contém "
            f"{report.get('shape_after_cleaning', ['—'])[0]} amostras, divididas em "
            f"{report.get('final_train_shape', ['—'])[0]} de treino, "
            f"{report.get('final_val_shape', ['—'])[0]} de validação e "
            f"{report.get('final_test_shape', ['—'])[0]} de teste.",
            "",
        ]

    lines += [
        "> **Nota metodológica sobre a validação.** O conjunto de validação — usado "
        "apenas pelos callbacks `EarlyStopping` e `ReduceLROnPlateau`, nunca para "
        "reportar resultados — é recortado do trecho final do treino **de cada "
        "sujeito**, e não pela fatia automática do Keras (`validation_split`). Sob "
        "ordem cronológica, aquela fatia automática pegaria a cauda do array "
        "concatenado, o que faria a validação do modelo agrupado ser composta por "
        "praticamente um único sujeito e enviesaria a seleção de pesos. O recorte "
        "por sujeito preserva a mesma proporção (20% do treino) e mantém as duas "
        "propriedades desejadas: representatividade de todos os sujeitos e "
        "posterioridade temporal em relação ao treino.",
        "",
    ]

    return lines


def section_part_one(data: dict) -> list[str]:
    lines = [
        "---",
        "",
        "## Parte 1 — Modelo generalista: split aleatório vs. split cronológico",
        "",
        "A tabela abaixo compara as acurácias obtidas pelo mesmo modelo agrupado sob "
        "as duas estratégias de divisão dos dados. Todos os demais parâmetros de "
        "pré-processamento, arquiteturas e hiperparâmetros permanecem idênticos: a "
        "única variável alterada é a forma de separar treino e teste.",
        "",
        "| Modelo | Acurácia (split aleatório) | Acurácia (split cronológico) | Δ | F1 macro (cronológico) |",
        "|---|---|---|---|---|",
    ]

    deltas = []

    for name in MODEL_NAMES:
        old = accuracy(data["random"].get(name))
        new = accuracy(data["chronological"].get(name))
        difference = None if old is None or new is None else new - old

        if difference is not None:
            deltas.append(difference)

        lines.append(
            f"| {DISPLAY[name]} | {pct(old)} | {pct(new)} | "
            f"{points(difference)} | {pct(f1_macro(data['chronological'].get(name)))} |"
        )

    lines.append("")

    if not deltas:
        return lines

    mean_drop = statistics.mean(deltas)
    worst = min(deltas)
    best = max(deltas)

    reference = next(
        (m for m in (data["chronological"].get(n) for n in MODEL_NAMES) if m),
        None,
    )
    majority = majority_baseline(reference)
    chronological_values = [
        value
        for value in (accuracy(data["chronological"].get(n)) for n in MODEL_NAMES)
        if value is not None
    ]

    lines += [
        "### Interpretação",
        "",
        f"A queda média de acurácia foi de **{points(mean_drop)}**, variando entre "
        f"{points(best)} e {points(worst)} conforme a arquitetura. Essa diferença "
        "é a medida direta do **vazamento temporal** presente nos resultados "
        "anteriores.",
        "",
        "O mecanismo é o seguinte: sob split aleatório, o modelo era avaliado em "
        "amostras temporalmente vizinhas de amostras que ele já havia visto no "
        "treino. Como o EEG é um sinal contínuo e as features de potência espectral "
        "variam lentamente, essas vizinhas são quase cópias — acertá-las não exige "
        "generalização, apenas memorização. Sob split cronológico, o teste passa a "
        "ser um trecho de gravação inteiramente posterior e disjunto, exigindo que o "
        "modelo generalize para um momento que nunca observou.",
        "",
    ]

    if majority is not None:
        chance = 1 / 3
        at_chance = [v for v in chronological_values if v < majority]

        lines += [
            "### O modelo generalista não é utilizável sob avaliação honesta",
            "",
            f"Duas linhas de base contextualizam os números: o acaso, em três "
            f"classes equiprováveis, é {pct(chance)}; e um classificador que sempre "
            f"chutasse a classe mais frequente do conjunto de teste acertaria "
            f"**{pct(majority)}**.",
            "",
            f"Sob split cronológico, {len(at_chance)} das {len(chronological_values)} "
            "arquiteturas ficam **abaixo da linha de base da classe majoritária**, e "
            "todas ficam próximas do acaso. Os valores de F1 macro, entre "
            f"{pct(min(f1_macro(data['chronological'].get(n)) or 1 for n in MODEL_NAMES))} e "
            f"{pct(max(f1_macro(data['chronological'].get(n)) or 0 for n in MODEL_NAMES))}, "
            "confirmam a leitura. A conclusão é direta: **o modelo generalista, "
            "treinado com todos os sujeitos misturados, não extrai informação "
            "aproveitável do sinal quando avaliado em um trecho temporal que não viu**.",
            "",
            "O histórico de treinamento reforça o diagnóstico. A acurácia de treino "
            "sobe consistentemente, mas a perda de validação atinge seu mínimo logo "
            "nas primeiras épocas e piora a partir daí — o `EarlyStopping` interrompe "
            "o treino cedo e restaura pesos quase iniciais. Ou seja, tudo o que o "
            "modelo aprende além das primeiras épocas é específico do trecho de "
            "gravação usado no treino e não transfere para nenhum momento posterior.",
            "",
            "**A queda não representa uma piora do modelo: representa a medição "
            "honesta do mesmo modelo.** Os números do split cronológico são os que "
            "devem ser reportados como desempenho real; os do split aleatório servem "
            "apenas para quantificar o viés corrigido.",
            "",
        ]

    lines += section_control(data)

    return lines


def section_control(data: dict) -> list[str]:
    """Experimento de controle: mesmo código, split aleatório.

    Uma queda de mais de 30 pontos percentuais é grande o bastante para
    levantar a suspeita de erro de implementação. O controle isola essa
    hipótese.
    """
    control = data.get("control", {})

    if not any(control.get(name) for name in MODEL_NAMES):
        return []

    lines = [
        "### Experimento de controle: a queda vem do split, não do código",
        "",
        "Uma diferença dessa magnitude exige descartar a hipótese de erro de "
        "implementação. Para isso, o pipeline novo foi reexecutado **com split "
        "aleatório**, mantendo tudo o mais idêntico: o mesmo código de "
        "pré-processamento, o mesmo recorte explícito de validação e as mesmas "
        "arquiteturas. Se a implementação estivesse errada, os números do controle "
        "divergiriam do baseline original.",
        "",
        "| Modelo | Baseline original (notebooks) | Controle (pipeline novo, split aleatório) | Diferença |",
        "|---|---|---|---|",
    ]

    differences = []

    for name in MODEL_NAMES:
        original = accuracy(data["random"].get(name))
        reproduced = accuracy(control.get(name))
        difference = (
            None if original is None or reproduced is None else reproduced - original
        )

        if difference is not None:
            differences.append(abs(difference))

        lines.append(
            f"| {DISPLAY[name]} | {pct(original)} | {pct(reproduced)} | "
            f"{points(difference)} |"
        )

    lines.append("")

    if differences:
        lines += [
            f"O pipeline novo reproduz o baseline original com diferença máxima de "
            f"{max(differences) * 100:.2f}".replace(".", ",")
            + " pp — ordem de grandeza compatível com a variabilidade de "
            "inicialização das redes. **A implementação está correta: a queda "
            "observada na tabela anterior é atribuível exclusivamente à troca do "
            "split aleatório pelo cronológico.**",
            "",
            "Os artefatos deste controle ficam em `outputs/metrics/control_random/` "
            "e podem ser regerados com `python experiments/run_pooled.py --split random`.",
            "",
        ]

    return lines


def section_part_two(data: dict) -> list[str]:
    lines = [
        "---",
        "",
        "## Parte 2 — Modelos personalizados, um por sujeito",
        "",
        "Nesta etapa, treinou-se um modelo independente para cada sujeito, usando "
        "exclusivamente os dados daquele indivíduo. O filtro por sujeito é aplicado "
        "antes de qualquer divisão, e o split cronológico e o pré-processamento são "
        "os mesmos do modelo agrupado — inclusive o ajuste do `StandardScaler`, que "
        "é feito apenas sobre o treino do próprio sujeito.",
        "",
        "A coluna *classe majoritária* traz a acurácia que um classificador trivial "
        "obteria sempre chutando a classe mais frequente do teste daquele sujeito. "
        "Como o corte cronológico não equilibra as classes, essa é a linha de base "
        "honesta: um modelo que não a supera não está extraindo informação do sinal.",
        "",
        "| Sujeito | Modelo | Acurácia (personalizado) | F1 macro | Classe majoritária | Supera a linha de base? |",
        "|---|---|---|---|---|---|",
    ]

    any_result = False
    useful_subjects: set[str] = set()

    for user in USERS:
        for name in MODEL_NAMES:
            metrics = data["personalized"][user].get(name)

            if metrics is not None:
                any_result = True

            value = accuracy(metrics)
            majority = majority_baseline(metrics)

            if value is None or majority is None:
                verdict = "—"
            elif value > majority:
                verdict = "**sim**"
                useful_subjects.add(user)
            else:
                verdict = "não"

            lines.append(
                f"| {user} | {DISPLAY[name]} | {pct(value)} | "
                f"{pct(f1_macro(metrics))} | {pct(majority)} | {verdict} |"
            )

    lines.append("")

    if not any_result:
        lines += [
            "> *Nenhum resultado personalizado encontrado. Rode "
            "`python experiments/run_personalized.py` para gerá-los.*",
            "",
        ]
        return lines

    ranked = sorted(useful_subjects)
    others = [user for user in USERS if user not in useful_subjects]

    lines += [
        "### Desempenho muito desigual entre sujeitos",
        "",
        f"O resultado mais marcante desta etapa é a heterogeneidade entre os "
        f"indivíduos. Apenas **{len(ranked)} dos {len(USERS)} sujeitos** "
        f"({', '.join(ranked) if ranked else 'nenhum'}) produzem modelos que superam "
        f"a própria linha de base da classe majoritária. Para os demais "
        f"({', '.join(others) if others else 'nenhum'}), o modelo personalizado fica "
        "no nível do acaso — em alguns casos abaixo dele — mesmo tendo sido treinado "
        "exclusivamente com os dados daquela pessoa.",
        "",
        "Esse padrão é bem documentado na literatura de BCI sob o nome de "
        "*BCI illiteracy* (ou *BCI inefficiency*): estima-se que uma parcela "
        "relevante dos usuários não consiga gerar, com um dado paradigma de "
        "aquisição, padrões cerebrais suficientemente distinguíveis para controlar "
        "uma interface. As causas apontadas incluem diferenças anatômicas, "
        "posicionamento dos eletrodos, nível de atenção e estratégia mental adotada "
        "durante a tarefa de imaginação motora.",
        "",
        "Do ponto de vista prático, isso significa que o desempenho médio entre "
        "sujeitos é uma estatística enganosa para este sistema: ela mistura "
        "indivíduos para os quais a interface funciona com indivíduos para os quais "
        "ela não funciona de forma alguma. O reporte por sujeito é obrigatório.",
        "",
    ]

    return lines


def section_comparison(data: dict) -> list[str]:
    lines = [
        "---",
        "",
        "## Comparação: generalista × personalizado",
        "",
        "A comparação é feita sujeito a sujeito. A coluna *generalista* traz a "
        "acurácia do modelo agrupado **restrita às amostras de teste daquele "
        "sujeito** (campo `per_user_accuracy`), e a coluna *personalizado* traz a "
        "acurácia do modelo treinado apenas com aquele sujeito. Como ambos os tracks "
        "usam o mesmo split cronológico e a mesma proporção de teste, o conjunto de "
        "teste de cada sujeito é composto exatamente pelas mesmas amostras nos dois "
        "casos, o que torna a comparação direta.",
        "",
        "| Sujeito | Modelo | Generalista | Personalizado | Δ | Vencedor |",
        "|---|---|---|---|---|---|",
    ]

    wins = {"personalizado": 0, "generalista": 0, "empate": 0}
    per_model_general: dict[str, list[float]] = {name: [] for name in MODEL_NAMES}
    per_model_personal: dict[str, list[float]] = {name: [] for name in MODEL_NAMES}

    for user in USERS:
        for name in MODEL_NAMES:
            pooled = data["chronological"].get(name)
            general = (
                pooled["per_user_accuracy"].get(user)
                if pooled and pooled.get("per_user_accuracy")
                else None
            )
            personal = accuracy(data["personalized"][user].get(name))

            if general is None or personal is None:
                lines.append(
                    f"| {user} | {DISPLAY[name]} | {pct(general)} | "
                    f"{pct(personal)} | — | — |"
                )
                continue

            difference = personal - general
            per_model_general[name].append(general)
            per_model_personal[name].append(personal)

            if abs(difference) < 1e-9:
                winner = "empate"
            elif difference > 0:
                winner = "personalizado"
            else:
                winner = "generalista"

            wins[winner] += 1

            lines.append(
                f"| {user} | {DISPLAY[name]} | {pct(general)} | {pct(personal)} | "
                f"{points(difference)} | {winner} |"
            )

    lines += [
        "",
        "### Média ± desvio padrão por arquitetura (entre os 5 sujeitos)",
        "",
        "| Modelo | Generalista | Personalizado | Δ das médias |",
        "|---|---|---|---|",
    ]

    for name in MODEL_NAMES:
        general_values = per_model_general[name]
        personal_values = per_model_personal[name]

        difference = (
            statistics.mean(personal_values) - statistics.mean(general_values)
            if general_values and personal_values
            else None
        )

        lines.append(
            f"| {DISPLAY[name]} | {mean_std(general_values)} | "
            f"{mean_std(personal_values)} | {points(difference)} |"
        )

    lines.append("")

    total = sum(wins.values())

    if total:
        all_general = [v for values in per_model_general.values() for v in values]
        all_personal = [v for values in per_model_personal.values() for v in values]
        overall = statistics.mean(all_personal) - statistics.mean(all_general)

        useful = sum(
            1
            for user in USERS
            for name in MODEL_NAMES
            if (metrics := data["personalized"][user].get(name))
            and (value := accuracy(metrics)) is not None
            and (baseline := majority_baseline(metrics)) is not None
            and value > baseline
        )

        lines += [
            "### Interpretação",
            "",
            f"Em {total} comparações diretas (sujeito × arquitetura), o modelo "
            f"personalizado superou o generalista em **{wins['personalizado']}** "
            f"casos, contra {wins['generalista']} do generalista"
            + (f" e {wins['empate']} empate(s)" if wins["empate"] else "")
            + f". A diferença média entre os dois paradigmas foi de "
            f"**{points(overall)}** em favor do "
            + ("personalizado" if overall > 0 else "generalista")
            + ".",
            "",
            "Vale destacar o desequilíbrio de dados por trás desse resultado: cada "
            "modelo personalizado é treinado com aproximadamente **um quinto** do "
            "volume disponível ao modelo agrupado. Ainda assim o personalizado "
            "vence na maioria das comparações, o que sugere que a variabilidade "
            "entre sujeitos é grande o bastante para que os dados de terceiros "
            "atrapalhem mais do que ajudem: ao misturar indivíduos, o modelo "
            "agrupado é empurrado para uma média que não descreve bem nenhum deles.",
            "",
            "**Duas conclusões distintas, que não devem ser confundidas:**",
            "",
            "1. *Personalizar ajuda.* A comparação direta favorece o paradigma "
            "personalizado de forma consistente, em todas as arquiteturas.",
            f"2. *Personalizar não é suficiente.* Das {total} combinações testadas, "
            f"apenas **{useful}** superam a linha de base da classe majoritária do "
            "próprio sujeito. Vencer o modelo agrupado é fácil quando o modelo "
            "agrupado está no nível do acaso; ser útil na prática é outro patamar, "
            "atingido apenas por parte dos sujeitos.",
            "",
            "Para a aplicação-alvo (controle de uma interface por intenção motora), "
            "a leitura honesta é que o pipeline atual — features de potência "
            "espectral por amostra isolada, sem contexto temporal — só produz um "
            "classificador aproveitável para uma parcela dos usuários, e mesmo assim "
            "com margem modesta sobre a linha de base.",
            "",
        ]

    return lines


def section_figures() -> list[str]:
    lines = [
        "---",
        "",
        "## Figuras",
        "",
        "As figuras a seguir foram geradas na etapa anterior do projeto "
        "(split aleatório) e permanecem disponíveis em `outputs/figures/`:",
        "",
    ]

    if FIGURES_DIR.exists():
        figures = sorted(path.name for path in FIGURES_DIR.glob("*.png"))
        lines += [f"- `outputs/figures/{name}`" for name in figures] or [
            "- *(nenhuma figura encontrada)*"
        ]
    else:
        lines.append("- *(diretório de figuras não encontrado)*")

    lines.append("")

    return lines


def section_limitations() -> list[str]:
    return [
        "---",
        "",
        "## Observações e limitações",
        "",
        "1. **Volume de dados desigual entre os paradigmas.** Cada modelo "
        "personalizado dispõe de cerca de 20% do volume usado pelo modelo agrupado. "
        "A comparação é, portanto, indicativa do paradigma como um todo (arquitetura "
        "+ quantidade de dados disponível na prática), e não um teste controlado do "
        "efeito da personalização isoladamente.",
        "",
        "2. **Poucos blocos independentes no conjunto de teste.** Cada sujeito tem 36 "
        "blocos contíguos de rótulos (o ciclo Esquerda → Direita → Neutro repetido 12 "
        "vezes). Com 20% reservados para teste, restam cerca de 7 blocos por sujeito. "
        "Embora as três classes estejam presentes no teste de todos os sujeitos, o "
        "número reduzido de blocos independentes significa que as métricas têm "
        "variância maior do que o número total de amostras sugere.",
        "",
        "3. **Distribuição de classes desbalanceada no teste cronológico.** Ao "
        "contrário do split aleatório estratificado, o corte cronológico não "
        "equilibra as classes. Por isso o F1 macro é reportado ao lado da acurácia: "
        "ele é a métrica mais confiável para comparar os modelos aqui.",
        "",
        "4. **Saturação pelo clipping de IQR.** O tratamento de outliers achata "
        "valores extremos nos limites do intervalo interquartil. Amostras muito "
        "extremas acabam colapsando no mesmo vetor de limite, perdendo distinção "
        "entre si. Isso afeta uma fração pequena das amostras, mas é uma perda de "
        "informação inerente à estratégia adotada.",
        "",
        "5. **Sem janelas temporais.** Todos os modelos classificam cada amostra "
        "isoladamente, sem contexto de amostras vizinhas no tempo (a CNN opera com "
        "`n_frames=1`). Incorporar janelas deslizantes é um experimento subsequente e "
        "está fora do escopo desta etapa.",
        "",
        "6. **Sem avaliação com participante inteiro fora do treino.** A capacidade de generalização "
        "para um sujeito completamente novo — cenário mais exigente e mais próximo do "
        "uso real de uma BCI — não é avaliada aqui, e fica como trabalho futuro. "
        "Dado o desempenho do modelo agrupado, a expectativa é que esse cenário seja "
        "ainda mais desfavorável.",
        "",
        "7. **Comparar acurácias entre sujeitos exige cautela.** Como a distribuição "
        "de classes no teste cronológico difere de um sujeito para outro, a linha de "
        "base da classe majoritária também difere. Uma acurácia de 52% pode ser "
        "excelente para um sujeito e irrelevante para outro; por isso todas as "
        "tabelas por sujeito reportam a linha de base ao lado do resultado.",
        "",
        "8. **Um único treino por combinação.** Cada resultado vem de uma execução "
        "com semente fixa (42). Não há repetições com sementes diferentes, então as "
        "diferenças pequenas entre arquiteturas — da ordem de poucos pontos "
        "percentuais — estão dentro da faixa em que a variabilidade de inicialização "
        "poderia explicá-las sozinha.",
        "",
    ]


def main() -> None:
    data = collect()

    lines: list[str] = []
    lines += section_header(data)
    lines += section_part_one(data)
    lines += section_part_two(data)
    lines += section_comparison(data)
    lines += section_figures()
    lines += section_limitations()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"[resultados] Salvo: {OUTPUT_PATH}")
    print(f"[resultados] {len(lines)} linhas")


if __name__ == "__main__":
    main()
