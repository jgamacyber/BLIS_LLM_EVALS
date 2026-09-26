"""
Análise dos resultados — a parte que transforma números em conclusão.

Quatro perguntas que esta análise responde:

1. **Quanto cada métrica concorda com a referência?** (`alinhamento`)
   Usando Scott's π e Spearman, não só o percentual de acerto.

2. **O juiz caro supera o baseline grátis?** (`custo_beneficio`)
   Se o G-Eval não bate o `contains`, ele não está pagando o próprio custo.

3. **Qual é a qualidade real do sistema, com margem de erro?** (`estimar_ppi`)
   Usando PPI para combinar poucas anotações com muitas predições.

4. **Onde as métricas discordam?** (`divergencias`)
   Os casos que valem leitura manual — é onde se aprende o que a métrica não vê.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from concordancia import Alinhamento, comparar_com_humano, tabela_alinhamento
from pipeline import Avaliacao, ResultadoCaso
from ppi import (
    IntervaloPPI,
    intervalo_classico,
    intervalo_ppi,
    intervalo_so_juiz,
    tabela_intervalos,
)

DIR_RESULTADOS = Path("resultados")

# Qual métrica automática corresponde a qual dimensão anotada.
# Este mapa é o coração da análise: sem ele, comparar "faithfulness" com
# "relevancia" produziria números sem sentido.
MAPA_DIMENSOES = {
    "fundamentacao": ["faithfulness", "geval_fundamentação", "contains", "f1_tokens"],
    "relevancia": ["answer_relevance", "geval_relevância", "contains", "f1_tokens"],
    "contexto_relevante": ["context_relevance"],
}

# Métricas gratuitas, usadas como baseline de custo-benefício.
METRICAS_GRATIS = {"exact_match", "contains", "f1_tokens", "cobertura"}


# --------------------------------------------------------------------------- #
# 1. Alinhamento com a referência
# --------------------------------------------------------------------------- #


def alinhamento(
    avaliacao: Avaliacao, dimensao: str, limiar: float = 0.5
) -> list[Alinhamento]:
    """
    Mede o alinhamento de cada métrica candidata com a dimensão anotada.

    Só compara métricas que fazem sentido para aquela dimensão (ver
    MAPA_DIMENSOES) — comparar tudo com tudo produziria correlações espúrias.
    """
    casos = avaliacao.com_anotacao(dimensao)
    if not casos:
        return []

    humanas = [float(c.anotacao_humana[dimensao]) for c in casos]
    candidatas = MAPA_DIMENSOES.get(dimensao, avaliacao.nomes_metricas())

    saida: list[Alinhamento] = []
    for nome in candidatas:
        valores = [c.metrica(nome) for c in casos]
        # Métrica ausente da execução (ex.: RAGAS não foi rodado)
        if all(v == 0.0 for v in valores):
            continue
        saida.append(
            comparar_com_humano(valores, humanas, nome, limiar_binario=limiar)
        )

    return saida


def imprimir_alinhamento(avaliacao: Avaliacao, limiar: float = 0.5) -> None:
    """Relatório de alinhamento para todas as dimensões anotadas."""
    dimensoes = [d for d in MAPA_DIMENSOES if avaliacao.com_anotacao(d)]
    if not dimensoes:
        print("\nNenhum caso tem anotação de referência — nada a comparar.")
        return

    for dimensao in dimensoes:
        print(f"\n{'=' * 88}")
        print(f"  Dimensão: {dimensao}")
        print("=" * 88)
        alinhamentos = alinhamento(avaliacao, dimensao, limiar)
        if alinhamentos:
            tabela_alinhamento(alinhamentos)
        else:
            print("  (nenhuma métrica aplicável foi calculada)")


# --------------------------------------------------------------------------- #
# 2. Custo-benefício do juiz
# --------------------------------------------------------------------------- #


@dataclass
class ComparacaoCusto:
    metrica: str
    scott_pi: float
    gratis: bool
    ganho_sobre_baseline: float = 0.0


def custo_beneficio(
    avaliacao: Avaliacao, dimensao: str, limiar: float = 0.5
) -> list[ComparacaoCusto]:
    """
    Compara cada métrica com o melhor baseline gratuito.

    Thakur et al. (2025) mostram que a métrica lexical `contains` ranqueia
    modelos quase tão bem quanto juízes LLM medianos. Este relatório torna essa
    comparação explícita: se o juiz pago não supera o `contains`, o custo não
    se justifica.
    """
    alinhamentos = alinhamento(avaliacao, dimensao, limiar)
    if not alinhamentos:
        return []

    baseline = max(
        (a for a in alinhamentos if a.nome in METRICAS_GRATIS),
        key=lambda a: a.scott_pi,
        default=None,
    )
    referencia = baseline.scott_pi if baseline else 0.0

    return [
        ComparacaoCusto(
            metrica=a.nome,
            scott_pi=a.scott_pi,
            gratis=a.nome in METRICAS_GRATIS,
            ganho_sobre_baseline=a.scott_pi - referencia,
        )
        for a in sorted(alinhamentos, key=lambda x: x.scott_pi, reverse=True)
    ]


def imprimir_custo_beneficio(
    avaliacao: Avaliacao, dimensao: str = "fundamentacao", limiar: float = 0.5
) -> None:
    comparacoes = custo_beneficio(avaliacao, dimensao, limiar)
    if not comparacoes:
        return

    baseline = next((c for c in comparacoes if c.gratis), None)
    nome_baseline = baseline.metrica if baseline else "—"

    print(f"\nO juiz pago vale o custo?  (dimensão: {dimensao})")
    print("-" * 74)
    print(f"{'Métrica':<26} {'Scott π':>9} {'Custo':>10} {'Ganho s/ baseline':>20}")
    print("-" * 74)

    for c in comparacoes:
        custo = "grátis" if c.gratis else "chamadas LLM"
        if c.gratis and c.metrica == nome_baseline:
            ganho = "  (baseline)"
        else:
            ganho = f"{c.ganho_sobre_baseline:>+19.3f}"
        print(f"{c.metrica:<26} {c.scott_pi:>9.3f} {custo:>10} {ganho:>20}")

    print("-" * 74)

    pagas = [c for c in comparacoes if not c.gratis]
    if pagas and baseline:
        melhor_paga = max(pagas, key=lambda c: c.scott_pi)
        if melhor_paga.scott_pi <= baseline.scott_pi:
            print(f"  → Nenhuma métrica paga supera '{nome_baseline}', que é grátis.")
            print("    Neste conjunto, o juiz LLM não está pagando o próprio custo.")
        else:
            print(f"  → '{melhor_paga.metrica}' supera o baseline grátis em "
                  f"{melhor_paga.ganho_sobre_baseline:+.3f} de Scott π.")


# --------------------------------------------------------------------------- #
# 3. PPI — qualidade com margem de erro
# --------------------------------------------------------------------------- #


def estimar_ppi(
    avaliacao: Avaliacao,
    metrica: str,
    dimensao: str,
    n_rotulado: int = 10,
    alpha: float = 0.05,
    semente: int = 42,
) -> dict[str, IntervaloPPI]:
    """
    Estima a qualidade real do sistema por três caminhos, para comparação.

    Simula o cenário do ARES: parte dos casos tem anotação (conjunto rotulado)
    e o resto só tem a predição do juiz. Como aqui conhecemos a anotação de
    todos, dá para verificar se o intervalo realmente cobre o valor verdadeiro.
    """
    import random

    casos = avaliacao.com_anotacao(dimensao)
    if not casos:
        return {}

    rng = random.Random(semente)
    embaralhados = list(casos)
    rng.shuffle(embaralhados)

    n_rotulado = max(2, min(n_rotulado, len(embaralhados) - 1))
    rotulados = embaralhados[:n_rotulado]
    nao_rotulados = embaralhados[n_rotulado:]

    pred_rot = [c.metrica(metrica) for c in rotulados]
    humanos_rot = [float(c.anotacao_humana[dimensao]) for c in rotulados]
    pred_nao_rot = [c.metrica(metrica) for c in nao_rotulados]

    return {
        "só anotação humana": intervalo_classico(humanos_rot, alpha),
        "só juiz (inválido)": intervalo_so_juiz(pred_nao_rot, alpha),
        "PPI (ARES)": intervalo_ppi(pred_rot, humanos_rot, pred_nao_rot, alpha),
    }


def imprimir_ppi(
    avaliacao: Avaliacao,
    metrica: str,
    dimensao: str,
    n_rotulado: int = 10,
    alpha: float = 0.05,
) -> None:
    intervalos = estimar_ppi(avaliacao, metrica, dimensao, n_rotulado, alpha)
    if not intervalos:
        print("\nSem casos anotados — PPI não aplicável.")
        return

    casos = avaliacao.com_anotacao(dimensao)
    verdadeiro = sum(
        float(c.anotacao_humana[dimensao]) for c in casos
    ) / len(casos)

    print(f"\nEstimativa da dimensão '{dimensao}' via métrica '{metrica}'")
    print(f"  ({n_rotulado} casos rotulados, {len(casos) - n_rotulado} não rotulados)")
    tabela_intervalos(intervalos)

    print(f"\n  Valor verdadeiro (média de TODAS as anotações): {verdadeiro:.3f}")
    for nome, i in intervalos.items():
        marca = "cobre" if i.contem(verdadeiro) else "NÃO cobre"
        erro = abs(i.estimativa - verdadeiro)
        print(f"    {nome:<24} {marca:<10} (erro: {erro:.3f})")

    # A leitura é derivada dos números desta execução, não de uma afirmação
    # genérica — o comportamento do PPI depende da qualidade do juiz.
    classico = intervalos["só anotação humana"]
    so_juiz = intervalos["só juiz (inválido)"]
    ppi = intervalos["PPI (ARES)"]

    print("\n  Leitura desta execução:")

    if not so_juiz.contem(verdadeiro):
        print(f"    - 'só juiz' é o intervalo mais estreito ({so_juiz.largura:.3f}) "
              f"e NÃO cobre o")
        print("      valor verdadeiro. Estreiteza não é acurácia: mais dados não")
        print("      corrigem um viés sistemático, só dão mais confiança no erro.")
    else:
        print("    - 'só juiz' cobriu o valor verdadeiro nesta amostra, mas o")
        print("      intervalo dele não tem garantia estatística — ele mede a")
        print("      incerteza sobre a média do juiz, não sobre a verdade.")

    erro_ppi = abs(ppi.estimativa - verdadeiro)
    erro_classico = abs(classico.estimativa - verdadeiro)
    if erro_ppi < erro_classico:
        print(f"    - O PPI chegou mais perto do valor verdadeiro "
              f"({erro_ppi:.3f}) do que usar")
        print(f"      só as {ppi.n_rotulado} anotações ({erro_classico:.3f}).")

    if ppi.largura < classico.largura:
        reducao = (1 - ppi.largura / classico.largura) * 100 if classico.largura else 0
        print(f"    - O PPI estreitou o intervalo em {reducao:.0f}% em relação a "
              f"usar só as anotações,")
        print("      aproveitando as predições do juiz nos casos não rotulados.")
    else:
        print(f"    - Aqui o PPI NÃO estreitou o intervalo "
              f"({ppi.largura:.3f} contra {classico.largura:.3f}).")
        print(f"      Isso acontece quando o juiz é ruim: o retificador "
              f"({ppi.retificador:+.3f}) varia")
        print("      muito entre os casos, e essa variância entra na conta. O PPI")
        print("      continua corrigindo o viés — só não ganha eficiência. Um juiz")
        print("      melhor estreita o intervalo; experimente com faithfulness.")


# --------------------------------------------------------------------------- #
# 4. Divergências
# --------------------------------------------------------------------------- #


def divergencias(
    avaliacao: Avaliacao, dimensao: str, limiar: float = 0.5
) -> list[tuple[ResultadoCaso, dict]]:
    """
    Casos em que alguma métrica discorda da anotação de referência.

    Ordenados pelo número de métricas que erraram — os casos que confundem
    todo mundo vêm primeiro, e são os mais informativos para ler à mão.
    """
    casos = avaliacao.com_anotacao(dimensao)
    candidatas = [
        m for m in MAPA_DIMENSOES.get(dimensao, [])
        if any(c.metrica(m) != 0.0 for c in casos)
    ]

    saida = []
    for caso in casos:
        humano = float(caso.anotacao_humana[dimensao])
        discordancias = {}
        for metrica in candidatas:
            valor = caso.metrica(metrica)
            binario = 1.0 if valor >= limiar else 0.0
            if binario != humano:
                discordancias[metrica] = valor
        if discordancias:
            saida.append((caso, discordancias))

    return sorted(saida, key=lambda t: len(t[1]), reverse=True)


def imprimir_divergencias(
    avaliacao: Avaliacao,
    dimensao: str = "fundamentacao",
    limite: int = 6,
    limiar: float = 0.5,
) -> None:
    casos = divergencias(avaliacao, dimensao, limiar)
    if not casos:
        print(f"\nNenhuma divergência em '{dimensao}': todas as métricas "
              f"concordaram com a referência.")
        return

    total = len(avaliacao.com_anotacao(dimensao))
    print(f"\nDivergências em '{dimensao}'  ({len(casos)} de {total} casos)")
    print("-" * 88)

    for caso, discordantes in casos[:limite]:
        humano = float(caso.anotacao_humana[dimensao])
        print(f"\n  {caso.caso_id}  — referência: {humano:.0f}")
        print(f"    pergunta: {caso.pergunta[:70]}")
        print(f"    resposta: {caso.resposta[:90]}")
        for metrica, valor in discordantes.items():
            print(f"      {metrica:<24} {valor:.3f}  (discorda)")

    print("-" * 88)
    print("  Estes são os casos que vale ler à mão: é onde a métrica não vê o")
    print("  que a pessoa vê.")


# --------------------------------------------------------------------------- #
# Exportação
# --------------------------------------------------------------------------- #


def exportar_csv(avaliacao: Avaliacao, caminho: str | Path) -> Path:
    """Uma linha por caso, com todas as métricas — para planilha."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)

    metricas = avaliacao.nomes_metricas()
    dimensoes = sorted(
        {d for r in avaliacao.resultados for d in r.anotacao_humana}
    )

    with caminho.open("w", newline="", encoding="utf-8-sig") as f:
        escritor = csv.writer(f, delimiter=";")
        escritor.writerow(
            ["caso_id", "pergunta", "resposta", "abstencao"]
            + metricas
            + [f"humano_{d}" for d in dimensoes]
            + ["erros"]
        )
        for r in avaliacao.resultados:
            escritor.writerow(
                [
                    r.caso_id,
                    r.pergunta[:150],
                    r.resposta.replace("\n", " ")[:250],
                    "sim" if r.abstencao else "nao",
                ]
                + [f"{r.metrica(m):.4f}".replace(".", ",") for m in metricas]
                + [r.anotacao_humana.get(d, "") for d in dimensoes]
                + ["; ".join(r.erros)[:200]]
            )
    return caminho


def exportar_markdown(avaliacao: Avaliacao, caminho: str | Path) -> Path:
    """Relatório em Markdown com as tabelas de alinhamento."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)

    linhas = [
        "# Relatório de avaliação",
        "",
        f"- **Juiz:** `{avaliacao.juiz_model}`",
        f"- **Métricas:** {', '.join(avaliacao.metricas_usadas)}",
        f"- **Casos:** {len(avaliacao.resultados)}",
        f"- **Execução:** {avaliacao.timestamp}",
        "",
        "> Material educacional. As anotações de referência não vêm de um estudo",
        "> com múltiplos anotadores humanos — veja `data/PROCEDENCIA.md`. Os",
        "> números abaixo medem concordância com essa referência, não com",
        "> \"humanos\" em geral.",
        "",
    ]

    for dimensao in MAPA_DIMENSOES:
        alinhamentos = alinhamento(avaliacao, dimensao)
        if not alinhamentos:
            continue

        linhas += [
            f"## Dimensão: {dimensao}",
            "",
            "| Métrica | Concordância | Scott π | Spearman | Kendall | Desvio |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for a in sorted(alinhamentos, key=lambda x: x.scott_pi, reverse=True):
            linhas.append(
                f"| `{a.nome}` | {a.concordancia:.1%} | {a.scott_pi:.3f} "
                f"| {a.spearman:.3f} | {a.kendall:.3f} | {a.desvio_medio:+.2f} |"
            )
        linhas.append("")

    linhas += [
        "---",
        "",
        "Scott π corrige a concordância pelo acaso; o percentual bruto engana "
        "quando os rótulos são desbalanceados. Desvio positivo indica juiz "
        "leniente.",
        "",
        "Gerado por `python main.py analisar --markdown`.",
    ]

    caminho.write_text("\n".join(linhas), encoding="utf-8")
    return caminho
