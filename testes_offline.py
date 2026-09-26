"""
Testes offline — validam a lógica sem chamar a API nem gastar créditos.

Cobrem as métricas lexicais, toda a estatística (Scott's π, Cohen's κ,
Spearman, Kendall, PPI), o carregamento e validação do conjunto, e as sondagens
de viés que não precisam de LLM.

Boa parte destes testes verifica propriedades matemáticas contra valores
calculados à mão — é o tipo de coisa que, se estiver errada, contamina todos os
números do relatório em silêncio.

    python testes_offline.py
"""

from __future__ import annotations

import sys

from concordancia import (
    cohen_kappa,
    comparar_com_humano,
    concordancia_bruta,
    interpretar_kappa,
    kendall_tau,
    pearson,
    proporcao_empates,
    scott_pi,
    spearman,
)
from dataset import Caso, carregar, dividir_para_ppi, distribuicao_anotacoes, validar
from metricas_lexicais import (
    avaliar_lexical,
    contains,
    cobertura_gabarito,
    eh_abstencao,
    exact_match,
    f1_tokens,
    normalizar,
)
from ppi import (
    intervalo_classico,
    intervalo_ppi,
    intervalo_so_juiz,
    z_critico,
)
from ragas import dividir_frases
from vieses import testar_leniencia, testar_vies_verbosidade

FALHAS: list[str] = []


def checar(condicao: bool, descricao: str) -> None:
    if condicao:
        print(f"  [ok]   {descricao}")
    else:
        print(f"  [FALHA] {descricao}")
        FALHAS.append(descricao)


def secao(titulo: str) -> None:
    print(f"\n{titulo}")
    print("-" * 64)


def perto(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) < tol


# --------------------------------------------------------------------------- #


def testar_lexicais() -> None:
    secao("1. Métricas lexicais")

    checar(exact_match("Canberra", "canberra") == 1.0, "EM ignora caixa")
    checar(exact_match("A Canberra.", "canberra") == 1.0,
           "EM ignora artigos e pontuação")
    checar(exact_match("Sydney", "canberra") == 0.0, "EM rejeita divergente")
    checar(exact_match("nao", ["nao", "não"]) == 1.0, "EM aceita alternativas")

    checar(contains("A capital é Canberra.", "canberra") == 1.0,
           "contains encontra a substring")
    checar(contains("Canberra", "a capital é canberra") == 0.0,
           "contains é direcional (gabarito dentro da resposta)")

    checar(perto(f1_tokens("capital canberra", "capital canberra"), 1.0),
           "F1 de tokens idênticos é 1.0")
    checar(f1_tokens("sydney melbourne", "canberra") == 0.0,
           "F1 sem sobreposição é 0")
    parcial = f1_tokens("a capital australiana e canberra", "canberra")
    checar(0 < parcial < 1, f"F1 dá crédito parcial ({parcial:.2f})")

    checar(perto(cobertura_gabarito("canberra fica na australia",
                                    "canberra australia"), 1.0),
           "cobertura acha todos os tokens do gabarito")

    checar(normalizar("  A Ação, RÁPIDA!  ") == "acao rapida",
           "normalização remove acentos, artigos e pontuação")

    # Abstenção
    checar(eh_abstencao("O contexto fornecido não contém informação suficiente "
                        "para responder."), "detecta abstenção padrão")
    checar(eh_abstencao("Não sei responder."), "detecta 'não sei'")
    checar(eh_abstencao("Insufficient Information"), "detecta abstenção em inglês")
    checar(not eh_abstencao("A capital é Canberra."), "não confunde resposta normal")

    # O caso especial: pergunta sem resposta possível
    correto = avaliar_lexical("O contexto fornecido não contém informação "
                              "suficiente para responder.", "")
    checar(correto.contains == 1.0,
           "sem gabarito, a abstenção conta como ACERTO")
    checar(correto.abstencao, "sem gabarito, a abstenção é sinalizada")

    errado = avaliar_lexical("A resposta é 42.", "")
    checar(errado.contains == 0.0,
           "sem gabarito, inventar uma resposta conta como erro")

    normal = avaliar_lexical("A capital é Canberra.", "canberra")
    checar(normal.contains == 1.0 and not normal.abstencao,
           "com gabarito, avalia normalmente")


def testar_concordancia_categorica() -> None:
    secao("2. Concordância corrigida pelo acaso")

    a = [1, 1, 0, 0, 1, 0]
    checar(concordancia_bruta(a, a) == 1.0, "concordância perfeita = 1.0")
    checar(concordancia_bruta([1, 1], [0, 0]) == 0.0, "discordância total = 0.0")
    checar(perto(concordancia_bruta([1, 1, 0, 0], [1, 0, 0, 0]), 0.75),
           "concordância parcial correta")

    checar(perto(scott_pi(a, a), 1.0), "Scott π de acordo perfeito é 1.0")
    checar(perto(cohen_kappa(a, a), 1.0), "Cohen κ de acordo perfeito é 1.0")

    # Caso calculado à mão:
    # a = [1,1,1,1,0,0,0,0], b = [1,1,1,0,1,0,0,0] -> Po = 6/8 = 0.75
    # p_a(1)=0.5, p_b(1)=0.5 -> Pe_scott = 0.5^2 + 0.5^2 = 0.5
    # pi = (0.75 - 0.5) / 0.5 = 0.5
    x = [1, 1, 1, 1, 0, 0, 0, 0]
    y = [1, 1, 1, 0, 1, 0, 0, 0]
    checar(perto(concordancia_bruta(x, y), 0.75), "Po do caso manual = 0.75")
    checar(perto(scott_pi(x, y), 0.5), "Scott π do caso manual = 0.5 (à mão)")
    checar(perto(cohen_kappa(x, y), 0.5),
           "com marginais iguais, κ e π coincidem")

    # O ponto central: concordância alta pode esconder π baixo.
    desbalanceado_a = [1] * 18 + [0, 0]
    desbalanceado_b = [1] * 20
    conc = concordancia_bruta(desbalanceado_a, desbalanceado_b)
    pi = scott_pi(desbalanceado_a, desbalanceado_b)
    checar(conc >= 0.9, f"concordância parece ótima ({conc:.0%})")
    checar(pi < 0.1, f"mas Scott π revela que é quase acaso ({pi:.3f})")

    # Marginais diferentes: κ e π divergem (juiz mais leniente)
    leniente = [1] * 15 + [0] * 5
    severo = [1] * 10 + [0] * 10
    checar(scott_pi(leniente, severo) != cohen_kappa(leniente, severo),
           "com marginais diferentes, π e κ divergem")

    checar(interpretar_kappa(0.85) == "quase perfeita", "interpretação de κ alto")
    checar(interpretar_kappa(-0.1) == "pior que o acaso", "interpretação de κ negativo")

    try:
        scott_pi([1, 0], [1])
        checar(False, "listas de tamanhos diferentes levantam erro")
    except ValueError:
        checar(True, "listas de tamanhos diferentes levantam erro")


def testar_correlacoes() -> None:
    secao("3. Correlações")

    crescente = [1.0, 2.0, 3.0, 4.0, 5.0]
    decrescente = [5.0, 4.0, 3.0, 2.0, 1.0]

    checar(perto(pearson(crescente, crescente), 1.0), "Pearson de si mesmo = 1.0")
    checar(perto(pearson(crescente, decrescente), -1.0), "Pearson invertido = -1.0")
    checar(pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) == 0.0,
           "Pearson com variância zero devolve 0 (sem divisão por zero)")

    checar(perto(spearman(crescente, crescente), 1.0), "Spearman de si mesmo = 1.0")
    checar(perto(spearman(crescente, decrescente), -1.0), "Spearman invertido = -1.0")

    # Spearman é invariante a transformação monotônica; Pearson não.
    exponencial = [2.0, 4.0, 8.0, 16.0, 32.0]
    checar(perto(spearman(crescente, exponencial), 1.0),
           "Spearman é 1.0 sob transformação monotônica")
    checar(pearson(crescente, exponencial) < 0.99,
           "Pearson não é 1.0 sob transformação não linear")

    # Spearman ignora deslocamento de escala — o ponto do G-Eval.
    deslocado = [3.0, 4.0, 5.0, 6.0, 7.0]
    checar(perto(spearman(crescente, deslocado), 1.0),
           "Spearman ignora deslocamento constante de escala")

    checar(perto(kendall_tau(crescente, crescente), 1.0), "Kendall de si mesmo = 1.0")
    checar(perto(kendall_tau(crescente, decrescente), -1.0), "Kendall invertido = -1.0")

    checar(perto(proporcao_empates([1.0, 1.0, 2.0]), 1 / 3),
           "proporção de empates correta")
    checar(proporcao_empates([1.0, 2.0, 3.0]) == 0.0, "sem empates = 0")
    checar(proporcao_empates([5.0] * 4) == 1.0, "tudo empatado = 1.0")


def testar_alinhamento() -> None:
    secao("4. Alinhamento consolidado")

    juiz = [0.9, 0.8, 0.2, 0.1, 0.7, 0.3]
    humano = [1.0, 1.0, 0.0, 0.0, 1.0, 0.0]

    a = comparar_com_humano(juiz, humano, "teste")
    checar(a.n == 6, "registra o n")
    checar(perto(a.concordancia, 1.0), "binarização acerta todos neste caso")
    checar(a.scott_pi > 0.9, "Scott π alto quando o juiz acerta tudo")
    # Com empates na referência (três 1.0 e três 0.0) o Spearman tem teto
    # abaixo de 1.0 mesmo com o juiz ordenando perfeitamente: ~0.88 aqui.
    checar(a.spearman > 0.85, f"Spearman alto ({a.spearman:.3f})")

    # Juiz leniente: soma 0,3 a todos.
    leniente = [min(1.0, v + 0.3) for v in juiz]
    al = comparar_com_humano(leniente, humano, "leniente")
    checar(al.desvio_medio > 0, "juiz leniente tem desvio positivo")
    checar(perto(al.spearman, a.spearman, 0.3),
           "leniência não destrói a correlação de ranking")

    vazio = comparar_com_humano([], [], "vazio")
    checar(vazio.n == 0, "listas vazias não quebram")

    try:
        comparar_com_humano([1.0], [1.0, 2.0], "erro")
        checar(False, "tamanhos diferentes levantam erro")
    except ValueError:
        checar(True, "tamanhos diferentes levantam erro")


def testar_ppi() -> None:
    secao("5. PPI — intervalos de confiança")

    checar(perto(z_critico(0.05), 1.96, 1e-3), "z de 95% é 1.96")
    checar(perto(z_critico(0.01), 2.5758, 1e-3), "z de 99% é 2.576")

    humanos = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    classico = intervalo_classico(humanos)
    checar(perto(classico.estimativa, 0.625), "estimativa clássica é a média")
    checar(classico.limite_inferior < classico.estimativa < classico.limite_superior,
           "intervalo contém a estimativa")
    checar(classico.metodo == "classico", "método identificado")

    # Juiz leniente: dá nota alta também para os casos que o humano reprovou.
    # É assim que a leniência aparece na prática — não como um deslocamento
    # uniforme, mas como falha em reprovar.
    def juiz_leniente(v: float) -> float:
        return 1.0 if v == 1.0 else 0.6

    pred_rot = [juiz_leniente(v) for v in humanos]
    pred_nao_rot = [juiz_leniente(v) for v in humanos] * 4

    ppi = intervalo_ppi(pred_rot, humanos, pred_nao_rot)
    checar(ppi.metodo == "ppi", "método PPI identificado")
    checar(ppi.retificador > 0, "retificador positivo detecta juiz leniente")
    checar(ppi.estimativa < ppi.media_juiz_nao_rotulado,
           "a estimativa PPI corrige a média inflada do juiz para baixo")
    checar(ppi.n_rotulado == 8 and ppi.n_nao_rotulado == 32,
           "registra os tamanhos dos conjuntos")

    # Juiz perfeito: retificador zero e intervalo mais estreito que o clássico.
    perfeito_rot = list(humanos)
    perfeito_nao_rot = [1.0, 0.0] * 16
    ppi_perfeito = intervalo_ppi(perfeito_rot, humanos, perfeito_nao_rot)
    checar(perto(ppi_perfeito.retificador, 0.0),
           "juiz perfeito tem retificador zero")
    checar(ppi_perfeito.largura < classico.largura,
           "juiz perfeito estreita o intervalo em relação ao clássico")

    # Degradação graciosa
    sem_nao_rotulado = intervalo_ppi(pred_rot, humanos, [])
    checar(sem_nao_rotulado.metodo == "classico",
           "sem casos não rotulados, degrada para o clássico")

    sem_rotulado = intervalo_ppi([], [], pred_nao_rot)
    checar(sem_rotulado.metodo == "so_juiz",
           "sem casos rotulados, degrada para só-juiz")

    so_juiz = intervalo_so_juiz(pred_nao_rot)
    checar(so_juiz.metodo == "so_juiz", "só-juiz identificado")

    # A demonstração central: o intervalo só-juiz é estreito e errado.
    verdadeiro = sum(humanos) / len(humanos)
    checar(not so_juiz.contem(verdadeiro),
           "o intervalo 'só juiz' NÃO cobre o valor verdadeiro (juiz enviesado)")
    checar(ppi.contem(verdadeiro),
           "o intervalo PPI cobre o valor verdadeiro")

    try:
        intervalo_ppi([1.0], [1.0, 0.0], [0.5])
        checar(False, "tamanhos incompatíveis levantam erro")
    except ValueError:
        checar(True, "tamanhos incompatíveis levantam erro")


def testar_dataset() -> None:
    secao("6. Conjunto de casos")

    casos = carregar()
    checar(len(casos) >= 20, f"conjunto tem {len(casos)} casos")
    checar(len({c.id for c in casos}) == len(casos), "ids únicos")
    checar(all(c.pergunta and c.contexto and c.resposta for c in casos),
           "nenhum caso com campo obrigatório vazio")

    anotados = [c for c in casos if c.tem_anotacao]
    checar(len(anotados) == len(casos), "todos os casos têm anotação")

    sem_resposta = [c for c in casos if c.sem_resposta_possivel]
    checar(len(sem_resposta) >= 3,
           f"{len(sem_resposta)} casos sem resposta possível (testam abstenção)")

    # As duas classes precisam existir, senão a concordância não mede nada.
    fundamentacao = [c.anotacao("fundamentacao") for c in anotados]
    checar(0 < sum(fundamentacao) < len(fundamentacao),
           "a dimensão 'fundamentacao' tem casos positivos E negativos")

    relevancia = [c.anotacao("relevancia") for c in anotados]
    checar(0 < sum(relevancia) < len(relevancia),
           "a dimensão 'relevancia' tem casos positivos E negativos")

    avisos = validar(casos)
    graves = [a for a in avisos if "duplicado" in a or "vazio" in a]
    checar(not graves, f"nenhum problema estrutural grave ({len(avisos)} avisos)")

    dist = distribuicao_anotacoes(casos)
    checar("fundamentacao" in dist and "nota_global" in dist,
           "distribuição calculada para as dimensões")

    rot, nao_rot = dividir_para_ppi(casos, n_rotulado=10)
    checar(len(rot) == 10, "divisão para PPI respeita n_rotulado")
    checar(len(rot) + len(nao_rot) == len(casos), "a divisão não perde casos")
    checar(dividir_para_ppi(casos, 10)[0][0].id == rot[0].id,
           "a divisão é determinística (semente fixa)")

    # O detector de problemas precisa detectar.
    quebrado = [Caso(id="x", pergunta="p", contexto="", resposta="r")]
    checar(any("vazio" in a for a in validar(quebrado)),
           "o validador encontra contexto vazio")


def testar_vieses_offline() -> None:
    secao("7. Sondagens de viés que não usam LLM")

    # Leniência
    juiz_leniente = [0.9, 0.9, 0.8, 0.9, 0.85]
    humano = [1.0, 0.0, 1.0, 0.0, 1.0]
    r = testar_leniencia(juiz_leniente, humano)
    checar(r.desvio > 0, "detecta juiz leniente")
    checar(r.tendencia == "leniente", "classifica como leniente")
    checar(r.concentracao >= 0.4, "detecta concentração das notas")

    juiz_severo = [0.1, 0.1, 0.2, 0.1, 0.15]
    checar(testar_leniencia(juiz_severo, humano).tendencia == "severo",
           "detecta juiz severo")

    calibrado = testar_leniencia([1.0, 0.0, 1.0, 0.0, 1.0], humano)
    checar(calibrado.tendencia == "calibrado", "detecta juiz calibrado")
    checar(testar_leniencia([], []).n == 0, "listas vazias não quebram")

    # Verbosidade
    respostas = ["a b", "a b c d e f", "a", "a b c d e f g h i j", "a b c"]
    notas_longas = [0.4, 0.8, 0.2, 1.0, 0.5]   # nota cresce com o comprimento
    v = testar_vies_verbosidade(respostas, notas_longas)
    checar(v.correlacao_juiz > 0.8, "detecta correlação com o comprimento")

    humano_plano = [0.5] * 5
    v2 = testar_vies_verbosidade(respostas, notas_longas, humano_plano)
    checar(v2.vies_excedente > 0.5,
           "viés excedente aparece quando o humano não premia comprimento")

    # Se o humano também premia o comprimento, não é viés do juiz.
    v3 = testar_vies_verbosidade(respostas, notas_longas, notas_longas)
    checar(perto(v3.vies_excedente, 0.0),
           "sem viés excedente quando humano e juiz concordam")


def testar_utilidades() -> None:
    secao("8. Utilidades")

    frases = dividir_frases("Primeira frase. Segunda frase! Terceira?")
    checar(len(frases) == 3, f"divide em 3 frases (obteve {len(frases)})")
    checar(dividir_frases("") == [], "texto vazio devolve lista vazia")
    checar(len(dividir_frases("Sem pontuação final")) == 1,
           "texto sem pontuação vira uma frase")


def main() -> int:
    print("=" * 64)
    print("  TESTES OFFLINE — sem API, sem custo")
    print("=" * 64)

    testar_lexicais()
    testar_concordancia_categorica()
    testar_correlacoes()
    testar_alinhamento()
    testar_ppi()
    testar_dataset()
    testar_vieses_offline()
    testar_utilidades()

    print("\n" + "=" * 64)
    if FALHAS:
        print(f"  {len(FALHAS)} FALHA(S):")
        for f in FALHAS:
            print(f"    - {f}")
        print("=" * 64)
        return 1

    print("  Todos os testes passaram.")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
