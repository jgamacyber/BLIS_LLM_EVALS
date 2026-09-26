"""
Concordância entre juízes — o LLM concorda com a anotação humana?

Este é o módulo que responde à pergunta que dá sentido a todo o resto: **o juiz
automático mede o que o humano mediria?** Sem isso, as métricas dos outros
módulos são números sem referencial.

Thakur et al. (2025), *Judging the Judges*, fazem uma advertência central:

> "Nossa análise destaca a necessidade de métricas de alinhamento além do
> percentual de concordância, pois juízes com alta concordância ainda podem
> atribuir notas muito diferentes."

Por isso aqui há três famílias de medida:

1. **Concordância bruta** (% de acordo) — fácil de entender e enganosa. Se 90%
   dos casos são positivos, um juiz que diz "sim" sempre acerta 90%.
2. **Concordância corrigida pelo acaso** — Scott's π e Cohen's κ descontam o
   acordo que aconteceria por sorte. O artigo mostra que π discrimina juízes
   que o percentual não distingue.
3. **Correlação de ranking** — Spearman ρ e Kendall τ, as métricas que o G-Eval
   (Liu et al., 2023) usa para se comparar com humanos. Medem se o juiz
   *ordena* igual, mesmo que a escala esteja deslocada.

Tudo implementado em Python puro, sem scipy — e sem custo de API.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# Concordância em rótulos categóricos
# --------------------------------------------------------------------------- #


def concordancia_bruta(a: list, b: list) -> float:
    """Fração de itens em que os dois avaliadores deram o mesmo rótulo."""
    if len(a) != len(b):
        raise ValueError("as duas listas devem ter o mesmo comprimento")
    if not a:
        return 0.0
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def _distribuicao(valores: list) -> dict:
    total = len(valores)
    dist: dict = {}
    for v in valores:
        dist[v] = dist.get(v, 0) + 1
    return {k: n / total for k, n in dist.items()}


def scott_pi(a: list, b: list) -> float:
    """
    Scott's π — concordância corrigida pelo acaso.

        π = (Po - Pe) / (1 - Pe)

    Po é a concordância observada. Pe, o acaso esperado, usa a distribuição
    **conjunta** dos dois avaliadores: assume que ambos partem da mesma
    distribuição marginal de rótulos.

        Pe = Σ_k ((p_a(k) + p_b(k)) / 2)²

    É a métrica que Thakur et al. recomendam sobre o percentual bruto. Difere do
    κ de Cohen justamente por essa suposição: π presume marginais compartilhadas,
    κ presume marginais independentes. Com juízes que têm vieses diferentes
    (um mais leniente que o outro), os dois divergem — e a divergência é
    informativa.

    Faixa: 1.0 é concordância perfeita, 0.0 é o nível do acaso, negativo é pior
    que o acaso.
    """
    if len(a) != len(b):
        raise ValueError("as duas listas devem ter o mesmo comprimento")
    if not a:
        return 0.0

    po = concordancia_bruta(a, b)

    dist_a = _distribuicao(a)
    dist_b = _distribuicao(b)
    categorias = set(dist_a) | set(dist_b)

    pe = sum(
        ((dist_a.get(k, 0.0) + dist_b.get(k, 0.0)) / 2) ** 2 for k in categorias
    )

    if pe >= 1.0:
        # Todos deram o mesmo rótulo o tempo todo: o acaso explica tudo,
        # e π é indefinido. Devolvemos 1.0 se de fato concordaram.
        return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1 - pe)


def cohen_kappa(a: list, b: list) -> float:
    """
    Cohen's κ — também corrigido pelo acaso, mas com marginais independentes.

        Pe = Σ_k p_a(k) · p_b(k)

    Reportado junto com π para deixar visível quando os dois avaliadores têm
    distribuições de rótulo diferentes (ou seja, quando um é mais leniente).
    """
    if len(a) != len(b):
        raise ValueError("as duas listas devem ter o mesmo comprimento")
    if not a:
        return 0.0

    po = concordancia_bruta(a, b)
    dist_a = _distribuicao(a)
    dist_b = _distribuicao(b)
    categorias = set(dist_a) | set(dist_b)

    pe = sum(dist_a.get(k, 0.0) * dist_b.get(k, 0.0) for k in categorias)
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1 - pe)


def interpretar_kappa(valor: float) -> str:
    """
    Rótulo qualitativo, na escala de Landis & Koch (1977).

    Use com parcimônia: são faixas convencionais, não verdades estatísticas.
    """
    if valor < 0:
        return "pior que o acaso"
    if valor < 0.20:
        return "desprezível"
    if valor < 0.40:
        return "fraca"
    if valor < 0.60:
        return "moderada"
    if valor < 0.80:
        return "substancial"
    return "quase perfeita"


# --------------------------------------------------------------------------- #
# Correlação em notas contínuas
# --------------------------------------------------------------------------- #


def _postos(valores: list[float]) -> list[float]:
    """Postos com média em caso de empate (necessário para Spearman)."""
    indexados = sorted(range(len(valores)), key=lambda i: valores[i])
    postos = [0.0] * len(valores)

    i = 0
    while i < len(indexados):
        j = i
        while j + 1 < len(indexados) and valores[indexados[j + 1]] == valores[indexados[i]]:
            j += 1
        posto_medio = (i + j) / 2 + 1
        for k in range(i, j + 1):
            postos[indexados[k]] = posto_medio
        i = j + 1

    return postos


def pearson(x: list[float], y: list[float]) -> float:
    """Correlação linear de Pearson."""
    if len(x) != len(y):
        raise ValueError("as duas listas devem ter o mesmo comprimento")
    n = len(x)
    if n < 2:
        return 0.0

    media_x = sum(x) / n
    media_y = sum(y) / n

    cov = sum((a - media_x) * (b - media_y) for a, b in zip(x, y))
    var_x = sum((a - media_x) ** 2 for a in x)
    var_y = sum((b - media_y) ** 2 for b in y)

    if var_x == 0 or var_y == 0:
        return 0.0
    return cov / math.sqrt(var_x * var_y)


def spearman(x: list[float], y: list[float]) -> float:
    """
    Correlação de Spearman: Pearson sobre os postos.

    É a métrica principal do G-Eval — que reporta ρ = 0,514 com humanos no
    SummEval, contra 0,192 do ROUGE-1. Insensível a deslocamento de escala:
    um juiz que dá sempre 2 pontos a mais ainda correlaciona 1,0.
    """
    if len(x) < 2:
        return 0.0
    return pearson(_postos(x), _postos(y))


def kendall_tau(x: list[float], y: list[float]) -> float:
    """
    Kendall τ-b: proporção líquida de pares concordantes, com correção de empates.

        τ = (C - D) / sqrt((C + D + Tx) · (C + D + Ty))

    O G-Eval reporta τ ao lado de ρ. O artigo faz uma ressalva importante: a
    variante sem probabilidades produz muitos empates, o que **infla** o τ sem
    refletir capacidade real de avaliação. Por isso reportamos também a
    proporção de empates.
    """
    if len(x) != len(y):
        raise ValueError("as duas listas devem ter o mesmo comprimento")
    n = len(x)
    if n < 2:
        return 0.0

    concordantes = discordantes = empates_x = empates_y = 0

    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            produto = dx * dy

            if dx == 0 and dy == 0:
                empates_x += 1
                empates_y += 1
            elif dx == 0:
                empates_x += 1
            elif dy == 0:
                empates_y += 1
            elif produto > 0:
                concordantes += 1
            else:
                discordantes += 1

    denominador = math.sqrt(
        (concordantes + discordantes + empates_x)
        * (concordantes + discordantes + empates_y)
    )
    if denominador == 0:
        return 0.0
    return (concordantes - discordantes) / denominador


def proporcao_empates(valores: list[float]) -> float:
    """
    Fração de pares empatados.

    Alta proporção de empates infla o Kendall τ. É a ressalva que o G-Eval faz
    sobre sua própria variante sem probabilidades — vale sempre reportar junto.
    """
    n = len(valores)
    if n < 2:
        return 0.0
    total = n * (n - 1) / 2
    empates = sum(
        1 for i in range(n) for j in range(i + 1, n) if valores[i] == valores[j]
    )
    return empates / total


# --------------------------------------------------------------------------- #
# Relatório consolidado
# --------------------------------------------------------------------------- #


@dataclass
class Alinhamento:
    """Alinhamento de um juiz automático com a referência humana."""

    nome: str = ""
    n: int = 0

    # categóricas
    concordancia: float = 0.0
    scott_pi: float = 0.0
    cohen_kappa: float = 0.0

    # contínuas
    spearman: float = 0.0
    kendall: float = 0.0
    pearson: float = 0.0
    empates: float = 0.0

    # viés de escala
    media_juiz: float = 0.0
    media_humana: float = 0.0
    detalhes: dict = field(default_factory=dict)

    @property
    def desvio_medio(self) -> float:
        """
        Quanto o juiz é mais leniente (positivo) ou mais severo (negativo).

        Thakur et al.: juízes com excelente alinhamento ainda diferem em até 5
        pontos das notas humanas. Correlação alta não significa escala correta.
        """
        return self.media_juiz - self.media_humana

    def imprimir(self) -> None:
        print(f"\n  {self.nome}  (n={self.n})")
        print("  " + "-" * 60)
        print(f"    concordância bruta   {self.concordancia:>7.1%}")
        print(f"    Scott's π            {self.scott_pi:>7.3f}   "
              f"({interpretar_kappa(self.scott_pi)})")
        print(f"    Cohen's κ            {self.cohen_kappa:>7.3f}")
        print(f"    Spearman ρ           {self.spearman:>7.3f}")
        print(f"    Kendall τ            {self.kendall:>7.3f}   "
              f"(empates: {self.empates:.0%})")
        print(f"    média juiz / humana  {self.media_juiz:>7.2f} / "
              f"{self.media_humana:.2f}")

        tendencia = (
            "mais leniente" if self.desvio_medio > 0.05
            else "mais severo" if self.desvio_medio < -0.05
            else "calibrado"
        )
        print(f"    desvio de escala     {self.desvio_medio:>+7.2f}   ({tendencia})")


def comparar_com_humano(
    notas_juiz: list[float],
    notas_humanas: list[float],
    nome: str = "juiz",
    limiar_binario: float = 0.5,
) -> Alinhamento:
    """
    Compara as notas de um juiz automático com a referência humana.

    Calcula tanto as métricas categóricas (binarizando pelo limiar) quanto as
    contínuas. As duas visões são necessárias: a categórica responde "acerta o
    veredito?" e a contínua responde "ordena igual?".
    """
    if len(notas_juiz) != len(notas_humanas):
        raise ValueError("as listas devem ter o mesmo comprimento")

    n = len(notas_juiz)
    if n == 0:
        return Alinhamento(nome=nome)

    binario_juiz = [1 if v >= limiar_binario else 0 for v in notas_juiz]
    binario_humano = [1 if v >= limiar_binario else 0 for v in notas_humanas]

    return Alinhamento(
        nome=nome,
        n=n,
        concordancia=concordancia_bruta(binario_juiz, binario_humano),
        scott_pi=scott_pi(binario_juiz, binario_humano),
        cohen_kappa=cohen_kappa(binario_juiz, binario_humano),
        spearman=spearman(notas_juiz, notas_humanas),
        kendall=kendall_tau(notas_juiz, notas_humanas),
        pearson=pearson(notas_juiz, notas_humanas),
        empates=proporcao_empates(notas_juiz),
        media_juiz=sum(notas_juiz) / n,
        media_humana=sum(notas_humanas) / n,
    )


def tabela_alinhamento(alinhamentos: list[Alinhamento]) -> None:
    """Tabela comparativa entre vários juízes/métricas."""
    if not alinhamentos:
        return

    print("\nAlinhamento com a referência humana")
    print("-" * 88)
    print(f"{'Métrica':<24} {'Concord.':>9} {'Scott π':>9} {'Spearman':>9} "
          f"{'Kendall':>9} {'Desvio':>9} {'Empates':>9}")
    print("-" * 88)

    for a in sorted(alinhamentos, key=lambda x: x.scott_pi, reverse=True):
        print(
            f"{a.nome:<24} {a.concordancia:>8.1%} {a.scott_pi:>9.3f} "
            f"{a.spearman:>9.3f} {a.kendall:>9.3f} {a.desvio_medio:>+9.2f} "
            f"{a.empates:>8.0%}"
        )
    print("-" * 88)
    print("  Scott π corrige o acaso; concordância bruta sozinha engana quando")
    print("  os rótulos são desbalanceados. Desvio > 0 indica juiz leniente.")
