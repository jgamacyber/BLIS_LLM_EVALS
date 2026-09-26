"""
PPI — Prediction-Powered Inference: intervalos de confiança para juízes LLM.

Implementação da etapa 3 do ARES (Saad-Falcon, Khattab, Potts & Zaharia, 2024),
que por sua vez aplica o método de Angelopoulos et al. (2023).

O problema que resolve
----------------------

Você tem duas opções ruins para estimar a qualidade de um sistema de RAG:

1. **Só anotação humana.** Confiável, mas cara. Com 40 exemplos anotados, o
   intervalo de confiança é largo demais para distinguir dois sistemas que
   diferem em poucos pontos.

2. **Só juiz LLM.** Barato e escalável, mas **enviesado**. Se o juiz é leniente,
   a média dele é otimista — e nenhuma quantidade de dados não rotulados
   corrige um viés sistemático. Mais dados só dão mais confiança na resposta
   errada.

O PPI combina as duas. Usa o juiz LLM no conjunto grande **e** o conjunto pequeno
anotado para estimar e corrigir o viés do juiz:

    θ_PPI  =  média(f(X_não_rotulado))  −  [média(f(X_rotulado)) − média(Y)]
              └─ estimativa do juiz ─┘     └──── retificador do viés ────┘

O termo entre colchetes é a **função retificadora**: o quanto o juiz erra, em
média, onde podemos conferir. Subtraindo-o, obtém-se uma estimativa não
enviesada — com intervalo mais estreito do que usar só os rótulos humanos.

A variância soma as duas fontes de incerteza:

    Var(θ_PPI) = Var(f)/N  +  Var(Y − f)/n

com N = tamanho do conjunto não rotulado e n = tamanho do rotulado. Note a
consequência prática: **quanto melhor o juiz, menor Var(Y − f) e mais estreito o
intervalo**. Um juiz ruim não quebra o método — só o torna menos eficiente, e o
intervalo alarga honestamente para mostrar isso.

O ARES usa ~150 pontos anotados e α = 0,05 (95% de confiança).

Tudo aqui é aritmética pura: nenhuma chamada de API, nenhuma dependência.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# Quantis da normal padrão para os níveis de confiança usuais.
_Z = {0.10: 1.6449, 0.05: 1.9600, 0.01: 2.5758}


def z_critico(alpha: float) -> float:
    """Quantil z para (1-alpha) de confiança bicaudal."""
    if alpha in _Z:
        return _Z[alpha]
    # Aproximação de Beasley-Springer-Moro para os demais casos.
    p = 1 - alpha / 2
    if not 0 < p < 1:
        raise ValueError("alpha deve estar entre 0 e 1")
    t = math.sqrt(-2.0 * math.log(1 - p)) if p > 0.5 else math.sqrt(-2.0 * math.log(p))
    c = [2.515517, 0.802853, 0.010328]
    d = [1.432788, 0.189269, 0.001308]
    x = t - ((c[2] * t + c[1]) * t + c[0]) / (((d[2] * t + d[1]) * t + d[0]) * t + 1.0)
    return x if p > 0.5 else -x


def _media(v: list[float]) -> float:
    return sum(v) / len(v) if v else 0.0


def _variancia(v: list[float]) -> float:
    """Variância amostral (divisor n-1)."""
    n = len(v)
    if n < 2:
        return 0.0
    m = _media(v)
    return sum((x - m) ** 2 for x in v) / (n - 1)


@dataclass
class IntervaloPPI:
    """Estimativa pontual e intervalo de confiança."""

    estimativa: float = 0.0
    limite_inferior: float = 0.0
    limite_superior: float = 0.0
    alpha: float = 0.05

    # diagnóstico
    media_juiz_nao_rotulado: float = 0.0
    media_juiz_rotulado: float = 0.0
    media_humana: float = 0.0
    retificador: float = 0.0
    n_rotulado: int = 0
    n_nao_rotulado: int = 0
    metodo: str = "ppi"

    @property
    def largura(self) -> float:
        return self.limite_superior - self.limite_inferior

    @property
    def margem(self) -> float:
        return self.largura / 2

    def contem(self, valor: float) -> bool:
        return self.limite_inferior <= valor <= self.limite_superior

    def __str__(self) -> str:
        confianca = int(round((1 - self.alpha) * 100))
        return (
            f"{self.estimativa:.3f} "
            f"[{self.limite_inferior:.3f}, {self.limite_superior:.3f}] "
            f"({confianca}%)"
        )

    def imprimir(self, nome: str = "") -> None:
        titulo = nome or "Estimativa"
        confianca = int(round((1 - self.alpha) * 100))
        print(f"\n  {titulo}  ({self.metodo})")
        print("  " + "-" * 58)
        print(f"    estimativa           {self.estimativa:>7.3f}")
        print(f"    IC {confianca}%              "
              f"[{self.limite_inferior:.3f}, {self.limite_superior:.3f}]  "
              f"(±{self.margem:.3f})")
        if self.metodo == "ppi":
            print(f"    média do juiz        {self.media_juiz_nao_rotulado:>7.3f}   "
                  f"(nos {self.n_nao_rotulado} não rotulados)")
            print(f"    retificador do viés  {self.retificador:>+7.3f}   "
                  f"(medido nos {self.n_rotulado} rotulados)")
            tendencia = (
                "juiz leniente" if self.retificador > 0.01
                else "juiz severo" if self.retificador < -0.01
                else "juiz calibrado"
            )
            print(f"    {'':<21}{'':<7}   {tendencia}")


# --------------------------------------------------------------------------- #
# Estimadores
# --------------------------------------------------------------------------- #


def intervalo_classico(
    rotulos_humanos: list[float], alpha: float = 0.05
) -> IntervaloPPI:
    """
    Intervalo usando SOMENTE as anotações humanas.

    É o baseline honesto, mas caro: com n pequeno, a margem fica larga. Serve
    de comparação para mostrar quanto o PPI estreita o intervalo.
    """
    n = len(rotulos_humanos)
    if n == 0:
        return IntervaloPPI(alpha=alpha, metodo="classico")

    media = _media(rotulos_humanos)
    erro_padrao = math.sqrt(_variancia(rotulos_humanos) / n) if n > 1 else 0.0
    margem = z_critico(alpha) * erro_padrao

    return IntervaloPPI(
        estimativa=media,
        limite_inferior=media - margem,
        limite_superior=media + margem,
        alpha=alpha,
        media_humana=media,
        n_rotulado=n,
        metodo="classico",
    )


def intervalo_so_juiz(
    predicoes: list[float], alpha: float = 0.05
) -> IntervaloPPI:
    """
    Intervalo usando SOMENTE o juiz LLM.

    ATENÇÃO: este intervalo é estatisticamente inválido como estimativa da
    qualidade real. Ele quantifica a incerteza sobre a MÉDIA DO JUIZ, não sobre
    a verdade. Se o juiz é enviesado, o intervalo é estreito e está errado —
    e aumentar N só o torna mais estreito e igualmente errado.

    Está implementado para que a comparação com o PPI mostre exatamente isso.
    """
    n = len(predicoes)
    if n == 0:
        return IntervaloPPI(alpha=alpha, metodo="so_juiz")

    media = _media(predicoes)
    erro_padrao = math.sqrt(_variancia(predicoes) / n) if n > 1 else 0.0
    margem = z_critico(alpha) * erro_padrao

    return IntervaloPPI(
        estimativa=media,
        limite_inferior=media - margem,
        limite_superior=media + margem,
        alpha=alpha,
        media_juiz_nao_rotulado=media,
        n_nao_rotulado=n,
        metodo="so_juiz",
    )


def intervalo_ppi(
    predicoes_rotulado: list[float],
    rotulos_humanos: list[float],
    predicoes_nao_rotulado: list[float],
    alpha: float = 0.05,
) -> IntervaloPPI:
    """
    Estimativa PPI com intervalo de confiança.

    Parâmetros
    ----------
    predicoes_rotulado
        Notas do juiz LLM nos casos que TAMBÉM têm anotação humana.
    rotulos_humanos
        As anotações humanas correspondentes (mesma ordem).
    predicoes_nao_rotulado
        Notas do juiz LLM nos casos SEM anotação humana (o conjunto grande).
    alpha
        1 - nível de confiança. 0.05 dá 95%, como no ARES.

    Se não houver casos não rotulados, o método degrada graciosamente para o
    intervalo clássico — que é a resposta correta nessa situação.
    """
    n = len(rotulos_humanos)
    N = len(predicoes_nao_rotulado)

    if len(predicoes_rotulado) != n:
        raise ValueError(
            "predicoes_rotulado e rotulos_humanos devem ter o mesmo comprimento"
        )
    if n == 0:
        return intervalo_so_juiz(predicoes_nao_rotulado, alpha)
    if N == 0:
        resultado = intervalo_classico(rotulos_humanos, alpha)
        resultado.media_juiz_rotulado = _media(predicoes_rotulado)
        return resultado

    # Retificador: o viés médio do juiz, medido onde há gabarito.
    residuos = [f - y for f, y in zip(predicoes_rotulado, rotulos_humanos)]
    retificador = _media(residuos)

    media_nao_rotulado = _media(predicoes_nao_rotulado)
    estimativa = media_nao_rotulado - retificador

    # Variância: incerteza do juiz no conjunto grande + incerteza do retificador.
    var_total = _variancia(predicoes_nao_rotulado) / N + _variancia(residuos) / n
    margem = z_critico(alpha) * math.sqrt(max(var_total, 0.0))

    return IntervaloPPI(
        estimativa=estimativa,
        limite_inferior=estimativa - margem,
        limite_superior=estimativa + margem,
        alpha=alpha,
        media_juiz_nao_rotulado=media_nao_rotulado,
        media_juiz_rotulado=_media(predicoes_rotulado),
        media_humana=_media(rotulos_humanos),
        retificador=retificador,
        n_rotulado=n,
        n_nao_rotulado=N,
        metodo="ppi",
    )


# --------------------------------------------------------------------------- #
# Comparação de sistemas
# --------------------------------------------------------------------------- #


@dataclass
class ComparacaoSistemas:
    """Resultado de comparar dois sistemas de RAG."""

    nome_a: str
    nome_b: str
    intervalo_a: IntervaloPPI
    intervalo_b: IntervaloPPI

    @property
    def diferenca(self) -> float:
        return self.intervalo_a.estimativa - self.intervalo_b.estimativa

    @property
    def intervalos_se_sobrepoem(self) -> bool:
        return not (
            self.intervalo_a.limite_inferior > self.intervalo_b.limite_superior
            or self.intervalo_b.limite_inferior > self.intervalo_a.limite_superior
        )

    @property
    def veredito(self) -> str:
        if self.intervalos_se_sobrepoem:
            return "diferença não conclusiva (intervalos se sobrepõem)"
        vencedor = self.nome_a if self.diferenca > 0 else self.nome_b
        return f"{vencedor} é melhor"

    def imprimir(self) -> None:
        print(f"\n  {self.nome_a}: {self.intervalo_a}")
        print(f"  {self.nome_b}: {self.intervalo_b}")
        print(f"  diferença: {self.diferenca:+.3f}")
        print(f"  veredito:  {self.veredito}")
        if self.intervalos_se_sobrepoem:
            print("  → colete mais anotações humanas ou avalie mais casos "
                  "antes de concluir.")


def comparar_sistemas(
    nome_a: str,
    intervalo_a: IntervaloPPI,
    nome_b: str,
    intervalo_b: IntervaloPPI,
) -> ComparacaoSistemas:
    """
    Compara dois sistemas pelos intervalos.

    A regra de sobreposição é conservadora: intervalos que se sobrepõem NÃO
    provam ausência de diferença, mas também não permitem afirmá-la. É
    exatamente a disciplina que o ARES traz para a comparação de sistemas de
    RAG — evitar declarar vencedor com base em uma diferença de 2 pontos numa
    amostra de 30 casos.
    """
    return ComparacaoSistemas(nome_a, nome_b, intervalo_a, intervalo_b)


def tabela_intervalos(intervalos: dict[str, IntervaloPPI]) -> None:
    """Tabela comparativa entre métodos de estimação."""
    if not intervalos:
        return

    print("\nComparação de métodos de estimação")
    print("-" * 78)
    print(f"{'Método':<24} {'Estimativa':>11} {'IC inferior':>12} "
          f"{'IC superior':>12} {'Largura':>10}")
    print("-" * 78)
    for nome, i in intervalos.items():
        print(
            f"{nome:<24} {i.estimativa:>11.3f} {i.limite_inferior:>12.3f} "
            f"{i.limite_superior:>12.3f} {i.largura:>10.3f}"
        )
    print("-" * 78)
