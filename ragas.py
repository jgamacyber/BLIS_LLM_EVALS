"""
RAGAS — avaliação de RAG sem gabarito.

Implementação de Es et al. (2023), *Ragas: Automated Evaluation of Retrieval
Augmented Generation*.

O ponto do artigo: avaliar RAG normalmente exige respostas de referência
anotadas à mão, o que é caro e lento. O RAGAS propõe três métricas **livres de
referência** (reference-free), que só precisam da pergunta, do contexto
recuperado e da resposta gerada.

    1. Faithfulness (F)      — a resposta está fundamentada no contexto?
    2. Answer Relevance (AR) — a resposta endereça a pergunta feita?
    3. Context Relevance (CR)— o contexto recuperado é focado, sem entulho?

As três são calculadas decompondo o problema em sub-tarefas que um LLM faz bem
(extrair afirmações, verificar suporte, gerar perguntas) em vez de pedir uma
nota direta — que é justamente o modo de falha que o G-Eval também ataca.

As três se separam por componente: CR avalia o **recuperador**, F e AR avaliam o
**gerador**. Quando a resposta está ruim, é isso que diz de quem é a culpa.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from config import SETTINGS, Settings, get_client

# --------------------------------------------------------------------------- #
# Prompts (traduzidos dos originais do artigo)
# --------------------------------------------------------------------------- #

PROMPT_EXTRAIR_AFIRMACOES = """Dada uma pergunta e uma resposta, crie uma ou mais \
afirmações a partir de cada frase da resposta.

Cada afirmação deve ser uma asserção curta, autocontida e verificável \
isoladamente. Decomponha frases longas em asserções mais curtas e focadas.

pergunta: {pergunta}
resposta: {resposta}

Responda APENAS com as afirmações, uma por linha, sem numeração e sem comentários."""

PROMPT_VERIFICAR = """Considere o contexto dado e as afirmações a seguir, e \
determine se elas são sustentadas pela informação presente no contexto.

Forneça uma breve explicação para cada afirmação antes de chegar ao veredito \
(Sim/Não). Forneça um veredito final para cada afirmação, em ordem, ao final, \
no formato indicado. Não se desvie do formato especificado.

contexto:
{contexto}

afirmações:
{afirmacoes}

Formato da saída — uma linha por afirmação, ao final:
VEREDITO 1: Sim
VEREDITO 2: Não
..."""

PROMPT_GERAR_PERGUNTA = """Gere uma pergunta para a resposta dada.

resposta: {resposta}

Responda APENAS com a pergunta, sem comentários."""

PROMPT_EXTRAIR_SENTENCAS = """Extraia do contexto fornecido as frases relevantes \
que possam ajudar a responder à pergunta a seguir.

Se nenhuma frase relevante for encontrada, ou se você acreditar que a pergunta \
não pode ser respondida a partir do contexto dado, responda exatamente com a \
frase "Informação Insuficiente".

Ao extrair as frases candidatas, você não pode fazer nenhuma alteração nas \
frases do contexto — copie-as literalmente.

pergunta: {pergunta}

contexto:
{contexto}"""

FRASE_INSUFICIENTE = "informação insuficiente"


# --------------------------------------------------------------------------- #
# Resultado
# --------------------------------------------------------------------------- #


@dataclass
class ResultadoRagas:
    """As três métricas, com o rastro de como cada uma foi obtida."""

    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    context_relevance: float = 0.0

    afirmacoes: list[str] = field(default_factory=list)
    vereditos: list[bool] = field(default_factory=list)
    perguntas_geradas: list[str] = field(default_factory=list)
    sentencas_extraidas: int = 0
    sentencas_totais: int = 0
    contexto_insuficiente: bool = False
    erros: list[str] = field(default_factory=list)

    @property
    def media_harmonica(self) -> float:
        """
        Combina as três em um número só.

        Média harmônica de propósito: ela pune desequilíbrio. Um sistema com
        faithfulness 1.0 e context relevance 0.1 não deve parecer mediano.
        """
        valores = [self.faithfulness, self.answer_relevance, self.context_relevance]
        if any(v <= 0 for v in valores):
            return 0.0
        return len(valores) / sum(1 / v for v in valores)

    def imprimir(self, detalhado: bool = False) -> None:
        print(f"    faithfulness       {self.faithfulness:>6.1%}   "
              f"({sum(self.vereditos)}/{len(self.vereditos)} afirmações sustentadas)")
        print(f"    answer relevance   {self.answer_relevance:>6.1%}   "
              f"({len(self.perguntas_geradas)} perguntas reversas)")
        print(f"    context relevance  {self.context_relevance:>6.1%}   "
              f"({self.sentencas_extraidas}/{self.sentencas_totais} frases úteis)")
        print(f"    média harmônica    {self.media_harmonica:>6.1%}")

        if self.contexto_insuficiente:
            print("    [!] o contexto foi julgado insuficiente para a pergunta")

        if detalhado and self.afirmacoes:
            print("\n    Afirmações extraídas da resposta:")
            for afirmacao, ok in zip(self.afirmacoes, self.vereditos):
                marca = "OK " if ok else "SEM SUPORTE"
                print(f"      [{marca:^11}] {afirmacao[:80]}")

        for erro in self.erros:
            print(f"    [erro] {erro}")


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #


def _chamar(prompt: str, settings: Settings, max_tokens: int = 600) -> str:
    client = get_client(settings)
    resposta = client.chat.completions.create(
        model=settings.juiz_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return (resposta.choices[0].message.content or "").strip()


def _linhas(texto: str) -> list[str]:
    """Linhas limpas, sem numeração nem marcadores."""
    saida = []
    for linha in (texto or "").splitlines():
        linha = linha.strip()
        if not linha:
            continue
        linha = re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", linha).strip()
        if linha:
            saida.append(linha)
    return saida


def dividir_frases(texto: str) -> list[str]:
    """Segmentação simples em frases, usada pelo denominador do CR."""
    texto = re.sub(r"\s+", " ", str(texto)).strip()
    if not texto:
        return []
    partes = re.split(r"(?<=[.!?])\s+(?=[A-ZÀ-Ú0-9])", texto)
    return [p.strip() for p in partes if p.strip()]


# --------------------------------------------------------------------------- #
# 1. Faithfulness
# --------------------------------------------------------------------------- #


def extrair_afirmacoes(
    pergunta: str, resposta: str, settings: Settings | None = None
) -> list[str]:
    """Decompõe a resposta em asserções curtas e verificáveis (passo 1 de F)."""
    settings = settings or SETTINGS
    texto = _chamar(
        PROMPT_EXTRAIR_AFIRMACOES.format(pergunta=pergunta, resposta=resposta),
        settings,
        400,
    )
    return _linhas(texto)


def verificar_afirmacoes(
    afirmacoes: list[str], contexto: str, settings: Settings | None = None
) -> list[bool]:
    """
    Verifica cada afirmação contra o contexto (passo 2 de F).

    Uma única chamada avalia todas as afirmações, como no artigo. Afirmações
    cujo veredito não puder ser lido são contadas como NÃO sustentadas — o
    padrão conservador, que evita inflar a métrica por falha de parsing.
    """
    settings = settings or SETTINGS
    if not afirmacoes:
        return []

    numeradas = "\n".join(f"{i}. {a}" for i, a in enumerate(afirmacoes, start=1))
    texto = _chamar(
        PROMPT_VERIFICAR.format(contexto=contexto, afirmacoes=numeradas),
        settings,
        800,
    )

    vereditos: dict[int, bool] = {}
    for match in re.finditer(
        r"VEREDITO\s*(\d+)\s*:\s*(sim|não|nao|yes|no)", texto, re.IGNORECASE
    ):
        indice = int(match.group(1))
        valor = match.group(2).lower()
        vereditos[indice] = valor in {"sim", "yes"}

    return [vereditos.get(i, False) for i in range(1, len(afirmacoes) + 1)]


def faithfulness(
    pergunta: str, resposta: str, contexto: str, settings: Settings | None = None
) -> tuple[float, list[str], list[bool]]:
    """
    F = |V| / |S| — afirmações sustentadas sobre o total de afirmações.

    Mede alucinação: quanto da resposta pode ser rastreado até o contexto.
    Uma resposta curta e cautelosa pontua alto; uma resposta rica em detalhes
    inventados pontua baixo, mesmo que "pareça" melhor.
    """
    settings = settings or SETTINGS
    afirmacoes = extrair_afirmacoes(pergunta, resposta, settings)
    if not afirmacoes:
        return 0.0, [], []

    vereditos = verificar_afirmacoes(afirmacoes, contexto, settings)
    return sum(vereditos) / len(vereditos), afirmacoes, vereditos


# --------------------------------------------------------------------------- #
# 2. Answer Relevance
# --------------------------------------------------------------------------- #


def _cosseno(a: list[float], b: list[float]) -> float:
    produto = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return produto / (na * nb)


def _embeddings(textos: list[str], settings: Settings) -> list[list[float]]:
    client = get_client(settings)
    resposta = client.embeddings.create(
        model=settings.embedding_model, input=textos
    )
    itens = sorted(resposta.data, key=lambda d: d.index)
    return [list(i.embedding) for i in itens]


def answer_relevance(
    pergunta: str,
    resposta: str,
    settings: Settings | None = None,
    n: int | None = None,
) -> tuple[float, list[str]]:
    """
    AR = (1/n) Σ sim(q, q_i)  —  Eq. 1 do artigo.

    A ideia é engenhosa: em vez de pedir uma nota, pede-se ao LLM que gere n
    perguntas a partir da RESPOSTA, e mede-se o quanto essas perguntas
    reconstruídas se parecem com a pergunta original.

    Se a resposta endereça a pergunta, a pergunta reconstruída a partir dela
    será parecida com a original. Se a resposta é evasiva, incompleta ou cheia
    de informação redundante, a reconstrução diverge.

    Note que AR **não** mede factualidade — só alinhamento com a pergunta.
    """
    settings = settings or SETTINGS
    n = n or settings.ragas_n_perguntas

    geradas: list[str] = []
    for _ in range(max(1, n)):
        texto = _chamar(PROMPT_GERAR_PERGUNTA.format(resposta=resposta), settings, 100)
        linhas = _linhas(texto)
        if linhas:
            geradas.append(linhas[0])

    if not geradas:
        return 0.0, []

    vetores = _embeddings([pergunta] + geradas, settings)
    vetor_original, vetores_gerados = vetores[0], vetores[1:]

    similaridades = [_cosseno(vetor_original, v) for v in vetores_gerados]
    return sum(similaridades) / len(similaridades), geradas


# --------------------------------------------------------------------------- #
# 3. Context Relevance
# --------------------------------------------------------------------------- #


def context_relevance(
    pergunta: str, contexto: str, settings: Settings | None = None
) -> tuple[float, int, int, bool]:
    """
    CR = frases extraídas / frases totais no contexto  —  Eq. 2 do artigo.

    Penaliza contexto inflado. Recuperar 10 passagens em que só uma importa dá
    CR baixo mesmo que a resposta saia certa — e isso é deliberado: contexto
    redundante custa tokens e, como mostra o *Lost in the Middle*, atrapalha o
    gerador a encontrar o que importa.

    Devolve (CR, n_extraidas, n_totais, insuficiente).
    """
    settings = settings or SETTINGS

    frases_contexto = dividir_frases(contexto)
    total = len(frases_contexto)
    if total == 0:
        return 0.0, 0, 0, True

    texto = _chamar(
        PROMPT_EXTRAIR_SENTENCAS.format(pergunta=pergunta, contexto=contexto),
        settings,
        800,
    )

    if FRASE_INSUFICIENTE in texto.lower():
        return 0.0, 0, total, True

    extraidas = [f for f in _linhas(texto) if len(f) > 15]
    n_extraidas = min(len(extraidas), total)  # o LLM às vezes repete frases
    return n_extraidas / total, n_extraidas, total, False


# --------------------------------------------------------------------------- #
# Avaliação completa
# --------------------------------------------------------------------------- #


def avaliar(
    pergunta: str,
    resposta: str,
    contexto: str,
    settings: Settings | None = None,
    metricas: list[str] | None = None,
) -> ResultadoRagas:
    """
    Calcula as três métricas do RAGAS para um caso.

    Custo: cerca de 2 + n chamadas de chat e 1 de embeddings por caso (com
    n = ragas_n_perguntas). Falhas em uma métrica não derrubam as outras.
    """
    settings = settings or SETTINGS
    metricas = metricas or ["faithfulness", "answer_relevance", "context_relevance"]
    resultado = ResultadoRagas()

    if "faithfulness" in metricas:
        try:
            f, afirmacoes, vereditos = faithfulness(
                pergunta, resposta, contexto, settings
            )
            resultado.faithfulness = f
            resultado.afirmacoes = afirmacoes
            resultado.vereditos = vereditos
        except Exception as erro:  # noqa: BLE001
            resultado.erros.append(f"faithfulness: {erro}")

    if "answer_relevance" in metricas:
        try:
            ar, geradas = answer_relevance(pergunta, resposta, settings)
            resultado.answer_relevance = ar
            resultado.perguntas_geradas = geradas
        except Exception as erro:  # noqa: BLE001
            resultado.erros.append(f"answer_relevance: {erro}")

    if "context_relevance" in metricas:
        try:
            cr, extraidas, total, insuficiente = context_relevance(
                pergunta, contexto, settings
            )
            resultado.context_relevance = cr
            resultado.sentencas_extraidas = extraidas
            resultado.sentencas_totais = total
            resultado.contexto_insuficiente = insuficiente
        except Exception as erro:  # noqa: BLE001
            resultado.erros.append(f"context_relevance: {erro}")

    return resultado
