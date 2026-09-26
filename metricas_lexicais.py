"""
Métricas lexicais — os baselines baratos.

Não são sofisticadas, mas são **grátis, determinísticas e informativas**.
Thakur et al. (2025), *Judging the Judges*, encontram um resultado incômodo para
quem quer usar LLM para tudo: a métrica lexical `contains` tem alinhamento com
humanos comparável ao de juízes LLM de porte médio na tarefa de **ranquear**
modelos — porque, embora erre mais, seus vieses são mais consistentes.

A conclusão prática: sempre reporte o baseline lexical ao lado do juiz LLM. Se o
juiz caro não supera o `contains`, ele não está pagando o próprio custo.
"""

from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter
from dataclasses import dataclass


def remover_acentos(texto: str) -> str:
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


# Artigos e preposições que não devem pesar na comparação de respostas curtas.
ARTIGOS_PT = {"o", "a", "os", "as", "um", "uma", "uns", "umas", "de", "do", "da",
              "dos", "das", "em", "no", "na", "nos", "nas"}


def normalizar(texto: str, remover_artigos: bool = True) -> str:
    """
    Normalização padrão de QA: minúsculas, sem acentos, sem pontuação,
    sem artigos, espaços colapsados.

    É a mesma receita do SQuAD, adaptada ao português.
    """
    texto = remover_acentos(str(texto).lower())
    texto = "".join(c if c not in string.punctuation else " " for c in texto)
    tokens = texto.split()
    if remover_artigos:
        tokens = [t for t in tokens if t not in ARTIGOS_PT]
    return " ".join(tokens)


def tokenizar(texto: str) -> list[str]:
    return normalizar(texto).split()


# --------------------------------------------------------------------------- #
# Métricas
# --------------------------------------------------------------------------- #


def exact_match(resposta: str, gabarito) -> float:
    """
    Correspondência exata após normalização.

    Rigorosa demais na prática: penaliza respostas corretas mas verbosas. É o
    piso — se o EM já vai bem, talvez você nem precise de um juiz LLM.
    """
    alternativas = gabarito if isinstance(gabarito, list) else [gabarito]
    alvo = normalizar(resposta)
    return 1.0 if any(normalizar(a) == alvo for a in alternativas) else 0.0


def contains(resposta: str, gabarito) -> float:
    """
    O gabarito aparece dentro da resposta.

    É a métrica que Thakur et al. destacam: mais permissiva que o EM, tolera
    verbosidade, e ranqueia modelos quase tão bem quanto juízes LLM medianos.
    Falha em negações ("a resposta NÃO é Canberra" conta como acerto).
    """
    alternativas = gabarito if isinstance(gabarito, list) else [gabarito]
    alvo = normalizar(resposta)
    return 1.0 if any(normalizar(a) in alvo for a in alternativas) else 0.0


def f1_tokens(resposta: str, gabarito) -> float:
    """
    F1 sobre a sobreposição de tokens (métrica do SQuAD).

    Meio-termo entre EM e contains: dá crédito parcial por acerto parcial.
    Com lista de gabaritos, devolve o melhor F1.
    """
    alternativas = gabarito if isinstance(gabarito, list) else [gabarito]
    tokens_resposta = tokenizar(resposta)
    melhor = 0.0

    for alternativa in alternativas:
        tokens_alvo = tokenizar(str(alternativa))
        if not tokens_resposta or not tokens_alvo:
            melhor = max(melhor, float(tokens_resposta == tokens_alvo))
            continue

        comuns = Counter(tokens_resposta) & Counter(tokens_alvo)
        n_comuns = sum(comuns.values())
        if n_comuns == 0:
            continue

        precisao = n_comuns / len(tokens_resposta)
        revocacao = n_comuns / len(tokens_alvo)
        melhor = max(melhor, 2 * precisao * revocacao / (precisao + revocacao))

    return melhor


def cobertura_gabarito(resposta: str, gabarito) -> float:
    """Fração dos tokens do gabarito presentes na resposta (revocação pura)."""
    alternativas = gabarito if isinstance(gabarito, list) else [gabarito]
    tokens_resposta = set(tokenizar(resposta))
    melhor = 0.0
    for alternativa in alternativas:
        tokens_alvo = tokenizar(str(alternativa))
        if not tokens_alvo:
            continue
        presentes = sum(1 for t in tokens_alvo if t in tokens_resposta)
        melhor = max(melhor, presentes / len(tokens_alvo))
    return melhor


# --------------------------------------------------------------------------- #
# Detecção de abstenção
# --------------------------------------------------------------------------- #

PADROES_ABSTENCAO = [
    r"n[ãa]o\s+(sei|tenho|disponho|consigo|posso|h[áa])",
    r"n[ãa]o\s+(é|e)\s+poss[íi]vel",
    r"sem\s+informa[çc][ãa]o",
    r"informa[çc][ãa]o\s+insuficiente",
    r"n[ãa]o\s+cont[ée]m\s+informa[çc][ãa]o",
    r"contexto\s+fornecido\s+n[ãa]o",
    r"insufficient\s+information",
]


def eh_abstencao(resposta: str) -> bool:
    """
    Detecta se a resposta é uma recusa honesta.

    Importa muito numa avaliação de RAG: um sistema que se abstém quando não
    sabe é melhor que um que inventa, mas as métricas de acerto tratam os dois
    como erro. Separar a abstenção do erro factual muda a leitura do resultado.
    """
    texto = remover_acentos(str(resposta).lower())
    return any(re.search(p, texto) for p in PADROES_ABSTENCAO)


METRICAS = {
    "exact_match": exact_match,
    "contains": contains,
    "f1_tokens": f1_tokens,
    "cobertura": cobertura_gabarito,
}


@dataclass
class ResultadoLexical:
    exact_match: float = 0.0
    contains: float = 0.0
    f1_tokens: float = 0.0
    cobertura: float = 0.0
    abstencao: bool = False

    def como_dict(self) -> dict:
        return {
            "exact_match": self.exact_match,
            "contains": self.contains,
            "f1_tokens": self.f1_tokens,
            "cobertura": self.cobertura,
            "abstencao": self.abstencao,
        }


def avaliar_lexical(resposta: str, gabarito) -> ResultadoLexical:
    """
    Aplica todas as métricas lexicais de uma vez. Custo zero.

    Caso especial importante: quando NÃO existe gabarito, a pergunta não tem
    resposta possível a partir do contexto. Aí a resposta correta é a
    abstenção — e é assim que se pontua.

    Sem esse tratamento, um sistema que acerta ao recusar seria contado como
    erro por todas as métricas, o que inverte o sinal justamente nos casos que
    mais importam numa avaliação de RAG.
    """
    abstencao = eh_abstencao(resposta)

    if gabarito in (None, "", []):
        nota = 1.0 if abstencao else 0.0
        return ResultadoLexical(
            exact_match=nota,
            contains=nota,
            f1_tokens=nota,
            cobertura=nota,
            abstencao=abstencao,
        )

    return ResultadoLexical(
        exact_match=exact_match(resposta, gabarito),
        contains=contains(resposta, gabarito),
        f1_tokens=f1_tokens(resposta, gabarito),
        cobertura=cobertura_gabarito(resposta, gabarito),
        abstencao=abstencao,
    )
