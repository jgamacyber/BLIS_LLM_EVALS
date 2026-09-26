"""
G-Eval — avaliação com auto-CoT e pontuação ponderada por probabilidade.

Implementação de Liu et al. (2023), *G-Eval: NLG Evaluation using GPT-4 with
Better Human Alignment* (Microsoft Cognitive Services Research).

Três componentes:

1. **Prompt** com a introdução da tarefa e os critérios de avaliação.
2. **Auto-CoT**: o próprio LLM gera os *passos de avaliação* detalhados a partir
   dos critérios. Escrever esses passos à mão para cada dimensão é trabalhoso;
   o artigo mostra que o modelo os produz bem, e que eles melhoram o resultado.
3. **Pontuação ponderada**: em vez de aceitar o inteiro que o modelo cuspiu,
   calcula-se a esperança da nota sob a distribuição de probabilidade dos
   tokens de saída:

       score = Σ p(s_i) × s_i

Por que o passo 3 importa — os dois problemas que o artigo identifica na
pontuação direta:

- **Um dígito domina.** Numa escala de 1 a 5, o modelo responde "3" quase
  sempre. Variância baixa, correlação baixa com humanos.
- **Só saem inteiros.** Mesmo pedindo decimais. Isso gera muitos empates, e
  empates escondem diferenças reais entre textos.

A ponderação resolve os dois: produz uma nota contínua (2,59 em vez de 3) que
reflete a incerteza real do avaliador.

Resultado do artigo: G-Eval-4 atinge ρ = 0,514 com humanos no SummEval, contra
0,192 do ROUGE-1 e 0,225 do BERTScore.

**Ressalva do próprio artigo**, que vale repetir: juízes LLM têm viés a favor de
textos gerados por LLM. Usar G-Eval como sinal de recompensa para treinar um LLM
arrisca auto-reforço.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from config import SETTINGS, Settings, get_client

# --------------------------------------------------------------------------- #
# Critérios prontos, adaptados a RAG
# --------------------------------------------------------------------------- #


@dataclass
class Criterio:
    """Uma dimensão de avaliação: nome, definição e escala."""

    nome: str
    definicao: str
    escala_min: int = 1
    escala_max: int = 5
    passos: str = ""  # preenchido pelo auto-CoT

    def descricao_escala(self) -> str:
        return f"{self.nome} ({self.escala_min}-{self.escala_max})"


CRITERIOS = {
    "fundamentacao": Criterio(
        nome="Fundamentação",
        definicao=(
            "o quanto as afirmações da resposta são sustentadas pelo contexto "
            "fornecido. Uma resposta bem fundamentada não contém nenhuma "
            "informação que não possa ser verificada no contexto. Informação "
            "correta no mundo real, mas ausente do contexto, NÃO conta como "
            "fundamentada."
        ),
    ),
    "relevancia": Criterio(
        nome="Relevância",
        definicao=(
            "o quanto a resposta endereça diretamente a pergunta feita. "
            "Penalize respostas incompletas, evasivas, ou que incluam "
            "informação redundante não solicitada."
        ),
    ),
    "coerencia": Criterio(
        nome="Coerência",
        definicao=(
            "a qualidade coletiva das frases. A resposta deve ser bem "
            "estruturada e organizada, construindo um corpo coerente de "
            "informação, e não um amontoado de fatos soltos."
        ),
    ),
    "completude": Criterio(
        nome="Completude",
        definicao=(
            "o quanto a resposta cobre todos os aspectos da pergunta que o "
            "contexto permite responder. Penalize omissões de informação "
            "relevante que estava disponível no contexto."
        ),
    ),
    "concisao": Criterio(
        nome="Concisão",
        definicao=(
            "a ausência de verbosidade desnecessária. Penalize preâmbulos, "
            "repetições e reformulações da pergunta. Não penalize detalhes "
            "necessários para responder corretamente."
        ),
    ),
}


PROMPT_PASSOS = """Você vai avaliar textos em uma dimensão de qualidade.

Dimensão: {nome}
Definição: {definicao}
Escala: {minimo} a {maximo}, onde {minimo} é o pior e {maximo} é o melhor.

Escreva os passos detalhados de avaliação que um avaliador deve seguir para \
atribuir uma nota nessa dimensão.

Responda APENAS com os passos numerados, sem preâmbulo. No máximo 5 passos."""


PROMPT_AVALIACAO = """Você vai avaliar uma resposta gerada por um sistema de \
perguntas e respostas.

Leia e entenda estas instruções com atenção. Mantenha este documento aberto \
durante a revisão e consulte-o quando necessário.

Critério de avaliação:

{nome} ({minimo}-{maximo}) - {definicao}

Passos de avaliação:

{passos}

Exemplo a avaliar:

CONTEXTO FORNECIDO AO SISTEMA:
{contexto}

PERGUNTA:
{pergunta}

RESPOSTA GERADA:
{resposta}

Formulário de avaliação (APENAS a nota):
- {nome}:"""


# --------------------------------------------------------------------------- #
# Resultado
# --------------------------------------------------------------------------- #


@dataclass
class ResultadoGEval:
    """Nota de uma dimensão, com a distribuição que a originou."""

    criterio: str = ""
    score: float = 0.0
    score_bruto: int = 0
    distribuicao: dict = field(default_factory=dict)
    metodo: str = ""  # "logprobs" | "amostragem" | "direto"
    passos: str = ""
    erro: str = ""

    @property
    def normalizado(self) -> float:
        """Nota reescalada para [0, 1], para comparar com outras métricas."""
        maximo = SETTINGS.geval_escala_max
        if maximo <= 1:
            return 0.0
        return (self.score - 1) / (maximo - 1)

    def imprimir(self) -> None:
        print(f"    {self.criterio:<16} {self.score:>5.2f}  "
              f"(bruto {self.score_bruto}, via {self.metodo})")
        if self.distribuicao:
            partes = " ".join(
                f"{nota}:{p:.0%}" for nota, p in sorted(self.distribuicao.items())
            )
            print(f"      distribuição: {partes}")
        if self.erro:
            print(f"      [erro] {self.erro}")


# --------------------------------------------------------------------------- #
# Auto-CoT: gerar os passos de avaliação
# --------------------------------------------------------------------------- #

_CACHE_PASSOS: dict[str, str] = {}


def gerar_passos(
    criterio: Criterio, settings: Settings | None = None, usar_cache: bool = True
) -> str:
    """
    Auto-CoT: o LLM escreve os passos de avaliação a partir da definição.

    Os passos são estáveis para um dado critério, então ficam em cache na
    memória: gerar de novo a cada caso seria desperdício puro.
    """
    settings = settings or SETTINGS
    chave = f"{criterio.nome}|{settings.juiz_model}"

    if usar_cache and chave in _CACHE_PASSOS:
        return _CACHE_PASSOS[chave]

    client = get_client(settings)
    resposta = client.chat.completions.create(
        model=settings.juiz_model,
        messages=[
            {
                "role": "user",
                "content": PROMPT_PASSOS.format(
                    nome=criterio.nome,
                    definicao=criterio.definicao,
                    minimo=criterio.escala_min,
                    maximo=criterio.escala_max,
                ),
            }
        ],
        temperature=0.0,
        max_tokens=400,
    )
    passos = (resposta.choices[0].message.content or "").strip()
    _CACHE_PASSOS[chave] = passos
    return passos


def limpar_cache_passos() -> None:
    _CACHE_PASSOS.clear()


# --------------------------------------------------------------------------- #
# Pontuação
# --------------------------------------------------------------------------- #


def _extrair_nota(texto: str, minimo: int, maximo: int) -> int | None:
    """Extrai o primeiro inteiro dentro da escala."""
    for match in re.finditer(r"\b(\d+)\b", texto or ""):
        valor = int(match.group(1))
        if minimo <= valor <= maximo:
            return valor
    return None


def _score_por_logprobs(
    resposta_api, criterio: Criterio
) -> tuple[float, dict] | None:
    """
    Esperança da nota sob a distribuição dos tokens de saída.

        score = Σ p(s_i) × s_i

    Depende de a API devolver `logprobs`. Quando não devolve, o chamador cai
    para a amostragem — que é o que o próprio artigo faz com o GPT-4.
    """
    try:
        conteudo = resposta_api.choices[0].logprobs.content
    except (AttributeError, IndexError, TypeError):
        return None
    if not conteudo:
        return None

    # Procura a primeira posição cujo token seja um número da escala.
    for posicao in conteudo:
        candidatos: dict[int, float] = {}

        alternativas = getattr(posicao, "top_logprobs", None) or []
        for alternativa in alternativas:
            token = (getattr(alternativa, "token", "") or "").strip()
            if token.isdigit():
                valor = int(token)
                if criterio.escala_min <= valor <= criterio.escala_max:
                    p = math.exp(getattr(alternativa, "logprob", -math.inf))
                    candidatos[valor] = candidatos.get(valor, 0.0) + p

        if candidatos:
            massa = sum(candidatos.values())
            if massa <= 0:
                continue
            distribuicao = {k: v / massa for k, v in candidatos.items()}
            score = sum(nota * p for nota, p in distribuicao.items())
            return score, distribuicao

    return None


def _score_por_amostragem(
    prompt: str, criterio: Criterio, settings: Settings
) -> tuple[float, dict, int]:
    """
    Estima a distribuição amostrando n vezes com temperatura 1.

    É exatamente o que o artigo faz para o GPT-4, que não expunha logprobs:
    "n = 20, temperature = 1, top_p = 1 para amostrar 20 vezes e estimar as
    probabilidades dos tokens". Aqui o n padrão é menor, por custo.
    """
    client = get_client(settings)
    notas: list[int] = []

    for _ in range(max(1, settings.geval_amostras)):
        resposta = client.chat.completions.create(
            model=settings.juiz_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0,
            top_p=1.0,
            max_tokens=8,
        )
        nota = _extrair_nota(
            resposta.choices[0].message.content or "",
            criterio.escala_min,
            criterio.escala_max,
        )
        if nota is not None:
            notas.append(nota)

    if not notas:
        return 0.0, {}, 0

    contagem: dict[int, int] = {}
    for n in notas:
        contagem[n] = contagem.get(n, 0) + 1

    distribuicao = {k: v / len(notas) for k, v in contagem.items()}
    score = sum(nota * p for nota, p in distribuicao.items())
    moda = max(contagem.items(), key=lambda kv: kv[1])[0]
    return score, distribuicao, moda


def avaliar_criterio(
    pergunta: str,
    resposta: str,
    contexto: str,
    criterio: Criterio | str,
    settings: Settings | None = None,
    passos: str = "",
) -> ResultadoGEval:
    """
    Avalia uma dimensão com o pipeline completo do G-Eval.

    Tenta a ponderação por logprobs; se a API não devolver, cai na amostragem;
    se nem isso funcionar, usa a nota direta (e marca o método como "direto",
    para que o relatório mostre que aquele número é menos confiável).
    """
    settings = settings or SETTINGS

    if isinstance(criterio, str):
        if criterio not in CRITERIOS:
            raise ValueError(
                f"Critério '{criterio}' desconhecido. "
                f"Disponíveis: {', '.join(CRITERIOS)}"
            )
        criterio = CRITERIOS[criterio]

    criterio = Criterio(
        nome=criterio.nome,
        definicao=criterio.definicao,
        escala_min=criterio.escala_min,
        escala_max=settings.geval_escala_max,
        passos=criterio.passos,
    )

    resultado = ResultadoGEval(criterio=criterio.nome)

    # Passo 1-2: auto-CoT
    try:
        resultado.passos = passos or criterio.passos or gerar_passos(criterio, settings)
    except Exception as erro:  # noqa: BLE001
        resultado.erro = f"falha ao gerar passos: {erro}"
        resultado.passos = "1. Leia o contexto e a resposta. 2. Atribua a nota."

    prompt = PROMPT_AVALIACAO.format(
        nome=criterio.nome,
        definicao=criterio.definicao,
        minimo=criterio.escala_min,
        maximo=criterio.escala_max,
        passos=resultado.passos,
        contexto=contexto[:6000],
        pergunta=pergunta,
        resposta=resposta,
    )

    # Passo 3: pontuação
    if settings.geval_usar_logprobs:
        try:
            client = get_client(settings)
            resposta_api = client.chat.completions.create(
                model=settings.juiz_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=8,
                logprobs=True,
                top_logprobs=10,
            )
            bruto = _extrair_nota(
                resposta_api.choices[0].message.content or "",
                criterio.escala_min,
                criterio.escala_max,
            )
            ponderado = _score_por_logprobs(resposta_api, criterio)

            if ponderado:
                resultado.score, resultado.distribuicao = ponderado
                resultado.score_bruto = bruto or round(resultado.score)
                resultado.metodo = "logprobs"
                return resultado

            if bruto is not None:
                resultado.score = float(bruto)
                resultado.score_bruto = bruto
                resultado.metodo = "direto"
                return resultado
        except Exception as erro:  # noqa: BLE001
            resultado.erro = f"logprobs indisponível ({erro}); usando amostragem"

    # Fallback: amostragem, como o artigo faz com o GPT-4
    try:
        score, distribuicao, moda = _score_por_amostragem(prompt, criterio, settings)
        resultado.score = score
        resultado.distribuicao = distribuicao
        resultado.score_bruto = moda
        resultado.metodo = "amostragem"
    except Exception as erro:  # noqa: BLE001
        resultado.erro = f"amostragem falhou: {erro}"

    return resultado


def avaliar(
    pergunta: str,
    resposta: str,
    contexto: str,
    criterios: list[str] | None = None,
    settings: Settings | None = None,
) -> dict[str, ResultadoGEval]:
    """Avalia várias dimensões de uma vez."""
    settings = settings or SETTINGS
    criterios = criterios or ["fundamentacao", "relevancia"]
    return {
        nome: avaliar_criterio(pergunta, resposta, contexto, nome, settings)
        for nome in criterios
    }
