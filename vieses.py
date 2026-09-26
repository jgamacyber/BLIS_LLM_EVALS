"""
Vieses do juiz — testar o avaliador antes de confiar nele.

Baseado em Thakur et al. (2025), *Judging the Judges: Evaluating Alignment and
Vulnerabilities in LLMs-as-Judges*, que documenta que juízes LLM são sensíveis
ao comprimento e à qualidade do prompt, exibem **leniência**, e falham mesmo
quando pedidos para avaliar uma correspondência literal com a referência.

A ideia central deste módulo: **um juiz é um instrumento de medida, e
instrumentos precisam ser calibrados antes de produzir números**. Rodar essas
sondagens custa algumas chamadas e evita publicar uma avaliação inteira
construída sobre um juiz quebrado.

Quatro vieses testados:

1. **Posição** — na comparação par a par, a ordem em que as respostas aparecem
   muda o vencedor? Um juiz sem viés escolhe a mesma resposta nas duas ordens.
2. **Verbosidade** — respostas mais longas ganham nota maior só por serem
   longas?
3. **Leniência** — o juiz distribui notas altas sistematicamente?
4. **Autopreferência** — o juiz prefere texto gerado por ele mesmo? (Alerta
   levantado pelo G-Eval; aqui é uma verificação estrutural.)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from concordancia import pearson
from config import SETTINGS, Settings, get_client

PROMPT_PARWISE = """Você vai comparar duas respostas para a mesma pergunta e \
decidir qual é melhor.

PERGUNTA:
{pergunta}

CONTEXTO:
{contexto}

RESPOSTA A:
{resposta_a}

RESPOSTA B:
{resposta_b}

Qual resposta é melhor? Considere correção, fundamentação no contexto e \
relevância para a pergunta.

Responda APENAS com uma letra: A ou B."""


# --------------------------------------------------------------------------- #
# 1. Viés de posição
# --------------------------------------------------------------------------- #


@dataclass
class ResultadoPosicao:
    """Resultado do teste de viés de posição."""

    n_pares: int = 0
    consistentes: int = 0
    preferiu_primeira: int = 0
    preferiu_segunda: int = 0
    inconclusivos: int = 0
    detalhes: list[dict] = field(default_factory=list)

    @property
    def taxa_consistencia(self) -> float:
        """Fração de pares em que o juiz manteve a escolha ao inverter a ordem."""
        return self.consistentes / self.n_pares if self.n_pares else 0.0

    @property
    def taxa_inversao(self) -> float:
        return 1.0 - self.taxa_consistencia

    @property
    def vies_posicional(self) -> float:
        """
        Assimetria da preferência nas comparações inconsistentes.

        0 = sem viés; +1 = sempre prefere a primeira; -1 = sempre a segunda.
        """
        total = self.preferiu_primeira + self.preferiu_segunda
        if total == 0:
            return 0.0
        return (self.preferiu_primeira - self.preferiu_segunda) / total

    def imprimir(self) -> None:
        print(f"\n  Viés de posição  ({self.n_pares} pares, cada um nas 2 ordens)")
        print("  " + "-" * 60)
        print(f"    consistência        {self.taxa_consistencia:>7.1%}   "
              f"({self.consistentes}/{self.n_pares} mantiveram a escolha)")
        print(f"    taxa de inversão    {self.taxa_inversao:>7.1%}")
        print(f"    viés posicional     {self.vies_posicional:>+7.2f}   ", end="")

        if abs(self.vies_posicional) < 0.2:
            print("(sem viés relevante)")
        elif self.vies_posicional > 0:
            print("(tende a preferir a PRIMEIRA posição)")
        else:
            print("(tende a preferir a SEGUNDA posição)")

        if self.taxa_consistencia < 0.7:
            print("    [!] consistência baixa: este juiz não é confiável para")
            print("        comparação par a par. Use avaliação pontual ou troque")
            print("        de modelo.")


def _comparar_par(
    pergunta: str,
    contexto: str,
    resposta_a: str,
    resposta_b: str,
    settings: Settings,
) -> str | None:
    """Devolve 'A', 'B' ou None se a resposta for ilegível."""
    client = get_client(settings)
    resposta = client.chat.completions.create(
        model=settings.juiz_model,
        messages=[
            {
                "role": "user",
                "content": PROMPT_PARWISE.format(
                    pergunta=pergunta,
                    contexto=contexto[:3000],
                    resposta_a=resposta_a,
                    resposta_b=resposta_b,
                ),
            }
        ],
        temperature=0.0,
        max_tokens=5,
    )
    texto = (resposta.choices[0].message.content or "").strip().upper()
    match = re.search(r"\b([AB])\b", texto)
    return match.group(1) if match else None


def testar_vies_posicao(
    pares: list[dict], settings: Settings | None = None, verboso: bool = True
) -> ResultadoPosicao:
    """
    Apresenta cada par nas duas ordens e mede se o juiz mantém a escolha.

    Cada par é um dict com `pergunta`, `contexto`, `resposta_1` e `resposta_2`.
    Custa 2 chamadas por par.

    Um juiz sem viés de posição escolhe a mesma RESPOSTA nas duas ordens —
    ou seja, se escolheu A na primeira rodada (onde A = resposta_1), deve
    escolher B na segunda (onde B = resposta_1).
    """
    settings = settings or SETTINGS
    resultado = ResultadoPosicao(n_pares=len(pares))

    for i, par in enumerate(pares, start=1):
        pergunta = par["pergunta"]
        contexto = par.get("contexto", "")
        r1, r2 = par["resposta_1"], par["resposta_2"]

        try:
            # Ordem original: A = r1, B = r2
            escolha_1 = _comparar_par(pergunta, contexto, r1, r2, settings)
            # Ordem invertida: A = r2, B = r1
            escolha_2 = _comparar_par(pergunta, contexto, r2, r1, settings)
        except Exception as erro:  # noqa: BLE001
            print(f"  [viés posição] falha no par {i}: {erro}")
            resultado.inconclusivos += 1
            continue

        if escolha_1 is None or escolha_2 is None:
            resultado.inconclusivos += 1
            continue

        # Qual resposta real foi escolhida em cada rodada?
        vencedor_1 = "r1" if escolha_1 == "A" else "r2"
        vencedor_2 = "r2" if escolha_2 == "A" else "r1"

        consistente = vencedor_1 == vencedor_2
        if consistente:
            resultado.consistentes += 1
        else:
            # Inconsistente: o juiz seguiu a posição, não o conteúdo.
            if escolha_1 == "A" and escolha_2 == "A":
                resultado.preferiu_primeira += 1
            elif escolha_1 == "B" and escolha_2 == "B":
                resultado.preferiu_segunda += 1

        resultado.detalhes.append({
            "par": i,
            "ordem_original": vencedor_1,
            "ordem_invertida": vencedor_2,
            "consistente": consistente,
        })

        if verboso:
            marca = "OK " if consistente else "INVERTEU"
            print(f"    par {i}: [{marca:^8}] {vencedor_1} / {vencedor_2}")

    return resultado


# --------------------------------------------------------------------------- #
# 2. Viés de verbosidade
# --------------------------------------------------------------------------- #


@dataclass
class ResultadoVerbosidade:
    """Correlação entre comprimento da resposta e nota atribuída."""

    correlacao_juiz: float = 0.0
    correlacao_humana: float = 0.0
    n: int = 0

    @property
    def vies_excedente(self) -> float:
        """
        Quanto do efeito do comprimento é viés do juiz.

        Respostas mais longas podem ser genuinamente melhores — por isso
        subtraímos a correlação observada nas notas HUMANAS. O que sobra é o
        efeito que o juiz atribui ao comprimento e o humano não.
        """
        return self.correlacao_juiz - self.correlacao_humana

    def imprimir(self) -> None:
        print(f"\n  Viés de verbosidade  (n={self.n})")
        print("  " + "-" * 60)
        print(f"    corr(comprimento, nota do juiz)    {self.correlacao_juiz:>+7.3f}")
        print(f"    corr(comprimento, nota humana)     {self.correlacao_humana:>+7.3f}")
        print(f"    viés excedente                     {self.vies_excedente:>+7.3f}   ",
              end="")

        if abs(self.vies_excedente) < 0.15:
            print("(desprezível)")
        elif self.vies_excedente > 0:
            print("(premia respostas longas)")
        else:
            print("(penaliza respostas longas)")


def testar_vies_verbosidade(
    respostas: list[str],
    notas_juiz: list[float],
    notas_humanas: list[float] | None = None,
) -> ResultadoVerbosidade:
    """
    Mede a correlação entre comprimento e nota. Custo zero — só aritmética.

    Sem notas humanas, devolve só a correlação bruta, que é ambígua: pode ser
    viés ou pode ser que respostas longas sejam mesmo melhores neste conjunto.
    """
    comprimentos = [float(len(r.split())) for r in respostas]

    corr_humana = 0.0
    if notas_humanas and len(notas_humanas) == len(comprimentos):
        corr_humana = pearson(comprimentos, notas_humanas)

    return ResultadoVerbosidade(
        correlacao_juiz=pearson(comprimentos, notas_juiz),
        correlacao_humana=corr_humana,
        n=len(respostas),
    )


# --------------------------------------------------------------------------- #
# 3. Leniência
# --------------------------------------------------------------------------- #


@dataclass
class ResultadoLeniencia:
    """Distribuição das notas do juiz contra a referência humana."""

    media_juiz: float = 0.0
    media_humana: float = 0.0
    desvio: float = 0.0
    concentracao: float = 0.0
    nota_dominante: float = 0.0
    n: int = 0

    @property
    def tendencia(self) -> str:
        if self.desvio > 0.1:
            return "leniente"
        if self.desvio < -0.1:
            return "severo"
        return "calibrado"

    def imprimir(self) -> None:
        print(f"\n  Leniência  (n={self.n})")
        print("  " + "-" * 60)
        print(f"    média do juiz       {self.media_juiz:>7.3f}")
        print(f"    média humana        {self.media_humana:>7.3f}")
        print(f"    desvio              {self.desvio:>+7.3f}   ({self.tendencia})")
        print(f"    concentração        {self.concentracao:>7.1%}   "
              f"(na nota {self.nota_dominante:g})")

        if self.concentracao > 0.6:
            print("    [!] o juiz concentra as notas em um único valor. É o")
            print("        problema que o G-Eval resolve com a ponderação por")
            print("        probabilidade — verifique se os logprobs estão ativos.")


def testar_leniencia(
    notas_juiz: list[float], notas_humanas: list[float]
) -> ResultadoLeniencia:
    """
    Compara a distribuição das notas do juiz com a humana. Custo zero.

    Thakur et al. observam que juízes "exibem leniência, reduzindo a
    consistência da avaliação". A concentração também importa: um juiz que dá
    3 para tudo tem correlação zero por construção.
    """
    n = len(notas_juiz)
    if n == 0:
        return ResultadoLeniencia()

    media_juiz = sum(notas_juiz) / n
    media_humana = sum(notas_humanas) / len(notas_humanas) if notas_humanas else 0.0

    # Concentração: fração da nota mais frequente (arredondada).
    contagem: dict[float, int] = {}
    for nota in notas_juiz:
        chave = round(nota * 2) / 2  # agrupa em passos de 0,5
        contagem[chave] = contagem.get(chave, 0) + 1
    dominante, frequencia = max(contagem.items(), key=lambda kv: kv[1])

    return ResultadoLeniencia(
        media_juiz=media_juiz,
        media_humana=media_humana,
        desvio=media_juiz - media_humana,
        concentracao=frequencia / n,
        nota_dominante=dominante,
        n=n,
    )


# --------------------------------------------------------------------------- #
# 4. Autopreferência (verificação estrutural)
# --------------------------------------------------------------------------- #


@dataclass
class AvisoAutopreferencia:
    risco: bool = False
    mensagem: str = ""

    def imprimir(self) -> None:
        if self.risco:
            print(f"\n  [!] Autopreferência: {self.mensagem}")
        else:
            print(f"\n  Autopreferência: {self.mensagem}")


def verificar_autopreferencia(settings: Settings | None = None) -> AvisoAutopreferencia:
    """
    Verifica se o juiz é o mesmo modelo avaliado.

    O G-Eval alerta: "métricas baseadas em LLM têm um problema potencial de
    preferir textos gerados por LLM, o que pode levar ao auto-reforço dos LLMs
    se essas métricas forem usadas como sinal de recompensa".

    Não é um teste empírico (exigiria vários modelos), mas é a verificação que
    impede o erro mais comum: usar `gpt-4o-mini` para gerar e para julgar, e
    tratar o resultado como avaliação independente.
    """
    settings = settings or SETTINGS

    if settings.juiz_e_avaliado_iguais:
        return AvisoAutopreferencia(
            risco=True,
            mensagem=(
                f"juiz e sistema avaliado usam o mesmo modelo "
                f"({settings.juiz_model}). Os resultados podem estar inflados "
                f"por autopreferência. Defina JUIZ_MODEL diferente de "
                f"MODELO_AVALIADO no .env."
            ),
        )
    return AvisoAutopreferencia(
        risco=False,
        mensagem=(
            f"juiz ({settings.juiz_model}) difere do avaliado "
            f"({settings.modelo_avaliado})."
        ),
    )
