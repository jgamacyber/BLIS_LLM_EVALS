"""
Testes com LLM simulado — validam os caminhos que chamam a API, sem gastar nada.

Cobrem o RAGAS (extração de afirmações, verificação, perguntas reversas,
extração de frases), o G-Eval (auto-CoT, ponderação por logprobs e por
amostragem), o teste de viés de posição e o pipeline completo.

A parte mais valiosa aqui é verificar a **ponderação por probabilidade** do
G-Eval contra um valor calculado à mão: é a peça do artigo mais fácil de
implementar errado em silêncio.

    python testes_mock.py
"""

from __future__ import annotations

import math
import sys
from types import SimpleNamespace

import config
import g_eval as mod_geval
import pipeline as mod_pipeline
import ragas as mod_ragas
import vieses as mod_vieses
from dataset import Caso

FALHAS: list[str] = []
CHAMADAS: list[dict] = []


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
# Dublê do cliente
# --------------------------------------------------------------------------- #


class ChatFalso:
    def __init__(self, respostas: list[str], logprobs: dict | None = None) -> None:
        self.respostas = respostas
        self.logprobs = logprobs
        self.i = 0

    def create(self, **kwargs):
        CHAMADAS.append(kwargs)
        texto = self.respostas[min(self.i, len(self.respostas) - 1)]
        self.i += 1

        escolha = SimpleNamespace(message=SimpleNamespace(content=texto))

        if self.logprobs is not None and kwargs.get("logprobs"):
            alternativas = [
                SimpleNamespace(token=str(nota), logprob=math.log(max(p, 1e-12)))
                for nota, p in self.logprobs.items()
            ]
            escolha.logprobs = SimpleNamespace(
                content=[SimpleNamespace(token=texto.strip(),
                                         top_logprobs=alternativas)]
            )
        else:
            escolha.logprobs = None

        return SimpleNamespace(
            choices=[escolha],
            usage=SimpleNamespace(prompt_tokens=200, completion_tokens=40),
        )


class EmbeddingsFalsos:
    """Embeddings determinísticos: textos iguais dão vetores iguais."""

    def create(self, **kwargs):
        CHAMADAS.append({"tipo": "embeddings", **kwargs})
        entradas = kwargs["input"]
        if isinstance(entradas, str):
            entradas = [entradas]
        dados = [
            SimpleNamespace(
                index=i,
                embedding=[float(len(t) % 5), float(sum(map(ord, t[:6])) % 7), 1.0],
            )
            for i, t in enumerate(entradas)
        ]
        return SimpleNamespace(data=dados)


class ClienteFalso:
    def __init__(self, respostas: list[str], logprobs: dict | None = None) -> None:
        self.chat = SimpleNamespace(completions=ChatFalso(respostas, logprobs))
        self.embeddings = EmbeddingsFalsos()


def instalar(respostas: list[str], logprobs: dict | None = None) -> ClienteFalso:
    CHAMADAS.clear()
    cliente = ClienteFalso(respostas, logprobs)
    falso = lambda settings=None: cliente  # noqa: E731
    for modulo in (config, mod_ragas, mod_geval, mod_vieses, mod_pipeline):
        if hasattr(modulo, "get_client"):
            modulo.get_client = falso
    return cliente


class ClienteQuebrado:
    def __init__(self) -> None:
        erro = lambda **kw: (_ for _ in ()).throw(RuntimeError("falha de rede"))  # noqa: E731
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=erro))
        self.embeddings = SimpleNamespace(create=erro)


class SettingsFalsas:
    api_key = "sk-teste"
    juiz_model = "modelo/juiz"
    modelo_avaliado = "modelo/avaliado"
    embedding_model = "modelo/embed"
    temperature = 0.0
    max_tokens = 800
    geval_usar_logprobs = True
    geval_amostras = 4
    geval_escala_max = 5
    ragas_n_perguntas = 2
    ppi_alpha = 0.05
    tem_chave = True
    juiz_e_avaliado_iguais = False


S = SettingsFalsas()

CONTEXTO = (
    "O DPR usa um bi-encoder com dimensão 768. Em acurácia top-5, o DPR atinge "
    "65,2% contra 42,9% do BM25. O treinamento usa negativos no lote."
)


# --------------------------------------------------------------------------- #


def testar_ragas_faithfulness() -> None:
    secao("1. RAGAS — faithfulness")

    instalar([
        "O DPR usa um bi-encoder.\nA dimensão é 768.\nO DPR custou 1 milhão de dólares.",
        "Explicação 1...\nExplicação 2...\nExplicação 3...\n"
        "VEREDITO 1: Sim\nVEREDITO 2: Sim\nVEREDITO 3: Não",
    ])

    f, afirmacoes, vereditos = faithful = mod_ragas.faithfulness(
        "Como funciona o DPR?", "resposta qualquer", CONTEXTO, S
    )

    checar(len(afirmacoes) == 3, "extrai 3 afirmações da resposta")
    checar(vereditos == [True, True, False], "lê os vereditos na ordem certa")
    checar(perto(f, 2 / 3), f"F = |V|/|S| = 2/3 (obteve {f:.3f})")
    checar(len(CHAMADAS) == 2, "usa 2 chamadas: extrair + verificar em lote")

    # Vereditos ausentes são conservadoramente 'não sustentado'.
    instalar([
        "Afirmação um.\nAfirmação dois.",
        "VEREDITO 1: Sim",  # o segundo veredito não veio
    ])
    f2, _, v2 = mod_ragas.faithfulness("p", "r", CONTEXTO, S)
    checar(v2 == [True, False],
           "veredito ausente é tratado como NÃO sustentado (conservador)")
    checar(perto(f2, 0.5), "F conservador = 0.5")

    instalar(["", "VEREDITO 1: Sim"])
    f3, a3, _ = mod_ragas.faithfulness("p", "r", CONTEXTO, S)
    checar(f3 == 0.0 and a3 == [], "resposta sem afirmações devolve F=0 sem quebrar")

    instalar(["Uma afirmação.", "VEREDITO 1: Sim"])
    f4, _, _ = mod_ragas.faithfulness("p", "r", CONTEXTO, S)
    checar(perto(f4, 1.0), "tudo sustentado dá F=1.0")


def testar_ragas_answer_relevance() -> None:
    secao("2. RAGAS — answer relevance")

    instalar(["Como funciona o DPR?", "O que é o DPR?"])
    ar, geradas = mod_ragas.answer_relevance(
        "Como funciona o DPR?", "O DPR usa um bi-encoder.", S, n=2
    )

    checar(len(geradas) == 2, "gera n perguntas reversas")
    checar(-1.0 <= ar <= 1.0, f"AR está no intervalo do cosseno ({ar:.3f})")

    chamadas_chat = [c for c in CHAMADAS if "messages" in c]
    chamadas_emb = [c for c in CHAMADAS if c.get("tipo") == "embeddings"]
    checar(len(chamadas_chat) == 2, "uma chamada de chat por pergunta gerada")
    checar(len(chamadas_emb) == 1, "uma única chamada de embeddings em lote")
    checar(len(chamadas_emb[0]["input"]) == 3,
           "embedda a pergunta original + as 2 geradas")

    # Pergunta reconstruída idêntica à original -> similaridade máxima.
    instalar(["Qual é a capital?"])
    ar_perfeito, _ = mod_ragas.answer_relevance("Qual é a capital?", "Canberra.", S, n=1)
    checar(perto(ar_perfeito, 1.0, 1e-6),
           "pergunta reconstruída idêntica dá AR = 1.0")

    instalar([""])
    ar_vazio, g = mod_ragas.answer_relevance("p", "r", S, n=1)
    checar(ar_vazio == 0.0 and g == [], "nenhuma pergunta gerada devolve AR=0")


def testar_ragas_context_relevance() -> None:
    secao("3. RAGAS — context relevance")

    contexto = ("Frase relevante sobre o DPR e seus resultados. "
                "Frase irrelevante sobre outra coisa qualquer. "
                "Terceira frase também irrelevante aqui.")

    instalar(["Frase relevante sobre o DPR e seus resultados."])
    cr, extraidas, total, insuf = mod_ragas.context_relevance("Como o DPR vai?",
                                                              contexto, S)
    checar(total == 3, f"conta 3 frases no contexto (obteve {total})")
    checar(extraidas == 1, "extraiu 1 frase relevante")
    checar(perto(cr, 1 / 3), f"CR = 1/3 (obteve {cr:.3f})")
    checar(not insuf, "não marcou como insuficiente")

    instalar(["Informação Insuficiente"])
    cr2, _, _, insuf2 = mod_ragas.context_relevance("p", contexto, S)
    checar(cr2 == 0.0 and insuf2,
           "'Informação Insuficiente' zera o CR e marca a flag")

    cr3, _, total3, insuf3 = mod_ragas.context_relevance("p", "", S)
    checar(total3 == 0 and insuf3, "contexto vazio é insuficiente")

    # O LLM às vezes devolve mais frases do que existem; o CR não pode passar de 1.
    instalar(["Frase A longa o suficiente para contar.\n"
              "Frase B longa o suficiente para contar.\n"
              "Frase C longa o suficiente para contar.\n"
              "Frase D longa o suficiente para contar."])
    cr4, _, _, _ = mod_ragas.context_relevance("p", contexto, S)
    checar(cr4 <= 1.0, f"CR nunca passa de 1.0 (obteve {cr4:.3f})")


def testar_geval_ponderacao() -> None:
    secao("4. G-Eval — ponderação por probabilidade")

    # Distribuição escolhida para conferência à mão:
    #   score = 1(0.1) + 2(0.1) + 3(0.4) + 4(0.3) + 5(0.1)
    #         = 0.1 + 0.2 + 1.2 + 1.2 + 0.5 = 3.2
    distribuicao = {1: 0.1, 2: 0.1, 3: 0.4, 4: 0.3, 5: 0.1}
    instalar(["1. Leia o contexto.\n2. Compare.\n3. Atribua a nota.", "3"],
             logprobs=distribuicao)

    mod_geval.limpar_cache_passos()
    r = mod_geval.avaliar_criterio("Pergunta?", "Resposta.", CONTEXTO,
                                   "fundamentacao", S)

    checar(r.metodo == "logprobs", "usou a ponderação por logprobs")
    checar(perto(r.score, 3.2, 1e-6), f"score ponderado = 3.2 (obteve {r.score:.4f})")
    checar(r.score_bruto == 3, "registra também a nota bruta (3)")
    checar(r.score != r.score_bruto,
           "a nota ponderada difere da bruta — é o ponto do método")
    checar(len(r.distribuicao) == 5, "guarda a distribuição completa")
    checar(perto(sum(r.distribuicao.values()), 1.0), "a distribuição soma 1")
    checar(perto(r.normalizado, (3.2 - 1) / 4), "normalização para [0,1] correta")

    # Tokens fora da escala devem ser ignorados na normalização.
    instalar(["passos", "4"], logprobs={4: 0.5, 9: 0.3, 2: 0.2})
    mod_geval.limpar_cache_passos()
    r2 = mod_geval.avaliar_criterio("p", "r", CONTEXTO, "relevancia", S)
    esperado = (4 * 0.5 + 2 * 0.2) / 0.7  # 9 está fora da escala 1-5
    checar(perto(r2.score, esperado, 1e-6),
           f"ignora tokens fora da escala (esperado {esperado:.3f})")


def testar_geval_amostragem() -> None:
    secao("5. G-Eval — fallback por amostragem")

    class SemLogprobs(SettingsFalsas):
        geval_usar_logprobs = False
        geval_amostras = 4

    # 4 amostras: 3, 4, 4, 5 -> média = 4.0
    instalar(["1. Passo um.", "3", "4", "4", "5"])
    mod_geval.limpar_cache_passos()
    r = mod_geval.avaliar_criterio("p", "r", CONTEXTO, "fundamentacao",
                                   SemLogprobs())

    checar(r.metodo == "amostragem", "caiu no fallback de amostragem")
    checar(perto(r.score, 4.0), f"média das amostras = 4.0 (obteve {r.score:.3f})")
    checar(r.score_bruto == 4, "a moda é 4")
    checar(perto(r.distribuicao.get(4, 0), 0.5), "a distribuição empírica está certa")

    chamadas_avaliacao = [c for c in CHAMADAS if c.get("max_tokens") == 8]
    checar(len(chamadas_avaliacao) == 4, "fez exatamente 4 amostras")
    checar(all(c["temperature"] == 1.0 for c in chamadas_avaliacao),
           "amostra com temperatura 1.0, como o artigo")


def testar_geval_cache_e_falhas() -> None:
    secao("6. G-Eval — cache dos passos e degradação")

    mod_geval.limpar_cache_passos()
    instalar(["1. Passos gerados.", "4", "4"], logprobs={4: 1.0})

    mod_geval.avaliar_criterio("p1", "r1", CONTEXTO, "fundamentacao", S)
    chamadas_primeira = len(CHAMADAS)
    mod_geval.avaliar_criterio("p2", "r2", CONTEXTO, "fundamentacao", S)
    chamadas_segunda = len(CHAMADAS) - chamadas_primeira

    checar(chamadas_primeira == 2, "primeira avaliação: gera passos + avalia")
    checar(chamadas_segunda == 1,
           "segunda avaliação reusa os passos do cache (1 chamada)")

    mod_geval.limpar_cache_passos()
    mod_geval.get_client = lambda settings=None: ClienteQuebrado()
    r = mod_geval.avaliar_criterio("p", "r", CONTEXTO, "fundamentacao", S)
    checar(bool(r.erro), "falha de rede é registrada em vez de propagar")
    checar(r.score == 0.0, "score fica 0 quando tudo falha")

    try:
        mod_geval.avaliar_criterio("p", "r", CONTEXTO, "inexistente", S)
        checar(False, "critério desconhecido levanta erro")
    except ValueError:
        checar(True, "critério desconhecido levanta erro")


def testar_vies_posicao() -> None:
    secao("7. Viés de posição")

    par = {"pergunta": "p", "contexto": CONTEXTO,
           "resposta_1": "boa", "resposta_2": "ruim"}

    # Juiz consistente: escolhe a resposta_1 nas duas ordens.
    # Ordem original (A=r1) -> "A"; ordem invertida (A=r2, B=r1) -> "B".
    instalar(["A", "B"])
    r = mod_vieses.testar_vies_posicao([par], S, verboso=False)
    checar(r.consistentes == 1, "detecta juiz consistente")
    checar(perto(r.taxa_consistencia, 1.0), "taxa de consistência 100%")
    checar(perto(r.vies_posicional, 0.0), "sem viés posicional")
    checar(len(CHAMADAS) == 2, "2 chamadas por par (as duas ordens)")

    # Juiz que sempre escolhe a primeira posição, seja qual for o conteúdo.
    instalar(["A", "A"])
    r2 = mod_vieses.testar_vies_posicao([par], S, verboso=False)
    checar(r2.consistentes == 0, "detecta inconsistência")
    checar(r2.preferiu_primeira == 1, "identifica preferência pela primeira posição")
    checar(perto(r2.vies_posicional, 1.0), "viés posicional máximo (+1)")

    instalar(["B", "B"])
    r3 = mod_vieses.testar_vies_posicao([par], S, verboso=False)
    checar(perto(r3.vies_posicional, -1.0), "viés pela segunda posição (-1)")

    instalar(["resposta ilegível", "também ilegível"])
    r4 = mod_vieses.testar_vies_posicao([par], S, verboso=False)
    checar(r4.inconclusivos == 1, "resposta ilegível conta como inconclusiva")


def testar_pipeline() -> None:
    secao("8. Pipeline completo")

    casos = [
        Caso(id="t1", pergunta="Qual a acurácia top-5 do DPR?", contexto=CONTEXTO,
             resposta="É 65,2%.", gabarito="65,2",
             anotacao_humana={"fundamentacao": 1, "relevancia": 1}),
        Caso(id="t2", pergunta="Qual a dimensão?", contexto=CONTEXTO,
             resposta="É 768 e custou 1 milhão.", gabarito="768",
             anotacao_humana={"fundamentacao": 0, "relevancia": 1}),
    ]

    # Só lexical: nenhuma chamada deve acontecer.
    instalar([])
    avaliacao = mod_pipeline.executar(casos, ["lexical"], verboso=False)
    checar(len(avaliacao.resultados) == 2, "avalia todos os casos")
    checar(len(CHAMADAS) == 0, "métricas lexicais não chamam a API")
    checar(avaliacao.resultados[0].contains == 1.0, "lexical calculada")
    checar(avaliacao.resultados[0].anotacao_humana["fundamentacao"] == 1,
           "preserva a anotação de referência")

    # Com RAGAS
    instalar(["Afirmação.", "VEREDITO 1: Sim", "Pergunta reversa?",
              "Pergunta reversa?", "Frase relevante extraída do contexto."])
    avaliacao2 = mod_pipeline.executar(casos[:1], ["lexical", "ragas"], S,
                                       verboso=False)
    r = avaliacao2.resultados[0]
    checar(r.faithfulness > 0, "faithfulness calculada")
    checar(r.n_afirmacoes == 1, "registra o número de afirmações")
    checar(r.latencia > 0, "registra a latência")
    checar("ragas" in avaliacao2.metricas_usadas, "registra as métricas usadas")

    # Falha não derruba o pipeline
    for modulo in (mod_ragas, mod_geval):
        modulo.get_client = lambda settings=None: ClienteQuebrado()
    avaliacao3 = mod_pipeline.executar(casos[:1], ["lexical", "ragas"], S,
                                       verboso=False)
    checar(bool(avaliacao3.resultados[0].erros),
           "erro de rede é registrado no resultado")
    checar(avaliacao3.resultados[0].contains == 1.0,
           "as métricas lexicais sobrevivem à falha do RAGAS")

    # Persistência
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        caminho = Path(tmp) / "av.json"
        avaliacao2.salvar(caminho)
        recarregada = mod_pipeline.Avaliacao.carregar(caminho)
        checar(len(recarregada.resultados) == 1, "salva e recarrega")
        checar(perto(recarregada.resultados[0].faithfulness, r.faithfulness),
               "preserva os valores")

    # Estimativa de custo não chama nada
    CHAMADAS.clear()
    custo = mod_pipeline.estimar_custo(10, ["ragas", "geval"],
                                       ["fundamentacao"], S)
    checar(custo["chamadas_chat"] > 0, "estima chamadas de chat")
    checar(custo["chamadas_embeddings"] == 10, "estima 1 embedding por caso")
    checar(len(CHAMADAS) == 0, "a estimativa não faz nenhuma chamada")


def main() -> int:
    print("=" * 64)
    print("  TESTES COM LLM SIMULADO — sem API, sem custo")
    print("=" * 64)

    testar_ragas_faithfulness()
    testar_ragas_answer_relevance()
    testar_ragas_context_relevance()
    testar_geval_ponderacao()
    testar_geval_amostragem()
    testar_geval_cache_e_falhas()
    testar_vies_posicao()
    testar_pipeline()

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
