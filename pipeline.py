"""
Pipeline de avaliação: roda as métricas sobre todos os casos e guarda o rastro.

A ordem importa e é deliberada:

    1. Métricas lexicais   — grátis, sempre. São o baseline contra o qual o
                             juiz caro precisa se justificar.
    2. RAGAS               — decompõe a qualidade por componente (recuperador
                             vs. gerador).
    3. G-Eval              — nota contínua ponderada por probabilidade.

Os resultados vão para JSON, de modo que a análise (`analise.py`) possa ser
refeita quantas vezes for preciso sem repetir as chamadas pagas.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import g_eval
import ragas as mod_ragas
from config import SETTINGS, Settings
from dataset import Caso
from metricas_lexicais import avaliar_lexical

CAMINHO_RESULTADOS = Path("resultados/avaliacao.json")


@dataclass
class ResultadoCaso:
    """Todas as métricas calculadas para um caso."""

    caso_id: str
    pergunta: str = ""
    resposta: str = ""

    # lexicais (grátis)
    exact_match: float = 0.0
    contains: float = 0.0
    f1_tokens: float = 0.0
    cobertura: float = 0.0
    abstencao: bool = False

    # RAGAS
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    context_relevance: float = 0.0
    ragas_media: float = 0.0
    n_afirmacoes: int = 0
    n_sustentadas: int = 0

    # G-Eval (normalizado para [0,1])
    geval: dict = field(default_factory=dict)
    geval_bruto: dict = field(default_factory=dict)
    geval_metodo: str = ""

    # referência
    anotacao_humana: dict = field(default_factory=dict)

    # custo
    latencia: float = 0.0
    erros: list[str] = field(default_factory=list)

    def metrica(self, nome: str) -> float:
        """Acesso uniforme a qualquer métrica pelo nome."""
        if nome in self.geval:
            return self.geval[nome]
        return float(getattr(self, nome, 0.0))


@dataclass
class Avaliacao:
    """O experimento inteiro, serializável."""

    resultados: list[ResultadoCaso] = field(default_factory=list)
    juiz_model: str = ""
    metricas_usadas: list[str] = field(default_factory=list)
    criterios_geval: list[str] = field(default_factory=list)
    timestamp: str = ""

    def salvar(self, caminho: str | Path = CAMINHO_RESULTADOS) -> Path:
        caminho = Path(caminho)
        caminho.parent.mkdir(parents=True, exist_ok=True)
        caminho.write_text(
            json.dumps(
                {
                    "juiz_model": self.juiz_model,
                    "metricas_usadas": self.metricas_usadas,
                    "criterios_geval": self.criterios_geval,
                    "timestamp": self.timestamp,
                    "resultados": [asdict(r) for r in self.resultados],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return caminho

    @classmethod
    def carregar(cls, caminho: str | Path = CAMINHO_RESULTADOS) -> "Avaliacao":
        caminho = Path(caminho)
        if not caminho.exists():
            raise FileNotFoundError(
                f"Resultados não encontrados em {caminho}. "
                f"Rode primeiro: python main.py avaliar"
            )
        dados = json.loads(caminho.read_text(encoding="utf-8"))
        return cls(
            resultados=[ResultadoCaso(**r) for r in dados["resultados"]],
            juiz_model=dados.get("juiz_model", ""),
            metricas_usadas=dados.get("metricas_usadas", []),
            criterios_geval=dados.get("criterios_geval", []),
            timestamp=dados.get("timestamp", ""),
        )

    def nomes_metricas(self) -> list[str]:
        """Todas as métricas presentes, lexicais + RAGAS + G-Eval."""
        base = [
            "exact_match", "contains", "f1_tokens", "cobertura",
            "faithfulness", "answer_relevance", "context_relevance", "ragas_media",
        ]
        geval = []
        for r in self.resultados:
            for nome in r.geval:
                if nome not in geval:
                    geval.append(nome)
        return base + geval

    def serie(self, metrica: str) -> list[float]:
        return [r.metrica(metrica) for r in self.resultados]

    def serie_humana(self, dimensao: str) -> list[float]:
        return [float(r.anotacao_humana.get(dimensao, 0)) for r in self.resultados]

    def com_anotacao(self, dimensao: str) -> list[ResultadoCaso]:
        return [r for r in self.resultados if dimensao in r.anotacao_humana]


# --------------------------------------------------------------------------- #
# Execução
# --------------------------------------------------------------------------- #


def avaliar_caso(
    caso: Caso,
    metricas: list[str],
    criterios_geval: list[str],
    settings: Settings | None = None,
) -> ResultadoCaso:
    """Aplica as métricas pedidas a um caso."""
    settings = settings or SETTINGS
    inicio = time.perf_counter()

    resultado = ResultadoCaso(
        caso_id=caso.id,
        pergunta=caso.pergunta,
        resposta=caso.resposta,
        anotacao_humana=dict(caso.anotacao_humana),
    )

    # 1. Lexicais — sempre, porque não custam nada.
    lexical = avaliar_lexical(caso.resposta, caso.gabarito)
    resultado.exact_match = lexical.exact_match
    resultado.contains = lexical.contains
    resultado.f1_tokens = lexical.f1_tokens
    resultado.cobertura = lexical.cobertura
    resultado.abstencao = lexical.abstencao

    # 2. RAGAS
    if "ragas" in metricas:
        try:
            r = mod_ragas.avaliar(
                caso.pergunta, caso.resposta, caso.contexto, settings
            )
            resultado.faithfulness = r.faithfulness
            resultado.answer_relevance = r.answer_relevance
            resultado.context_relevance = r.context_relevance
            resultado.ragas_media = r.media_harmonica
            resultado.n_afirmacoes = len(r.afirmacoes)
            resultado.n_sustentadas = sum(r.vereditos)
            resultado.erros.extend(r.erros)
        except Exception as erro:  # noqa: BLE001
            resultado.erros.append(f"ragas: {erro}")

    # 3. G-Eval
    if "geval" in metricas:
        try:
            notas = g_eval.avaliar(
                caso.pergunta, caso.resposta, caso.contexto,
                criterios_geval, settings,
            )
            for nome, nota in notas.items():
                chave = f"geval_{nome.lower()}"
                resultado.geval[chave] = nota.normalizado
                resultado.geval_bruto[chave] = nota.score
                resultado.geval_metodo = nota.metodo
                if nota.erro:
                    resultado.erros.append(f"geval[{nome}]: {nota.erro}")
        except Exception as erro:  # noqa: BLE001
            resultado.erros.append(f"geval: {erro}")

    resultado.latencia = time.perf_counter() - inicio
    return resultado


def executar(
    casos: list[Caso],
    metricas: list[str] | None = None,
    criterios_geval: list[str] | None = None,
    settings: Settings | None = None,
    verboso: bool = True,
) -> Avaliacao:
    """Roda a avaliação sobre todos os casos."""
    settings = settings or SETTINGS
    metricas = metricas or ["lexical"]
    criterios_geval = criterios_geval or ["fundamentacao", "relevancia"]

    avaliacao = Avaliacao(
        juiz_model=settings.juiz_model,
        metricas_usadas=metricas,
        criterios_geval=criterios_geval if "geval" in metricas else [],
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    for i, caso in enumerate(casos, start=1):
        resultado = avaliar_caso(caso, metricas, criterios_geval, settings)
        avaliacao.resultados.append(resultado)

        if verboso:
            partes = [f"cont={resultado.contains:.0f}"]
            if "ragas" in metricas:
                partes.append(f"faith={resultado.faithfulness:.2f}")
                partes.append(f"ar={resultado.answer_relevance:.2f}")
            if resultado.geval:
                media = sum(resultado.geval.values()) / len(resultado.geval)
                partes.append(f"geval={media:.2f}")
            marca = "ERRO" if resultado.erros else "    "
            print(f"  [{marca}] {caso.id:<8} {' '.join(partes)}  ({i}/{len(casos)})")

    return avaliacao


def estimar_custo(
    n_casos: int, metricas: list[str], criterios_geval: list[str],
    settings: Settings | None = None,
) -> dict:
    """Estimativa de chamadas antes de gastar. Nenhuma chamada é feita aqui."""
    settings = settings or SETTINGS
    chat = 0
    embeddings = 0

    if "ragas" in metricas:
        # extrair afirmações + verificar + n perguntas reversas + extrair frases
        chat += n_casos * (2 + settings.ragas_n_perguntas + 1)
        embeddings += n_casos

    if "geval" in metricas:
        n_criterios = len(criterios_geval)
        # passos são gerados uma vez por critério (cache)
        chat += n_criterios
        if settings.geval_usar_logprobs:
            chat += n_casos * n_criterios
        else:
            chat += n_casos * n_criterios * settings.geval_amostras

    return {
        "chamadas_chat": chat,
        "chamadas_embeddings": embeddings,
        "total": chat + embeddings,
    }
