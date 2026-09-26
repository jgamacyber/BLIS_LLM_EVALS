"""
Carregamento e validação dos casos de avaliação.

Um caso é uma linha do que o sistema de RAG produziu: a pergunta, o contexto que
o recuperador trouxe, a resposta que o gerador deu e — quando existe — o
gabarito e a anotação de referência.

A anotação de referência é o que permite medir o juiz automático. Sem ela, as
métricas rodam mas não há como saber se estão certas.

Veja `data/PROCEDENCIA.md` para a origem e as limitações do conjunto incluído.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

CAMINHO_CASOS = Path("data/casos.json")

# Dimensões binárias esperadas na anotação (0 ou 1).
DIMENSOES_BINARIAS = ["fundamentacao", "relevancia", "contexto_relevante"]
# Dimensão em escala.
DIMENSAO_ESCALA = "nota_global"


@dataclass
class Caso:
    """Um caso de avaliação: o que o sistema produziu, e o que deveria."""

    id: str
    pergunta: str
    contexto: str
    resposta: str
    gabarito: object = ""
    contexto_id: str = ""
    anotacao_humana: dict = field(default_factory=dict)
    metadados: dict = field(default_factory=dict)

    @property
    def tem_anotacao(self) -> bool:
        return bool(self.anotacao_humana)

    @property
    def tem_gabarito(self) -> bool:
        return bool(self.gabarito)

    @property
    def sem_resposta_possivel(self) -> bool:
        """Casos em que o contexto não cobre a pergunta."""
        return not self.tem_gabarito

    def anotacao(self, dimensao: str, padrao: float = 0.0) -> float:
        valor = self.anotacao_humana.get(dimensao, padrao)
        try:
            return float(valor)
        except (TypeError, ValueError):
            return padrao


# --------------------------------------------------------------------------- #
# Carregamento
# --------------------------------------------------------------------------- #


def carregar(caminho: str | Path = CAMINHO_CASOS) -> list[Caso]:
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(
            f"Casos não encontrados: {caminho}. "
            f"Veja data/PROCEDENCIA.md para o formato esperado."
        )

    dados = json.loads(caminho.read_text(encoding="utf-8"))
    if not isinstance(dados, list):
        raise ValueError("o arquivo de casos deve conter uma lista")

    casos = []
    for i, d in enumerate(dados):
        faltando = {"pergunta", "contexto", "resposta"} - set(d)
        if faltando:
            raise ValueError(f"caso {i} sem os campos obrigatórios: {faltando}")
        casos.append(
            Caso(
                id=d.get("id", f"caso_{i:03d}"),
                pergunta=d["pergunta"],
                contexto=d["contexto"],
                resposta=d["resposta"],
                gabarito=d.get("gabarito", ""),
                contexto_id=d.get("contexto_id", ""),
                anotacao_humana=d.get("anotacao_humana", {}),
                metadados=d.get("metadados", {}),
            )
        )
    return casos


def salvar(casos: list[Caso], caminho: str | Path) -> Path:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "id": c.id,
            "contexto_id": c.contexto_id,
            "pergunta": c.pergunta,
            "contexto": c.contexto,
            "resposta": c.resposta,
            "gabarito": c.gabarito,
            "anotacao_humana": c.anotacao_humana,
            "metadados": c.metadados,
        }
        for c in casos
    ]
    caminho.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return caminho


# --------------------------------------------------------------------------- #
# Divisão rotulado / não rotulado (para o PPI)
# --------------------------------------------------------------------------- #


def dividir_para_ppi(
    casos: list[Caso], n_rotulado: int | None = None, semente: int = 42
) -> tuple[list[Caso], list[Caso]]:
    """
    Separa os casos em conjunto rotulado e não rotulado, como o ARES espera.

    Na prática você teria muitos casos sem anotação e poucos com. Aqui, como
    todos os casos vêm anotados, `n_rotulado` permite simular esse cenário:
    os demais entram como "não rotulados" (a anotação é ignorada), o que deixa
    verificar se o intervalo PPI cobre o valor humano verdadeiro.

    Devolve (rotulados, nao_rotulados).
    """
    import random

    anotados = [c for c in casos if c.tem_anotacao]
    if n_rotulado is None or n_rotulado >= len(anotados):
        return anotados, [c for c in casos if not c.tem_anotacao]

    rng = random.Random(semente)
    embaralhados = list(anotados)
    rng.shuffle(embaralhados)
    return embaralhados[:n_rotulado], embaralhados[n_rotulado:]


# --------------------------------------------------------------------------- #
# Validação
# --------------------------------------------------------------------------- #


def validar(casos: list[Caso]) -> list[str]:
    """Procura problemas estruturais. Devolve a lista de avisos."""
    avisos: list[str] = []

    vistos: set[str] = set()
    for c in casos:
        if c.id in vistos:
            avisos.append(f"id duplicado: {c.id}")
        vistos.add(c.id)

    for c in casos:
        if not c.contexto.strip():
            avisos.append(f"'{c.id}': contexto vazio")
        if not c.resposta.strip():
            avisos.append(f"'{c.id}': resposta vazia")
        if len(c.contexto) > 20000:
            avisos.append(f"'{c.id}': contexto muito longo ({len(c.contexto)} chars)")

    anotados = [c for c in casos if c.tem_anotacao]
    if not anotados:
        avisos.append(
            "nenhum caso tem anotação de referência: as métricas vão rodar, mas "
            "não haverá como medir o alinhamento do juiz nem calcular o PPI"
        )
    elif len(anotados) < 20:
        avisos.append(
            f"apenas {len(anotados)} casos anotados. O ARES usa ~150; com menos "
            f"de 20 os intervalos de confiança ficam largos demais para comparar "
            f"sistemas"
        )

    for c in anotados:
        for dimensao in DIMENSOES_BINARIAS:
            if dimensao in c.anotacao_humana:
                valor = c.anotacao_humana[dimensao]
                if valor not in (0, 1, 0.0, 1.0, True, False):
                    avisos.append(
                        f"'{c.id}': '{dimensao}' deveria ser 0 ou 1, veio {valor!r}"
                    )

    # Desbalanceamento: importante porque a concordância bruta engana quando
    # uma classe domina.
    for dimensao in DIMENSOES_BINARIAS:
        valores = [
            c.anotacao(dimensao) for c in anotados if dimensao in c.anotacao_humana
        ]
        if len(valores) >= 10:
            positivos = sum(valores) / len(valores)
            if positivos > 0.9 or positivos < 0.1:
                avisos.append(
                    f"'{dimensao}' está desbalanceada ({positivos:.0%} positivos): "
                    f"a concordância bruta vai parecer alta por construção — "
                    f"leia o Scott's π, não o percentual"
                )

    return avisos


def resumo(casos: list[Caso]) -> str:
    anotados = sum(1 for c in casos if c.tem_anotacao)
    com_gabarito = sum(1 for c in casos if c.tem_gabarito)
    sem_resposta = sum(1 for c in casos if c.sem_resposta_possivel)
    contextos = len({c.contexto_id for c in casos if c.contexto_id})

    return (
        f"{len(casos)} casos | {anotados} anotados | {com_gabarito} com gabarito "
        f"| {sem_resposta} sem resposta possível | {contextos} contextos distintos"
    )


def distribuicao_anotacoes(casos: list[Caso]) -> dict:
    """Distribuição das anotações por dimensão, para inspeção rápida."""
    anotados = [c for c in casos if c.tem_anotacao]
    saida: dict = {}

    for dimensao in DIMENSOES_BINARIAS:
        valores = [
            c.anotacao(dimensao) for c in anotados if dimensao in c.anotacao_humana
        ]
        if valores:
            saida[dimensao] = {
                "n": len(valores),
                "positivos": int(sum(valores)),
                "taxa": sum(valores) / len(valores),
            }

    valores = [
        c.anotacao(DIMENSAO_ESCALA)
        for c in anotados
        if DIMENSAO_ESCALA in c.anotacao_humana
    ]
    if valores:
        contagem: dict = {}
        for v in valores:
            contagem[v] = contagem.get(v, 0) + 1
        saida[DIMENSAO_ESCALA] = {
            "n": len(valores),
            "media": sum(valores) / len(valores),
            "distribuicao": dict(sorted(contagem.items())),
        }

    return saida
