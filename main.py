"""
BLIS — Módulo 07: Avaliação (Evals)
Pipeline de avaliação de RAG com análise dos resultados.

Uso:
    python main.py validar                   # checa o conjunto (sem API)
    python main.py inspecionar               # vê um caso e as anotações (sem API)
    python main.py lexical                   # métricas lexicais (sem API)
    python main.py avaliar --metricas ragas  # roda as métricas (custa)
    python main.py analisar                  # análise da última avaliação
    python main.py ppi                       # intervalos de confiança
    python main.py vieses                    # sonda o juiz (custa)
    python main.py concordancia a.json b.json  # concordância entre 2 anotadores
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import analise as mod_analise
import vieses as mod_vieses
from concordancia import (
    cohen_kappa,
    concordancia_bruta,
    interpretar_kappa,
    scott_pi,
)
from config import SETTINGS, resumo_config
from dataset import (
    DIMENSOES_BINARIAS,
    carregar,
    distribuicao_anotacoes,
    resumo,
    validar,
)
from metricas_lexicais import avaliar_lexical, eh_abstencao
from pipeline import CAMINHO_RESULTADOS, Avaliacao, estimar_custo, executar

LARGURA = 76


def banner(titulo: str) -> None:
    print("\n" + "=" * LARGURA)
    print(f"  {titulo}")
    print("=" * LARGURA)
    print(f"  {resumo_config()}")


def _carregar(args) -> list:
    casos = carregar(getattr(args, "casos", "data/casos.json"))
    ids = getattr(args, "ids", None)
    if ids:
        alvo = set(ids)
        casos = [c for c in casos if c.id in alvo]
    limite = getattr(args, "limite", 0)
    if limite:
        casos = casos[:limite]
    if not casos:
        print("Nenhum caso após os filtros.")
        sys.exit(1)
    return casos


# --------------------------------------------------------------------------- #
# Comandos sem API
# --------------------------------------------------------------------------- #


def cmd_validar(args) -> None:
    banner("VALIDAÇÃO DO CONJUNTO")
    casos = carregar(args.casos)
    print(f"\n  {resumo(casos)}")

    print("\nDistribuição das anotações de referência:")
    for dimensao, info in distribuicao_anotacoes(casos).items():
        if "taxa" in info:
            print(f"  {dimensao:<22} {info['positivos']}/{info['n']} positivos "
                  f"({info['taxa']:.0%})")
        else:
            dist = " ".join(f"{k:g}:{v}" for k, v in info["distribuicao"].items())
            print(f"  {dimensao:<22} média {info['media']:.2f}   [{dist}]")

    avisos = validar(casos)
    if not avisos:
        print("\nNenhum problema estrutural encontrado.")
    else:
        print(f"\n{len(avisos)} aviso(s):")
        for a in avisos:
            print(f"  [aviso] {a}")

    print("\n  Lembrete: as anotações deste conjunto são de referência, escritas")
    print("  para o exercício — veja data/PROCEDENCIA.md antes de reportar números.")


def cmd_inspecionar(args) -> None:
    banner("INSPEÇÃO DE CASO")
    casos = carregar(args.casos)
    caso = next((c for c in casos if c.id == args.caso_id), casos[0])

    print(f"\n  id: {caso.id}   contexto: {caso.contexto_id}")
    print(f"\n  PERGUNTA:\n    {caso.pergunta}")
    print(f"\n  CONTEXTO ({len(caso.contexto)} chars):\n    {caso.contexto[:700]}...")
    print(f"\n  RESPOSTA:\n    {caso.resposta}")
    print(f"\n  GABARITO: {caso.gabarito or '(sem resposta possível)'}")

    print("\n  ANOTAÇÃO DE REFERÊNCIA:")
    for dimensao, valor in caso.anotacao_humana.items():
        print(f"    {dimensao:<22} {valor}")

    lexical = avaliar_lexical(caso.resposta, caso.gabarito)
    print("\n  MÉTRICAS LEXICAIS (grátis):")
    for nome, valor in lexical.como_dict().items():
        print(f"    {nome:<22} {valor}")


def cmd_lexical(args) -> None:
    """Avaliação completa só com as métricas gratuitas."""
    banner("MÉTRICAS LEXICAIS  (sem API, sem custo)")
    casos = _carregar(args)

    avaliacao = executar(casos, metricas=["lexical"], verboso=False)
    avaliacao.salvar(CAMINHO_RESULTADOS)

    n = len(avaliacao.resultados)
    print(f"\n  {n} casos avaliados\n")
    print(f"  {'Métrica':<20} {'Média':>8}")
    print("  " + "-" * 30)
    for metrica in ("exact_match", "contains", "f1_tokens", "cobertura"):
        media = sum(avaliacao.serie(metrica)) / n
        print(f"  {metrica:<20} {media:>8.1%}")

    abstencoes = sum(1 for r in avaliacao.resultados if r.abstencao)
    print(f"\n  abstenções detectadas: {abstencoes}/{n}")

    mod_analise.imprimir_alinhamento(avaliacao)
    print(f"\nResultados salvos em {CAMINHO_RESULTADOS}")


def cmd_concordancia(args) -> None:
    """Concordância entre dois conjuntos de anotações (dois anotadores)."""
    banner("CONCORDÂNCIA ENTRE ANOTADORES  (sem API)")

    def ler(caminho: str) -> dict:
        dados = json.loads(Path(caminho).read_text(encoding="utf-8"))
        return {d["id"]: d.get("anotacao_humana", {}) for d in dados}

    a, b = ler(args.anotador_a), ler(args.anotador_b)
    comuns = sorted(set(a) & set(b))

    if not comuns:
        print("\n  Os dois arquivos não têm nenhum id em comum.")
        sys.exit(1)

    print(f"\n  {len(comuns)} casos em comum\n")
    print(f"  {'Dimensão':<22} {'Concord.':>9} {'Scott π':>9} {'Cohen κ':>9}  Leitura")
    print("  " + "-" * 72)

    for dimensao in DIMENSOES_BINARIAS + ["nota_global"]:
        va = [a[i].get(dimensao) for i in comuns if dimensao in a[i] and dimensao in b[i]]
        vb = [b[i].get(dimensao) for i in comuns if dimensao in a[i] and dimensao in b[i]]
        if not va:
            continue
        pi = scott_pi(va, vb)
        print(
            f"  {dimensao:<22} {concordancia_bruta(va, vb):>8.1%} "
            f"{pi:>9.3f} {cohen_kappa(va, vb):>9.3f}  {interpretar_kappa(pi)}"
        )

    print("\n  A concordância entre humanos é o TETO do que um juiz automático")
    print("  pode atingir. O RAGAS reporta ~95% entre seus dois anotadores.")


# --------------------------------------------------------------------------- #
# Comandos que usam a API
# --------------------------------------------------------------------------- #


def cmd_avaliar(args) -> None:
    banner("AVALIAÇÃO")
    casos = _carregar(args)
    metricas = args.metricas or ["ragas"]
    criterios = args.criterios or ["fundamentacao", "relevancia"]

    aviso = mod_vieses.verificar_autopreferencia(SETTINGS)
    aviso.imprimir()

    custo = estimar_custo(len(casos), metricas, criterios)
    print(f"\n  {len(casos)} casos | métricas: {', '.join(metricas)}")
    if "geval" in metricas:
        print(f"  critérios G-Eval: {', '.join(criterios)}")
    print(f"  estimativa: {custo['chamadas_chat']} chamadas de chat + "
          f"{custo['chamadas_embeddings']} de embeddings")

    if not args.sim and custo["total"] > 60:
        resposta = input(f"\n  São ~{custo['total']} chamadas. Continuar? [s/N] ")
        if resposta.strip().lower() not in {"s", "sim", "y"}:
            print("  Cancelado.")
            return

    print()
    avaliacao = executar(casos, metricas, criterios, SETTINGS)
    avaliacao.salvar(CAMINHO_RESULTADOS)

    n = len(avaliacao.resultados)
    print(f"\n  Médias ({n} casos):")
    for metrica in avaliacao.nomes_metricas():
        valores = avaliacao.serie(metrica)
        if any(v != 0 for v in valores):
            print(f"    {metrica:<26} {sum(valores) / n:>7.3f}")

    erros = sum(1 for r in avaliacao.resultados if r.erros)
    if erros:
        print(f"\n  [!] {erros} caso(s) com erro — veja o JSON salvo")

    print(f"\nResultados salvos em {CAMINHO_RESULTADOS}")
    print("Analise com: python main.py analisar")


def cmd_analisar(args) -> None:
    banner("ANÁLISE DOS RESULTADOS")
    avaliacao = Avaliacao.carregar(CAMINHO_RESULTADOS)
    print(f"  avaliação de {avaliacao.timestamp} | juiz {avaliacao.juiz_model}")
    print(f"  {len(avaliacao.resultados)} casos | "
          f"métricas: {', '.join(avaliacao.metricas_usadas)}")

    mod_analise.imprimir_alinhamento(avaliacao, args.limiar)
    mod_analise.imprimir_custo_beneficio(avaliacao, args.dimensao, args.limiar)

    if args.divergencias or args.tudo:
        mod_analise.imprimir_divergencias(avaliacao, args.dimensao, limiar=args.limiar)

    if args.csv or args.tudo:
        print(f"\n{mod_analise.exportar_csv(avaliacao, 'resultados/analise.csv')}")
    if args.markdown or args.tudo:
        print(f"{mod_analise.exportar_markdown(avaliacao, 'resultados/analise.md')}")


def cmd_ppi(args) -> None:
    banner("INTERVALOS DE CONFIANÇA (PPI)")
    avaliacao = Avaliacao.carregar(CAMINHO_RESULTADOS)
    mod_analise.imprimir_ppi(
        avaliacao, args.metrica, args.dimensao, args.n_rotulado, SETTINGS.ppi_alpha
    )


def cmd_vieses(args) -> None:
    banner("SONDAGEM DE VIESES DO JUIZ")
    casos = _carregar(args)

    mod_vieses.verificar_autopreferencia(SETTINGS).imprimir()

    # Leniência e verbosidade usam a avaliação já feita — custo zero.
    try:
        avaliacao = Avaliacao.carregar(CAMINHO_RESULTADOS)
        com_anotacao = avaliacao.com_anotacao(args.dimensao)
        if com_anotacao:
            notas_juiz = [c.metrica(args.metrica) for c in com_anotacao]
            notas_humanas = [
                float(c.anotacao_humana[args.dimensao]) for c in com_anotacao
            ]
            mod_vieses.testar_leniencia(notas_juiz, notas_humanas).imprimir()
            mod_vieses.testar_vies_verbosidade(
                [c.resposta for c in com_anotacao], notas_juiz, notas_humanas
            ).imprimir()
    except FileNotFoundError:
        print("\n  [i] rode `python main.py avaliar` antes para as sondagens")
        print("      de leniência e verbosidade (que são grátis).")

    if not args.posicao:
        print("\n  Viés de posição não testado (use --posicao; custa 2 chamadas")
        print("  por par).")
        return

    # Monta pares: resposta boa contra resposta ruim, sobre a mesma pergunta.
    from dataset import carregar as carregar_casos

    todos = carregar_casos(args.casos)
    por_contexto: dict[str, list] = {}
    for c in todos:
        if c.tem_anotacao:
            por_contexto.setdefault(c.contexto_id, []).append(c)

    pares = []
    for _, grupo in por_contexto.items():
        boas = [c for c in grupo if c.anotacao("nota_global") >= 4]
        ruins = [c for c in grupo if c.anotacao("nota_global") <= 2]
        for boa, ruim in zip(boas, ruins):
            pares.append({
                "pergunta": boa.pergunta,
                "contexto": boa.contexto,
                "resposta_1": boa.resposta,
                "resposta_2": ruim.resposta,
            })

    pares = pares[: args.n_pares]
    if not pares:
        print("\n  Não foi possível montar pares (preciso de respostas boas e")
        print("  ruins sobre o mesmo contexto).")
        return

    print(f"\n  Testando {len(pares)} pares, cada um nas duas ordens "
          f"({len(pares) * 2} chamadas)")
    mod_vieses.testar_vies_posicao(pares, SETTINGS).imprimir()


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BLIS — Módulo 07: Avaliação (Evals)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="comando", required=True)

    def add_casos(p):
        p.add_argument("--casos", default="data/casos.json")

    def add_filtros(p):
        p.add_argument("--ids", nargs="+")
        p.add_argument("--limite", type=int, default=0)

    p = sub.add_parser("validar", help="checa o conjunto (sem API)")
    add_casos(p)
    p.set_defaults(func=cmd_validar)

    p = sub.add_parser("inspecionar", help="vê um caso em detalhe (sem API)")
    add_casos(p)
    p.add_argument("--caso-id", default="")
    p.set_defaults(func=cmd_inspecionar)

    p = sub.add_parser("lexical", help="métricas lexicais (sem API)")
    add_casos(p)
    add_filtros(p)
    p.set_defaults(func=cmd_lexical)

    p = sub.add_parser("concordancia", help="concordância entre 2 anotadores")
    p.add_argument("anotador_a")
    p.add_argument("anotador_b")
    p.set_defaults(func=cmd_concordancia)

    p = sub.add_parser("avaliar", help="roda as métricas (custa)")
    add_casos(p)
    add_filtros(p)
    p.add_argument("--metricas", nargs="+", choices=["lexical", "ragas", "geval"])
    p.add_argument("--criterios", nargs="+",
                   choices=["fundamentacao", "relevancia", "coerencia",
                            "completude", "concisao"])
    p.add_argument("--sim", action="store_true", help="não pede confirmação")
    p.set_defaults(func=cmd_avaliar)

    p = sub.add_parser("analisar", help="análise da última avaliação (sem API)")
    p.add_argument("--dimensao", default="fundamentacao")
    p.add_argument("--limiar", type=float, default=0.5)
    p.add_argument("--divergencias", action="store_true")
    p.add_argument("--csv", action="store_true")
    p.add_argument("--markdown", action="store_true")
    p.add_argument("--tudo", action="store_true")
    p.set_defaults(func=cmd_analisar)

    p = sub.add_parser("ppi", help="intervalos de confiança (sem API)")
    p.add_argument("--metrica", default="faithfulness")
    p.add_argument("--dimensao", default="fundamentacao")
    p.add_argument("--n-rotulado", type=int, default=10)
    p.set_defaults(func=cmd_ppi)

    p = sub.add_parser("vieses", help="sonda os vieses do juiz")
    add_casos(p)
    add_filtros(p)
    p.add_argument("--metrica", default="faithfulness")
    p.add_argument("--dimensao", default="fundamentacao")
    p.add_argument("--posicao", action="store_true",
                   help="testa viés de posição (custa)")
    p.add_argument("--n-pares", type=int, default=5)
    p.set_defaults(func=cmd_vieses)

    return parser


def main() -> None:
    args = construir_parser().parse_args()
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nInterrompido.")
        sys.exit(130)
    except Exception as erro:  # noqa: BLE001
        print(f"\n[erro] {erro}")
        sys.exit(1)


if __name__ == "__main__":
    main()
