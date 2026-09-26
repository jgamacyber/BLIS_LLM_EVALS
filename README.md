# BLIS — Módulo 07: Avaliação (Evals)

Pipeline de avaliação para sistemas de RAG, com análise dos resultados. Avalia as saídas do RAG construído no [módulo 03/04](https://github.com/jgamacyber), ou de qualquer outro sistema, bastando trocar o arquivo de casos.

Tudo implementado do zero: as métricas, a estatística (Scott's π, Spearman, Kendall τ, PPI) e as sondagens de viés. Sem RAGAS-lib, sem DeepEval, sem scipy.

> ⚠️ **Aviso de uso.** Este material é **educacional**, voltado ao estudo de métodos de avaliação de sistemas de IA. As implementações são reproduções didáticas e simplificadas dos artigos originais, não substituem os frameworks publicados nem os resultados neles reportados.

> 📄 **Sobre os artigos.** Os artigos que fundamentam cada método são **citados ao longo de todo o trabalho**, nos docstrings dos módulos, nos comentários das decisões de projeto, nos relatórios exportados e na lista de referências ao final. Os PDFs **não** estão incluídos no repositório por questão de direitos autorais. Use as referências para localizá-los nas fontes originais.

> ⚠️ **Sobre as anotações.** O conjunto incluído tem rótulos de referência **escritos à mão para o exercício**, por uma única pessoa. Não é um estudo com múltiplos anotadores. Leia [`data/PROCEDENCIA.md`](./data/PROCEDENCIA.md) antes de reportar qualquer número.

## Artigos de referência

Cada método é atribuído ao artigo de origem no docstring do módulo que o implementa, nos comentários das decisões de projeto e nos relatórios exportados.

| Artigo | O que aparece no código | Onde |
|---|---|---|
| Es et al. (2023), **RAGAS** | Faithfulness, answer relevance e context relevance, avaliação sem gabarito | `ragas.py` |
| Liu et al. (2023), **G-Eval** | Auto-CoT dos passos de avaliação e pontuação ponderada por probabilidade | `g_eval.py` |
| Saad-Falcon et al. (2024), **ARES** | Prediction-powered inference: intervalos de confiança para juízes LLM | `ppi.py` |
| Thakur et al. (2025), **Judging the Judges** | Scott's π sobre o percentual bruto; vieses de posição, verbosidade e leniência; baselines lexicais | `concordancia.py`, `vieses.py`, `metricas_lexicais.py` |

## O que o pipeline faz

```
casos do RAG  ──┬─> métricas lexicais   (EM, contains, F1)       grátis
                ├─> RAGAS               (faithfulness, AR, CR)   ~6 chamadas/caso
                └─> G-Eval              (nota ponderada)         ~1 chamada/caso
                          │
                          ▼
                   ANÁLISE
                     ├─ alinhamento com a referência (Scott π, Spearman, Kendall)
                     ├─ custo-benefício: o juiz pago supera o baseline grátis?
                     ├─ PPI: qualidade real com intervalo de confiança
                     ├─ vieses: posição, verbosidade, leniência, autopreferência
                     └─ divergências: os casos que valem leitura manual
```

## Módulos

| Arquivo | Responsabilidade |
|---|---|
| `metricas_lexicais.py` | EM, contains, F1, cobertura, detecção de abstenção. Grátis e determinísticas |
| `ragas.py` | As três métricas do RAGAS, sem necessidade de gabarito |
| `g_eval.py` | Auto-CoT e ponderação por logprobs, com fallback por amostragem |
| `concordancia.py` | Scott's π, Cohen's κ, Spearman, Kendall τ-b, proporção de empates |
| `ppi.py` | Intervalos de confiança: clássico, só-juiz e PPI |
| `vieses.py` | Sondagens de posição, verbosidade, leniência e autopreferência |
| `dataset.py` | Carregamento e validação dos casos |
| `pipeline.py` | Orquestra as métricas e persiste os resultados |
| `analise.py` | Alinhamento, custo-benefício, PPI, divergências, exportação |
| `main.py` | CLI |
| `testes_offline.py` | 96 verificações sem API |
| `testes_mock.py` | 65 verificações das chamadas de API, com LLM simulado |

## Início rápido

```bash
python -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

pip install -r requirements.txt
cp .env.example .env               # preencha OPENROUTER_API_KEY

# Sem gastar nada:
python testes_offline.py           # 96 verificações
python testes_mock.py              # 65 verificações
python main.py validar             # checa o conjunto
python main.py inspecionar --caso-id c09
python main.py lexical             # métricas grátis e alinhamento
python main.py ppi --metrica f1_tokens

# Com a chave:
python main.py avaliar --metricas ragas geval
python main.py analisar --tudo
python main.py vieses --posicao
```

## Três ideias que este módulo defende

### 1. Concordância bruta engana; use Scott's π

Thakur et al. são explícitos: *"juízes com alta concordância ainda podem atribuir notas muito diferentes"*. Se 90% dos seus casos são positivos, um juiz que diz "sim" para tudo acerta 90%, e tem π ≈ 0, porque não está medindo nada.

O comando `analisar` reporta os dois lado a lado, junto com o **desvio de escala**, que mostra se o juiz é sistematicamente leniente mesmo quando correlaciona bem.

### 2. O juiz caro precisa superar o baseline grátis

O mesmo artigo encontra que a métrica lexical `contains` ranqueia modelos quase tão bem quanto juízes LLM de porte médio, porque, embora erre mais, seus vieses são mais consistentes.

`analisar` traz uma tabela explícita de custo-benefício. Se o G-Eval não bate o `contains`, ele não está pagando o próprio custo, e isso aparece como conclusão impressa.

**Um resultado que o conjunto incluído produz**, reproduzível sem gastar nada:

```
python main.py lexical
```

O `contains` tem **Scott's π negativo** na dimensão *fundamentação*. Não é bug: `contains` mede se o gabarito aparece na resposta, o que **não é** fundamentação no contexto. Os casos c09 a c12 são alucinações por adição, em que a resposta contém o gabarito correto e também detalhes inventados. O `contains` aprova; a referência reprova. É a demonstração concreta de que usar métrica lexical como proxy de fidelidade inverte o sinal.

### 3. Estreiteza não é acurácia

Um intervalo de confiança calculado só sobre as notas do juiz é estreito **e errado**, se o juiz for enviesado. Mais dados não corrigem viés sistemático, só dão mais confiança na resposta errada.

O PPI do ARES resolve isso: usa um punhado de anotações humanas para estimar o erro do juiz e corrigi-lo. `python main.py ppi` compara os três caminhos e mostra qual cobre o valor verdadeiro.

## Formato dos casos

```json
[
  {
    "id": "c01",
    "pergunta": "a consulta feita ao sistema",
    "contexto": "as passagens que o recuperador devolveu",
    "resposta": "o que o gerador respondeu",
    "gabarito": "a resposta esperada (vazio se não há resposta possível)",
    "anotacao_humana": {
      "fundamentacao": 1,
      "relevancia": 1,
      "contexto_relevante": 1,
      "nota_global": 5
    }
  }
]
```

As dimensões binárias são 0 ou 1; `nota_global` é uma escala de 1 a 5. Casos **sem gabarito** são perguntas que o contexto não cobre. Ali, abster-se é a resposta correta, e as métricas pontuam a abstenção como acerto.

Os contextos do conjunto incluído são **resumos autorais em português**, escritos a partir da leitura dos artigos. Não são traduções nem reproduções dos textos originais.

`python main.py validar` checa a estrutura, o balanceamento das classes e avisa quando o conjunto é pequeno demais para intervalos úteis.

## O que observar

1. **Inclua perguntas sem resposta possível.** Sem elas, não dá para distinguir um sistema que sabe responder de um que sempre responde. O conjunto incluído tem 4.
2. **Separe alucinação de erro factual.** Uma métrica de acerto trata as duas como falha; a `faithfulness` as separa. A diferença importa: uma vem de um gerador criativo demais, a outra de um recuperador ruim.
3. **Não use o mesmo modelo para gerar e julgar.** O G-Eval alerta para o viés a favor de texto gerado por LLM. O CLI avisa quando `JUIZ_MODEL` é igual a `MODELO_AVALIADO`.
4. **Meça a concordância entre humanos primeiro.** É o teto do que qualquer juiz automático pode atingir. `python main.py concordancia a.json b.json` faz isso.
5. **Leia as divergências.** `analisar --divergencias` lista os casos em que a métrica discorda da referência. É onde se aprende o que ela não vê.

## Reprodução

Roteiro completo com custos e saídas esperadas: **[REPRODUTIBILIDADE.md](./REPRODUTIBILIDADE.md)**.

## Licença

MIT, ver [`LICENSE`](./LICENSE).

A licença cobre o **código deste repositório**. Os artigos citados pertencem a seus respectivos autores e editoras.

## Referências

Todos os artigos abaixo são citados no código, nos pontos em que o método correspondente é implementado. Os PDFs não são distribuídos aqui.

1. Es, S., James, J., Espinosa-Anke, L. & Schockaert, S. (2023). *Ragas: Automated Evaluation of Retrieval Augmented Generation.*
2. Liu, Y., Iter, D., Xu, Y., Wang, S., Xu, R. & Zhu, C. (2023). *G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment.* EMNLP.
3. Saad-Falcon, J., Khattab, O., Potts, C. & Zaharia, M. (2024). *ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems.* NAACL.
4. Thakur, A. S., Choudhary, K., Ramayapally, V. S., Vaidyanathan, S. & Hupkes, D. (2025). *Judging the Judges: Evaluating Alignment and Vulnerabilities in LLMs-as-Judges.* GEM² Workshop, ACL.
5. Angelopoulos, A. N. et al. (2023). *Prediction-Powered Inference.* Base estatística do PPI usado pelo ARES.
6. Liu, N. F. et al. (2023). *Lost in the Middle.* Citado pelo RAGAS para justificar a métrica de relevância de contexto.
