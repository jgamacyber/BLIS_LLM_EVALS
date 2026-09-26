# Procedência dos dados

Leia isto antes de reportar qualquer número deste repositório.

## O que é `casos.json`

30 casos de avaliação de um sistema de RAG. Cada caso tem:

| Campo | Conteúdo |
|---|---|
| `pergunta` | A consulta feita ao sistema |
| `contexto` | As passagens que o recuperador devolveu |
| `resposta` | O que o gerador respondeu |
| `gabarito` | A resposta esperada (vazio quando não há resposta possível) |
| `anotacao_humana` | Rótulos de referência por dimensão |

Os **contextos** são **resumos autorais em português**, escritos a partir da leitura dos mesmos artigos usados no corpus do repositório de RAG (RAG, DPR, Lost in the Middle, HyDE, Self-RAG, RAPTOR, BEIR, MTEB). Não são traduções nem trechos reproduzidos dos textos originais, e os PDFs dos artigos não são distribuídos neste repositório.

## ⚠️ As anotações NÃO vêm de um estudo com anotadores humanos

Este é o ponto mais importante deste arquivo.

O campo `anotacao_humana` contém **rótulos de referência escritos à mão para o exercício**, por uma única pessoa, com o objetivo de exercitar o pipeline de avaliação. Não houve:

- múltiplos anotadores independentes;
- medição de concordância entre anotadores;
- protocolo de anotação cego;
- resolução de divergências por discussão.

Compare com o que os artigos de referência fizeram de verdade:

- **RAGAS** (Es et al., 2023) construiu o WikiEval com **dois anotadores** por item, reportando concordância de ~95% em fidelidade e relevância de contexto e ~90% em relevância da resposta, com divergências resolvidas por discussão.
- **ARES** (Saad-Falcon et al., 2024) usa um conjunto de validação de preferência humana com **~150 pontos anotados ou mais**.
- **Judging the Judges** (Thakur et al., 2025) usa anotação manual sobre 400 questões amostradas do TriviaQA.

## O que isso permite e o que não permite

**Permite:**

- Exercitar e verificar todo o pipeline de ponta a ponta
- Entender como cada métrica se comporta e onde falha
- Comparar RAGAS, G-Eval e métricas lexicais entre si
- Demonstrar o PPI, a concordância corrigida pelo acaso e a detecção de vieses

**Não permite:**

- Afirmar que "o G-Eval tem Scott's π de X com humanos", o denominador aqui não é "humanos", é uma anotação de referência única
- Comparar com os números publicados nos artigos
- Sustentar qualquer conclusão sobre qual métrica é melhor em geral

Os números que saem deste conjunto medem **concordância com esta anotação de referência**, e é assim que devem ser reportados.

## Casos construídos de propósito

Os casos foram escritos para cobrir os modos de falha que as métricas precisam distinguir:

| Tipo | Casos | Para que serve |
|---|---|---|
| Corretas e fundamentadas | c01–c08 | Piso de verdadeiros positivos |
| Alucinação por adição | c09–c12 | A resposta acerta, mas acrescenta detalhes ausentes do contexto, testa `faithfulness` |
| Factualmente erradas | c13–c15 | Contradizem o contexto |
| Evasivas | c16–c18 | Fundamentadas, mas não respondem, testa `answer relevance` |
| Verbosas e corretas | c19–c20 | Testa viés de verbosidade do juiz |
| Concisas e corretas | c21–c22 | Contraste com as verbosas |
| Abstenção correta | c23–c24, c29–c30 | O sistema recusa quando o contexto não cobre |
| Abstenção indevida | c25 | O sistema recusa embora a resposta esteja no contexto |
| Parcialmente corretas | c26–c28 | Testa crédito parcial |

A distinção entre **alucinação por adição** (c09–c12) e **erro factual** (c13–c15) é deliberada: uma métrica de acerto trata as duas como falha, mas a `faithfulness` do RAGAS as separa e a diferença importa, porque a primeira sai de um gerador bom demais e a segunda de um recuperador ruim.

## Usar seus próprios dados

Substitua `casos.json` pelas saídas reais do seu sistema. O formato está documentado no `README.md` na raiz, e `python main.py validar` checa a estrutura.

Se for anotar você mesmo: use **pelo menos dois anotadores** em uma amostra, meça a concordância entre eles com `python main.py concordancia`, e só então trate a anotação como referência. A concordância entre humanos é o teto do que qualquer juiz automático pode atingir — sem medi-la, não há como saber se um π de 0,6 é ruim ou é o máximo possível naquela tarefa.
