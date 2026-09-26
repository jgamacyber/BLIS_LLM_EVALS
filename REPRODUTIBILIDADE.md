# Documento de Reprodutibilidade

Roteiro para reproduzir todos os resultados deste módulo na sua máquina. Cada comando traz o que esperar e quanto custa.

Boa parte deste módulo roda **sem gastar nada** — a estatística toda (concordância, correlações, PPI) é aritmética local.

---

## 1. Ambiente

### Requisitos

- Python **3.10 ou superior** (o código usa `X | None`)
- Uma chave da OpenRouter: https://openrouter.ai/keys — só para a seção 5 em diante

```bash
python --version      # deve mostrar 3.10+

python -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows (PowerShell)

pip install -r requirements.txt
cp .env.example .env              # Windows: copy .env.example .env
```

Duas dependências apenas: `openai` e `python-dotenv`.

### Configuração importante

```
OPENROUTER_API_KEY=sk-or-v1-...
JUIZ_MODEL=openai/gpt-4o-mini
MODELO_AVALIADO=openai/gpt-4o-mini
```

**Mantenha `JUIZ_MODEL` diferente de `MODELO_AVALIADO` sempre que possível.** O G-Eval alerta que juízes LLM preferem texto gerado por LLM; usar o mesmo modelo dos dois lados infla os resultados. O CLI avisa quando os dois coincidem — o conjunto incluído neste repositório não foi gerado por nenhum modelo, então para ele o aviso é inofensivo.

---

## 2. Validação sem custo

```bash
python testes_offline.py     # 96 verificações
python testes_mock.py        # 65 verificações
```

**Esperado:** `Todos os testes passaram.` nos dois.

Os testes offline conferem a estatística contra valores calculados à mão — por exemplo, que `scott_pi([1,1,1,1,0,0,0,0], [1,1,1,0,1,0,0,0])` dá exatamente 0,5. Se essa conta estiver errada, todos os números do relatório ficam errados em silêncio.

Os testes com LLM simulado conferem, entre outras coisas, que a **ponderação por probabilidade do G-Eval** reproduz o valor esperado: com a distribuição {1:0,1 · 2:0,1 · 3:0,4 · 4:0,3 · 5:0,1}, o score tem que ser 3,2. É a peça do artigo mais fácil de implementar errado sem perceber.

---

## 3. Explorar o conjunto (sem custo)

```bash
python main.py validar
```

**Esperado:**

```
30 casos | 30 anotados | 26 com gabarito | 4 sem resposta possível | 8 contextos distintos

Distribuição das anotações de referência:
  fundamentacao          23/30 positivos (77%)
  relevancia             26/30 positivos (87%)
  contexto_relevante     26/30 positivos (87%)
  nota_global            média 3.57   [1:6 2:5 4:4 5:15]
```

O desbalanceamento (77% e 87% positivos) é exatamente a situação em que a concordância bruta engana — e por isso o relatório insiste no Scott's π.

```bash
python main.py inspecionar --caso-id c09
```

Mostra um caso de **alucinação por adição**: a resposta acerta os 400 milhões de parâmetros (que estão no contexto) e acrescenta um custo de treino inventado. Repare que `contains` = 1,0 enquanto a referência marca `fundamentacao` = 0.

---

## 4. Métricas lexicais e o primeiro achado (sem custo)

```bash
python main.py lexical
```

**Esperado** — médias sobre os 30 casos:

| Métrica | Valor |
|---|---|
| exact_match | 20,0% |
| contains | 36,7% |
| f1_tokens | 41,6% |
| cobertura | 73,4% |

E o alinhamento com a referência na dimensão *fundamentação*:

| Métrica | Concordância | Scott π | Spearman | Desvio |
|---|---:|---:|---:|---:|
| f1_tokens | 63,3% | 0,246 | 0,403 | −0,35 |
| contains | 40,0% | **−0,222** | −0,071 | −0,40 |

**O π negativo do `contains` é o achado, não um bug.** `contains` verifica se o gabarito aparece na resposta; `fundamentacao` verifica se a resposta inteira se sustenta no contexto. Nos casos c09–c12 a resposta contém o gabarito **e** informação inventada: `contains` aprova, a referência reprova. Fazer uma métrica lexical de proxy para fidelidade não só falha — inverte o sinal.

Esses números são **determinísticos**: reproduzem exatamente, porque nenhuma API é chamada.

---

## 5. PPI — intervalos de confiança (sem custo)

```bash
python main.py ppi --metrica f1_tokens --dimensao fundamentacao --n-rotulado 10
```

**Esperado:**

```
Método                    Estimativa  IC inferior  IC superior    Largura
só anotação humana             0.600        0.280        0.920      0.640
só juiz (inválido)             0.474        0.308        0.641      0.333
PPI (ARES)                     0.776        0.445        1.107      0.662

Valor verdadeiro (média de TODAS as anotações): 0.767
  só anotação humana       cobre      (erro: 0.167)
  só juiz (inválido)       NÃO cobre  (erro: 0.292)
  PPI (ARES)               cobre      (erro: 0.009)
```

Três coisas para ler nessa tabela:

1. **"Só juiz" é o mais estreito (0,333) e o único que erra o alvo.** Estreiteza não é acurácia. Aumentar o número de casos avaliados estreitaria ainda mais esse intervalo — e continuaria errado, porque nenhuma quantidade de dados corrige um viés sistemático.
2. **O PPI erra por 0,009** contra 0,167 de usar só as 10 anotações. O retificador identificou que o juiz subestima em 0,302 e corrigiu.
3. **Aqui o PPI não estreitou o intervalo** (0,662 contra 0,640). É honesto e esperado: o `f1_tokens` é um juiz ruim, então os resíduos variam muito e essa variância entra na conta. Um juiz melhor estreita. Depois de rodar a seção 6, repita com `--metrica faithfulness` e compare.

Valores **determinísticos** (semente fixa em 42).

---

## 6. RAGAS e G-Eval (**custa**)

```bash
python main.py avaliar --metricas ragas
```

**Custo:** o RAGAS faz por caso: 1 chamada para extrair afirmações, 1 para verificá-las, *n* para as perguntas reversas (padrão 3), 1 para extrair frases do contexto, mais 1 de embeddings. Com 30 casos: **180 chamadas de chat + 30 de embeddings**, algo entre **US$ 0,05 e 0,12**.

O comando estima e pede confirmação antes de começar.

Versão barata para um primeiro teste:

```bash
python main.py avaliar --metricas ragas --limite 5
```

Com G-Eval:

```bash
python main.py avaliar --metricas ragas geval --criterios fundamentacao relevancia
```

**Sobre o custo do G-Eval:** se a OpenRouter devolver `logprobs` para o modelo escolhido, é 1 chamada por caso por critério. Se não devolver, o código cai no fallback por amostragem — que é o que o próprio artigo faz com o GPT-4 — e passa a custar `GEVAL_AMOSTRAS` chamadas (padrão 8). O relatório mostra qual método foi usado no campo `geval_metodo`. Se aparecer "amostragem" e você quiser economizar, reduza `GEVAL_AMOSTRAS` no `.env`.

---

## 7. Análise (sem custo, lê o JSON salvo)

```bash
python main.py analisar
python main.py analisar --divergencias
python main.py analisar --tudo          # exporta CSV + Markdown
```

Reimprimir a análise **não custa nada**. Vale rodar várias vezes com `--dimensao` e `--limiar` diferentes.

```bash
python main.py analisar --dimensao relevancia --limiar 0.7
```

O `--limiar` controla onde a nota contínua vira veredito binário. Vale testar: um juiz pode parecer mal calibrado a 0,5 e bem calibrado a 0,7, e isso em si é informação sobre a escala dele.

---

## 8. Sondagem de vieses

```bash
python main.py vieses                    # leniência e verbosidade: grátis
python main.py vieses --posicao --n-pares 5   # posição: 10 chamadas
```

**Leniência e verbosidade** são calculadas sobre a avaliação já salva — custo zero.

**Viés de posição** apresenta cada par nas duas ordens e verifica se o juiz mantém a escolha. Custa 2 chamadas por par. Um juiz sem viés tem consistência próxima de 100%; abaixo de 70% o relatório recomenda não usar comparação par a par com aquele modelo.

---

## 9. Concordância entre anotadores (sem custo)

Se você anotar seus próprios casos com duas pessoas:

```bash
python main.py concordancia data/anotador_a.json data/anotador_b.json
```

**Este é o passo que a maioria pula e não deveria.** A concordância entre humanos é o **teto** do que qualquer juiz automático pode atingir. Sem medi-la, não há como saber se um π de 0,6 é ruim ou é o máximo possível naquela tarefa.

Referências dos artigos: o RAGAS reporta ~95% de concordância entre seus dois anotadores em fidelidade e relevância de contexto, e ~90% em relevância da resposta.

---

## 10. Resumo de custos

Com `openai/gpt-4o-mini`, em setembro de 2026. Confira os preços atuais em https://openrouter.ai/models.

| Comando | Chamadas | Custo aprox. |
|---|---|---|
| `testes_offline.py` / `testes_mock.py` | 0 | **US$ 0** |
| `validar` / `inspecionar` / `lexical` | 0 | **US$ 0** |
| `ppi` / `analisar` / `concordancia` | 0 | **US$ 0** |
| `vieses` (sem `--posicao`) | 0 | **US$ 0** |
| `avaliar --metricas ragas --limite 5` | ~35 | ~US$ 0,01 |
| `avaliar --metricas ragas` (30 casos) | ~210 | **US$ 0,05 – 0,12** |
| `avaliar --metricas ragas geval` | ~270 (com logprobs) | **US$ 0,08 – 0,18** |
| `vieses --posicao --n-pares 5` | 10 | <US$ 0,01 |

Reproduzir tudo, uma vez, fica abaixo de **US$ 0,30**.

---

## 11. Determinismo

**Reproduz exatamente:**

- Todas as métricas lexicais
- Scott's π, Cohen's κ, Spearman, Kendall τ, proporção de empates
- PPI (semente 42 na divisão rotulado/não-rotulado)
- Carregamento, validação e divisão do conjunto
- Toda a análise a partir de um `avaliacao.json` salvo

**Varia entre execuções:**

- RAGAS e G-Eval, por dependerem de geração — mesmo com `temperature=0`, APIs não garantem reprodutibilidade bit-a-bit
- O fallback por amostragem do G-Eval usa `temperature=1` **por definição** (é o método do artigo), então varia mais
- Vereditos do RAGAS em casos limítrofes

Para um relatório, rode as métricas com LLM **3 vezes** e reporte média e desvio.

---

## 12. Problemas comuns

**`OPENROUTER_API_KEY não encontrada`**
O `.env` precisa estar ao lado de `main.py`. Os comandos das seções 2 a 5 e 7 rodam sem chave.

**`Resultados não encontrados`**
`analisar`, `ppi` e `vieses` leem `resultados/avaliacao.json`. Rode `python main.py lexical` (grátis) ou `avaliar` antes.

**`geval_metodo` aparece como "amostragem" e o custo explodiu**
O modelo escolhido não devolve `logprobs` pela OpenRouter. Reduza `GEVAL_AMOSTRAS` no `.env` ou troque de modelo.

**Faithfulness sempre 0**
Provável falha no parsing dos vereditos. Inspecione o JSON salvo: se `n_afirmacoes` > 0 e `n_sustentadas` = 0 em todos os casos, o modelo não está seguindo o formato `VEREDITO N: Sim/Não`. O código conta veredito ilegível como "não sustentado", que é conservador de propósito.

**Scott's π negativo**
Não é necessariamente bug — significa concordância pior que o acaso. Veja a seção 4: para o `contains` na dimensão fundamentação, isso é o resultado correto e esperado.

**Todos os intervalos PPI cobrem tudo**
Conjunto pequeno demais. O ARES usa ~150 pontos anotados; com 30, os intervalos ficam largos. O `validar` avisa quando há menos de 20.

**`SyntaxError` com `|` em anotações de tipo**
Python anterior ao 3.10.

---

## 13. Usar seus próprios dados

1. Exporte as saídas do seu RAG para `data/casos.json` no formato documentado no README.
2. Anote uma amostra. **Com duas pessoas, se possível** — e meça a concordância entre elas antes de tratar a anotação como referência.
3. Rode `python main.py validar`. Ele avisa sobre desbalanceamento e tamanho insuficiente.
4. Comece pelo grátis: `python main.py lexical` já dá o baseline contra o qual o juiz pago terá que se justificar.

**Inclua perguntas sem resposta possível** (gabarito vazio). Elas testam se o sistema sabe se abster — e são os casos em que as métricas de acerto mais enganam.

---

## 14. Checklist de reprodução

- [ ] Python 3.10+ confirmado
- [ ] `venv` criado e ativado; dependências instaladas
- [ ] `python testes_offline.py` → 96 verificações passam
- [ ] `python testes_mock.py` → 65 verificações passam
- [ ] `python main.py validar` → 30 casos, 4 sem resposta possível
- [ ] `python main.py lexical` → `contains` com Scott π negativo em fundamentação
- [ ] `python main.py ppi` → "só juiz" não cobre o valor verdadeiro; PPI cobre
- [ ] `python main.py avaliar --metricas ragas --limite 5` → primeiro teste pago
- [ ] `python main.py analisar --tudo` → CSV e Markdown gerados
- [ ] `python main.py analisar --divergencias` → casos lidos à mão
- [ ] Números anotados **com o nome e a data do modelo juiz usado**
- [ ] `data/PROCEDENCIA.md` lido antes de reportar qualquer número
