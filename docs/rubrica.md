# Rubrica × Entrega

Esta página mapeia cada requisito da rubrica de EDA ao ponto correspondente desta documentação, facilitando a conferência de que todos os itens foram atendidos.

!!! success "Correções aplicadas nesta versão"
    - Agrupamento de categorias raras de `native.country` em `"Other"` agora implementado no pipeline (antes só estava no texto).
    - `sex` agora usa `OneHotEncoder(drop="if_binary")` no pipeline, gerando de fato uma coluna binária 0/1 (antes ia para o one-hot genérico junto das demais categóricas).
    - `native.country` agora é imputada pela moda no pipeline, como já previsto no texto da Seção 4.1 (antes recebia a categoria `"Missing"` como as demais).
    - Data de entrega e título do projeto adicionados ao cabeçalho.
    - Seção de referências bibliográficas adicionada ([Seção 6](06-referencias.md)).

## Carregamento e inspeção inicial dos dados

| Requisito | Onde está |
|---|---|
| Explicar cada feature do dataset | [Início — Visão geral do dataset](index.md#visao-geral-do-dataset) |
| Verificar número de instâncias e features | [1. Inspeção Inicial — Carregamento](01-inspecao-inicial.md#carregamento-dos-dados) |
| Identificar tipos de dados (numéricos, categóricos) | [1. Inspeção Inicial — Separação numéricas/categóricas](01-inspecao-inicial.md#separacao-entre-variaveis-numericas-e-categoricas) |
| Detectar valores ausentes e inconsistências | [1. Inspeção Inicial — Valores ausentes e inconsistências](01-inspecao-inicial.md#valores-ausentes-e-inconsistencias) |
| Verificar desbalanceamento de classes | [1. Inspeção Inicial — Desbalanceamento de classes](01-inspecao-inicial.md#desbalanceamento-de-classes) |
| Separar dados em treino e teste | [4. Pré-processamento — Separação em treino e teste](04-preprocessamento.md#separacao-em-treino-e-teste) |

## Análise univariada

| Requisito | Onde está |
|---|---|
| Estatísticas descritivas de todas as variáveis numéricas | [2. Análise Univariada — Estatísticas descritivas](02-analise-univariada.md#estatisticas-descritivas-das-variaveis-numericas) |
| Histogramas, boxplots e/ou violinos (numéricas, máx. 3) | [2. Análise Univariada — Histogramas, boxplots e violinos](02-analise-univariada.md#histogramas-boxplots-e-violinos-variaveis-numericas) |
| Frequências e gráficos de barras (categóricas, máx. 3) | [2. Análise Univariada — Frequências e gráficos de barras](02-analise-univariada.md#frequencias-e-graficos-de-barras-variaveis-categoricas) |

## Análise bivariada e multivariada

| Requisito | Onde está |
|---|---|
| Correlações entre numéricas (matriz + scatter, máx. 3) | [3. Bivariada/Multivariada — Correlações](03-analise-bivariada-multivariada.md#correlacoes-entre-variaveis-numericas) |
| Categóricas × `income` (barras/tabelas de contingência, máx. 3) | [3. Bivariada/Multivariada — Categóricas × income](03-analise-bivariada-multivariada.md#relacoes-entre-variaveis-categoricas-e-o-target-income) |
| Numéricas × categóricas (boxplots conjuntos, máx. 3) | [3. Bivariada/Multivariada — Numéricas × categóricas](03-analise-bivariada-multivariada.md#relacoes-entre-variaveis-numericas-e-categoricas) |

## Pré-processamento para modelagem

| Requisito | Onde está |
|---|---|
| Estratégia para valores ausentes, justificada | [4. Pré-processamento — 4.1 Valores ausentes](04-preprocessamento.md#41-valores-ausentes) |
| Estratégia para outliers, justificada | [4. Pré-processamento — 4.2 Outliers](04-preprocessamento.md#42-outliers) |
| Estratégia de encoding, justificada | [4. Pré-processamento — 4.3 Encoding](04-preprocessamento.md#43-encoding-de-variaveis-categoricas) |
| Estratégia de normalização/padronização, justificada | [4. Pré-processamento — 4.4 Normalização/padronização](04-preprocessamento.md#44-normalizacao-padronizacao) |
| Redução de dimensionalidade (PCA) nas numéricas | [4. Pré-processamento — 4.5 e 4.6 PCA](04-preprocessamento.md#45-e-46-reducao-de-dimensionalidade-pca) |
| Análise dos resultados da redução de dimensionalidade | [4. Pré-processamento — Loadings e separabilidade](04-preprocessamento.md#45-e-46-reducao-de-dimensionalidade-pca) |
| Pipeline com `Pipeline` e `ColumnTransformer` | [4. Pré-processamento — 4.7 Pipeline final](04-preprocessamento.md#47-pipeline-de-pre-processamento) |

## Documentação e apresentação

| Requisito | Onde está |
|---|---|
| Requisitos e figuras justificados objetivamente | Todas as seções (2–4), com justificativa textual após cada gráfico/tabela |
| Visualizações com títulos, rótulos e legendas adequados | Figuras nas seções 2 e 3, geradas com Matplotlib/Seaborn |
| Relatório resumindo achados e estratégias | [5. Conclusões do EDA](05-conclusoes.md) |
| Código organizado e comentado | Blocos de código replicados de `ProjetoML_EDA_final_4.ipynb`, comentados por etapa |
| Escolhas de pré-processamento fundamentadas nos achados da EDA | [4. Pré-processamento](04-preprocessamento.md) — cada subseção referencia os achados das seções 1–3 |
| Python com Pandas, NumPy, Matplotlib, Seaborn e Scikit-learn | Usado em todo o notebook — ver trechos de código em todas as seções |

## Requisitos gerais de entrega

| Requisito | Onde está |
|---|---|
| Identificação dos membros do grupo | [Início](index.md) — cabeçalho |
| Título do projeto | [Início](index.md) — cabeçalho |
| Data da entrega | [Início](index.md) — cabeçalho (14/09/2026) |
| Referências bibliográficas | [6. Referências](06-referencias.md) |
| Documentação aberta (bônus `+`) | Este próprio site MkDocs, publicável via `mkdocs gh-deploy` |
