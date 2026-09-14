# 5. Conclusões do EDA

Reunindo as observações levantadas ao longo da análise, chegamos às seguintes conclusões para a etapa de modelagem:

- O dataset apresenta um **desbalanceamento moderado** no target `income` (~3,16x mais registros em `<=50K`), o que deve ser considerado na avaliação dos modelos (ex: métricas além de acurácia, como F1-score ou AUC).
- Os ausentes, representados como `"?"`, se concentram em `workclass`, `occupation` e `native.country`. Optamos por imputar `native.country` pela moda (baixo impacto) e tratar `workclass`/`occupation` com uma categoria explícita `"Missing"`, preservando o possível padrão de não-resposta.
- Nenhuma variável numérica apresentou correlação linear relevante entre si, o que foi confirmado pela distribuição mais equilibrada da variância explicada no PCA — não há redundância forte a eliminar entre `age`, `fnlwgt` e `education.num`.
- `fnlwgt` é um peso amostral, não uma característica individual: seus outliers foram preservados, mas sua assimetria foi tratada via `log1p` antes da padronização.
- `education` e `education.num` são redundantes; mantivemos apenas `education.num` por já representar a hierarquia educacional de forma ordinal correta.
- Variáveis categóricas como `marital.status` e `education.num` mostraram associação mais forte com `income` do que as numéricas isoladamente, sugerindo que a separação entre as classes de renda depende mais da combinação de fatores sociodemográficos do que de qualquer variável numérica isolada.
- O pipeline de pré-processamento (imputação + transformação + encoding + padronização) foi construído com `Pipeline` e `ColumnTransformer`, permitindo reaplicação consistente em treino e teste nas próximas fases do projeto.
- Os dados foram separados em treino (80%) e teste (20%) de forma estratificada por `income`, e o `preprocessor` é ajustado (`fit`) apenas no treino, evitando vazamento de dados (*data leakage*) para o teste.
- A análise dos *loadings* do PCA mostrou que cada componente concentra o peso majoritariamente em uma única variável numérica, reforçando que `age`, `fnlwgt` e `education.num` carregam informações relativamente independentes entre si.

## Próximos passos sugeridos

Com base nesses achados, as fases de modelagem subsequentes devem:

1. Utilizar o `preprocessor` construído na [Seção 4.7](04-preprocessamento.md#47-pipeline-de-pre-processamento) como base para os pipelines de treinamento, evitando reimplementar o tratamento de dados.
2. Priorizar métricas robustas a desbalanceamento (F1-score, AUC-ROC, precision/recall) na avaliação dos modelos, em vez de acurácia isolada.
3. Considerar as variáveis categóricas (`marital.status`, `occupation`, `education.num`) como preditores centrais, já que se mostraram mais discriminativas que as numéricas isoladas.
4. Investigar possíveis técnicas de balanceamento de classes (ex: `class_weight`, oversampling/undersampling) caso os modelos apresentem baixa performance na classe minoritária (`>50K`).
