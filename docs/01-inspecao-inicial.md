# 1. Inspeção Inicial dos Dados

O dataset é composto por **11 features** e uma **target** (`income`), com **31.947 instâncias**. Para a descrição completa de cada feature e seu tipo de dado, veja a tabela na [página inicial](index.md#visao-geral-do-dataset).

## Carregamento dos dados

```python
import pandas as pd
import os
import re

df = pd.read_csv(os.path.join(path, "adult income1.csv"), encoding='latin-1')

display(df.shape)
display(df.dtypes)
df.head()
```

**Saída:**

```text
(31947, 12)

age                int64
workclass         object
fnlwgt             int64
education         object
education.num      int64
marital.status     object
occupation         object
relationship       object
race               object
sex                object
native.country     object
income             object
dtype: object
```

## Separação entre variáveis numéricas e categóricas

```python
NUM = [
    "age",
    "fnlwgt",
    "education.num"
]

CAT = [
    "workclass",
    "education",
    "marital.status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native.country"
]
```

## Valores ausentes e inconsistências

A inspeção das categorias de cada variável (`value_counts`) e a contagem de `"?"` no dataset revelam inconsistências que não aparecem em uma checagem padrão de nulos (`isna()`), já que os valores ausentes **não foram codificados como `NaN`**, e sim como texto:

```python
for c in CAT + ["income"]:
    print(df[c].value_counts(dropna=False), "\n")

display(df.eq("?").sum())
df.eq("?").any(axis=1).sum()
```

| Coluna | Ausentes (`"?"`) |
|---|---|
| `workclass` | 1.778 |
| `occupation` | 1.785 *(inclui inconsistência extra, ver abaixo)* |
| `native.country` | 25 |

Duas inconsistências foram identificadas:

- Os **valores ausentes** foram representados como `"?"` e estão presentes em **1.778 linhas** de `workclass`, **1.785 linhas** de `occupation` e **25 linhas** de `native.country`.
- Na feature `occupation`, existem **1.462 linhas** preenchidas literalmente com o texto `"occupation"` — provavelmente um valor padrão (*default*) do formulário/sistema de origem que acabou persistido em várias instâncias, sem relevância semântica para a análise. Essas linhas também são tratadas como ausentes.

## Desbalanceamento de classes

```python
contagem = df["income"].value_counts()
percentual = df["income"].value_counts(normalize=True).mul(100)

display(
    pd.DataFrame({
        "Quantidade": contagem,
        "Percentual (%)": percentual.round(2)
    })
)
```

| `income` | Quantidade | Percentual (%) |
|---|---|---|
| `<=50K` | 24.264 | 75,95 |
| `>50K` | 7.683 | 24,05 |

Existe um **desbalanceamento de classes moderado**: a classe majoritária (`<=50K`) possui cerca de **3,16 vezes mais registros** que a minoritária (`>50K`). Apesar de moderado, esse desbalanceamento pode afetar o treinamento, fazendo com que o modelo aprenda menos sobre como classificar indivíduos com renda maior que 50 mil dólares — o que reforça a necessidade de métricas de avaliação além da acurácia (ex: F1-score, AUC) nas fases seguintes.

!!! note "Sobre a separação treino/teste"
    A separação em treino e teste foi realizada **após** a inspeção inicial e a análise exploratória, imediatamente antes das etapas de pré-processamento que exigem `fit` (imputação, encoding, padronização). Essa decisão — e sua justificativa — está detalhada na [seção 4](04-preprocessamento.md#separacao-em-treino-e-teste).
