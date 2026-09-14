# APS1: Análise Exploratória de Dados (EDA)

**Título do projeto:** Análise Exploratória de Dados — Adult Income Census
**Membros:** Hector Mathias e Giovanni Rodrigues
**Data de entrega:** 14/09/2026
**Dataset:** [Adult Income Census](https://www.kaggle.com/datasets/anaghakp/adult-income-census) (Kaggle)

---

## Objetivo do projeto

Esta etapa do projeto tem como objetivo realizar uma **análise exploratória abrangente** do dataset `Adult Income Census`, entendendo sua estrutura, qualidade, distribuições e relações entre variáveis (numéricas e categóricas), além de identificar desafios como valores ausentes e desbalanceamento de classes. As conclusões obtidas aqui fundamentam as decisões de pré-processamento que serão reaplicadas nas fases seguintes de modelagem.

A variável alvo (`target`) do problema é `income`, indicando se a renda anual de um indivíduo é `<=50K` ou `>50K` dólares — um problema de **classificação binária**.

## Como navegar nesta documentação

A análise está organizada em cinco etapas, seguindo a mesma estrutura do notebook (`ProjetoML_EDA_final_4.ipynb`):

| Seção | Conteúdo |
|---|---|
| [1. Inspeção Inicial](01-inspecao-inicial.md) | Descrição das features, dimensões do dataset, tipos de dados, valores ausentes e desbalanceamento de classes |
| [2. Análise Univariada](02-analise-univariada.md) | Estatísticas descritivas, histogramas, boxplots, violinos e distribuições categóricas |
| [3. Análise Bivariada e Multivariada](03-analise-bivariada-multivariada.md) | Correlações, scatter plots, relações categórica × `income` e numérica × categórica |
| [4. Pré-processamento](04-preprocessamento.md) | Split treino/teste, tratamento de ausentes e outliers, encoding, padronização, PCA e pipeline final |
| [5. Conclusões](05-conclusoes.md) | Síntese dos principais achados e recomendações para a modelagem |
| [6. Referências](06-referencias.md) | Fontes e materiais de apoio utilizados no projeto |

Também disponibilizamos uma página de [**Rubrica × Entrega**](rubrica.md), mapeando cada requisito solicitado ao ponto correspondente desta documentação.

## Visão geral do dataset

O dataset é composto por **31.947 instâncias**, **11 features** e uma variável alvo (`income`):

| Feature          | Descrição                                                                                                                  | Tipo de dado |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------- | ------------ |
| `age`            | Idade do indivíduo, em anos.                                                                                               | Numérico     |
| `workclass`      | Tipo de emprego ou classe de trabalho do indivíduo (e.g., Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, etc.).     | Categórico   |
| `fnlwgt`         | Peso final. Estima quantas pessoas da população são representadas por cada observação do dataset.                          | Numérico     |
| `education`      | Maior nível de escolaridade alcançado pelo indivíduo (e.g., Bachelors, HS-grad, 11th, Masters, etc.).                      | Categórico   |
| `education.num`  | Representação numérica do nível de escolaridade (e.g., 13 para Bachelors, 9 para HS-grad, etc.).                           | Numérico     |
| `marital.status` | Estado civil do indivíduo (e.g., Married-civ-spouse, Divorced, Never-married, Separated, etc.).                            | Categórico   |
| `occupation`     | Ocupação profissional do indivíduo (e.g., Tech-support, Craft-repair, Other-service, Sales, etc.).                         | Categórico   |
| `relationship`   | Relação do indivíduo dentro do núcleo familiar (e.g., Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried). | Categórico   |
| `race`           | Raça ou etnia do indivíduo (e.g., White, Black, Asian-Pac-Islander, Amer-Indian-Eskimo, Other).                            | Categórico   |
| `sex`            | Sexo do indivíduo (e.g., Male, Female).                                                                                    | Categórico   |
| `native.country` | País de origem do indivíduo (e.g., United-States, Cambodia, England, Puerto-Rico, etc.).                                   | Categórico   |
| `income` *(target)* | Faixa de renda anual do indivíduo: `<=50K` ou `>50K`.                                                                    | Categórico (binário) |

## Bibliotecas utilizadas

O projeto foi desenvolvido inteiramente em **Python**, utilizando `Pandas`, `NumPy`, `Matplotlib`, `Seaborn` e `Scikit-learn`, conforme exigido.

!!! info "Sobre esta documentação"
    Este site foi gerado a partir do notebook `ProjetoML_EDA_final_4.ipynb`, reorganizando o conteúdo, os gráficos e as justificativas produzidos na análise em um formato de leitura mais estruturado, sem alterar as decisões técnicas tomadas pelo grupo.
