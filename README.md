# Documentação EDA — Adult Income Census (MkDocs)

Site de documentação gerado com [MkDocs](https://www.mkdocs.org/) + tema [Material](https://squidfunk.github.io/mkdocs-material/), a partir do notebook `ProjetoML_EDA_final_4.ipynb`.

## Como rodar localmente

```bash
pip install mkdocs mkdocs-material
mkdocs serve
```

Acesse em `http://127.0.0.1:8000`.

## Como gerar o site estático (HTML)

```bash
mkdocs build
```

Os arquivos finais ficam em `site/`, prontos para hospedar em GitHub Pages, Netlify, Vercel, etc.

## Publicar no GitHub Pages

```bash
mkdocs gh-deploy
```

## Estrutura do projeto

```
.
├── mkdocs.yml              # configuração do site (nav, tema, extensões)
└── docs/
    ├── index.md             # página inicial
    ├── 01-inspecao-inicial.md
    ├── 02-analise-univariada.md
    ├── 03-analise-bivariada-multivariada.md
    ├── 04-preprocessamento.md
    ├── 05-conclusoes.md
    ├── rubrica.md            # mapeamento rubrica x entrega
    └── assets/images/        # gráficos exportados do notebook
```
