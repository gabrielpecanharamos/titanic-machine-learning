# titanic-machine-learning

Este repositório contém o código e a documentação do projeto desenvolvido para o desafio do Titanic no Kaggle. 

O objetivo foi estruturar um fluxo contínuo de dados e modelagem preditiva utilizando a biblioteca Scikit-Learn.

## Tecnologias Utilizadas

- **Linguagem:** Python
- **Manipulação de Dados:** Pandas, NumPy
- **Machine Learning & Pipeline:** Scikit-Learn

## Estrutura do Pipeline

O tratamento dos dados e o treinamento do modelo foram centralizados em um pipeline para automatizar e padronizar todas as etapas:

1. Variáveis Numéricas (`Age`, `Fare`, `SibSp`, `Parch`, `Pclass`):
   - Imputação de valores ausentes pela mediana.
   - Padronização das variáveis com `StandardScaler`.

2. Variáveis Categóricas (`Sex`, `Embarked`):
   - Imputação de valores ausentes pelo valor mais frequente.
   - Codificação categórica via `OneHotEncoder(handle_unknown='ignore')`.

3. Modelo Preditivo:
   - `LogisticRegression` para a classificação binária de sobrevivência.

#### Fonte dos Dados
  
Os arquivos de dados (`.csv`) não estão versionados neste repositório, para rodar localmente:

1. Baixe os arquivos `train.csv` e `test.csv` diretamente na página oficial da [Competição do Titanic no Kaggle](https://www.kaggle.com/c/titanic/data).
2. Salve os arquivos dentro de uma pasta chamada `data/` na raiz do projeto.

#### Desempenho do Modelo

- Acurácia na Validação Local (80/20): 81,01%
- Pontuação Oficial no Kaggle: 0,76315
