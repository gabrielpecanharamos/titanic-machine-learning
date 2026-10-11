# Titanic-machine-learning

Language / Idioma: [Português](#português) | [English](#english)

---

## Português

### Titanic - Pipeline de Machine Learning e Classificação

Este repositório contém o código e a documentação do projeto desenvolvido para o desafio do Titanic no Kaggle. O objetivo foi estruturar um fluxo contínuo de dados e modelagem preditiva utilizando a biblioteca Scikit-Learn.

#### Desempenho do Modelo

- Acurácia na Validação Local (80/20): 81,0%
- Pontuação Oficial no Kaggle (Public Score): 0,76315

#### Estrutura do Pipeline

O tratamento dos dados e o treinamento do modelo foram centralizados em um pipeline para evitar o vazamento de dados (*data leakage*):

1. Variáveis Numéricas (`Age`, `Fare`, `SibSp`, `Parch`, `Pclass`):
   - Imputação de valores ausentes pela mediana.
   - Padronização das variáveis com `StandardScaler`.

2. Variáveis Categóricas (`Sex`, `Embarked`):
   - Imputação de valores ausentes pelo valor mais frequente.
   - Codificação categórica via `OneHotEncoder(handle_unknown='ignore')`.

3. Modelo Preditivo:
   - `LogisticRegression` para a classificação binária de sobrevivência.

---

## English

### Titanic - Machine Learning Pipeline & Classification

This repository contains the code and documentation for the Kaggle Titanic challenge. The primary goal was to structure a data and predictive modeling workflow using the Scikit-Learn library.

#### Model Performance

- Local Validation Accuracy (80/20): 81.0%
- Official Kaggle Public Score: 0.76315

#### Pipeline Structure

Data processing and model training were centralized into a single pipeline to prevent data leakage:

1. Numerical Features (`Age`, `Fare`, `SibSp`, `Parch`, `Pclass`):
   - Missing value imputation using the median.
   - Feature scaling with `StandardScaler`.

2. Categorical Features (`Sex`, `Embarked`):
   - Missing value imputation using the most frequent value.
   - Categorical encoding with `OneHotEncoder(handle_unknown='ignore')`.
3. Predictive Model:
   - `LogisticRegression` for binary survival classification.
