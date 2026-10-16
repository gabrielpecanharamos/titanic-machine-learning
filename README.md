# titanic-machine-learning

### Pipeline de Machine Learning e Classificação

Este repositório contém o código e a documentação do projeto desenvolvido para o desafio do Titanic no Kaggle. O objetivo foi estruturar um fluxo contínuo de dados e modelagem preditiva utilizando a biblioteca Scikit-Learn.

#### Desempenho do Modelo

- Acurácia na Validação Local (80/20): 81,01%
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
