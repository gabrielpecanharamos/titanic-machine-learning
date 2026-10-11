import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df_treino = pd.read_csv('train.csv')
df_teste = pd.read_csv('test.csv')

print(df_treino.shape)

print(df_treino.columns)

print(df_treino.dtypes)

print(df_treino.head())

print(df_treino.isnull().sum())

porcentagem_nulos = (df_treino.isnull().sum() / len(df_treino)) * 100
print(porcentagem_nulos)

print(df_treino.duplicated().sum())

# Divisão entre X e y, removendo a coluna a ser predita
X = df_treino.drop(columns=['Survived'])
y = df_treino['Survived']

# Seleção das colunas por categoria, mantendo apenas as que pretendo utilizar
# Cabin foi removida devido ao alto percentual de valores nulos (77%)
colunas_numericas = ['Age', 'Fare', 'SibSp', 'Parch', 'Pclass']
colunas_categoricas = ['Sex', 'Embarked']

# Criação do pipeline e do pré-processamento
# Preenchimento dos nulos numéricos com a mediana e aplicação de escalonamento
pipeline_numerico = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='median')),

    ('scaler', StandardScaler())])

# Preenchimento dos nulos categóricos pelo valor mais frequente e aplicação de One-Hot Encoding
pipeline_categorico = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Agrupamento dos pipelines de pré processamento
pre_processamento = ColumnTransformer(transformers=[

    ('num', pipeline_numerico, colunas_numericas),

    ('cat', pipeline_categorico, colunas_categoricas)])

# Divisão dos dados em treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

# Definição do modelo
pipeline_final = Pipeline(steps=[
    ('pre_processamento', pre_processamento),
    ('modelo', LogisticRegression())])

pipeline_final.fit(X_treino, y_treino)

predito = pipeline_final.predict(X_teste)
print(f"Acurácia: {accuracy_score(y_teste, predito):.2%}\n")

# Seleção das colunas de interesse no conjunto de teste para submissão
X_submissao = df_teste[colunas_numericas + colunas_categoricas]

# Previsão no conjunto de teste
previsao_final = pipeline_final.predict(X_submissao)

# Criação do DataFrame final no formato exigido para submissão
submissao = pd.DataFrame({
    'PassengerId': df_teste['PassengerId'],
    'Survived': previsao_final})

print(submissao.head())
print("\nTotal de linhas:", len(submissao))
print("\nDistribuição das previsões:")
print(submissao['Survived'].value_counts())

submissao.to_csv('submissao_titanic.csv', index=False)
