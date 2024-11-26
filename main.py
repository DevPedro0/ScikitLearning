import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


# Configuração de exibição
pd.set_option('display.max_columns', None)

# Carregando os dados
data = pd.read_csv('train.csv')

# Função para obter dados numéricos
def ObterDadosNumericos(file):
    file_copy = file.copy()
    file_copy[['LotFrontage', 'MasVnrArea', 'GarageYrBlt']] = file_copy[
        ['LotFrontage', 'MasVnrArea', 'GarageYrBlt']
    ].fillna(0)

    # Selecionar colunas numéricas
    lista_numerica = file_copy.select_dtypes(include=[np.number])
    return lista_numerica.dropna(axis=1)

file_numerico = ObterDadosNumericos(data)

# Função para ajustar nomes das colunas
def TratarColunasText(var_tratar_columns, file_principal):
    var_tratar_columns.columns = file_principal.select_dtypes(include=[np.number]).columns
    return var_tratar_columns

file_numerico_ajustado = TratarColunasText(file_numerico, data)

# Calculando a correlação com SalePrice
corr = file_numerico_ajustado.corr()
sale_corr = corr[['SalePrice']][corr['SalePrice'] > 0.1].sort_values('SalePrice', ascending=False)

# Visualizando correlação com SalePrice
fig = plt.figure(figsize=(10, 5))
plt.bar(sale_corr.index, sale_corr['SalePrice'], color='blue')
plt.xticks(rotation=90)
plt.title('Correlação com SalePrice')
plt.show()

# Função para obter dados categóricos
def ObterDadosCategoricos(file, colunas_numericas):
    file_categorico = file.drop(columns=colunas_numericas)
    # Remover colunas categóricas irrelevantes
    colunas_irrelevantes = ['Alley', 'Street', 'MasVnrType', 'PoolQC', 'Fence', 'MiscFeature']
    file_categorico = file_categorico.drop(columns=colunas_irrelevantes, errors='ignore')
    return file_categorico

dataframe_categorico = ObterDadosCategoricos(data, file_numerico.columns)

# Convertendo dados categóricos para numéricos com OneHotEncoder
one_hot = OneHotEncoder(sparse_output=False)
dataframe_codificado = pd.DataFrame(one_hot.fit_transform(dataframe_categorico))
dataframe_codificado.columns = one_hot.get_feature_names_out(dataframe_categorico.columns)

# Criando a correlação com dados categóricos codificados
corr_dados_categoricos = dataframe_codificado.corr()

# Normalizando os dados com StandardScaler
file_numeric_ajust_train = file_numerico_ajustado.drop(columns=['Id'], errors='ignore')

class TrainDataNumeric:
    def __init__(self, file):
        self.file = pd.DataFrame(file)
        self.X_train_numeric = self.file.drop(columns=['SalePrice'])
        self.y_train_numeric = self.file['SalePrice']
        self.standard = StandardScaler()
        self.train = None
    
    def train_numeric(self):
        # Normalizando os dados
        self.train = pd.DataFrame(
            self.standard.fit_transform(self.X_train_numeric),
            columns=self.X_train_numeric.columns
        )
        return self.train
    
    def grafico_transform(self):
        # Criar gráfico com os dados normalizados
        columns_sum = self.train.sum(axis=0)
        plot_data = pd.DataFrame({"Coluna": self.train.columns, "Soma": columns_sum.values})
        sns.barplot(x='Coluna', y='Soma', data=plot_data, color='red')
        plt.xticks(rotation=90)
        plt.title("Soma das colunas normalizadas")
        plt.show()

# Instanciando e treinando
train_data_numeric = TrainDataNumeric(file_numeric_ajust_train)
train_data_numeric.train_numeric()
train_data_numeric.grafico_transform()



# Testando e Treinando Os Dados

X_train, X_test, y_train, y_test = train_test_split(train_data_numeric.X_train_numeric, train_data_numeric.y_train_numeric, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Fazendo previsões
y_pred = model.predict(X_test)

# Avaliação do modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(1300424286.328392)

print(f"MSE: {mse}")
print(f"R²: {r2}")
print(data['SalePrice'].describe())
print(rmse)


plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2)
plt.xlabel('Valores Reais (y_test)')
plt.ylabel('Valores Preditos (y_pred)')
plt.title('Valores Reais vs. Preditos')
plt.show()

# Calculando os resíduos
residuos = y_test - y_pred

plt.figure(figsize=(8, 6))
sns.histplot(residuos, kde=True, bins=30, color='purple')
plt.axvline(0, color='red', linestyle='--', linewidth=2)
plt.title('Distribuição dos Resíduos')
plt.xlabel('Resíduos')
plt.ylabel('Frequência')
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuos, alpha=0.7, color='green')
plt.axhline(0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Valores Preditos (y_pred)')
plt.ylabel('Resíduos (y_test - y_pred)')
plt.title('Resíduos vs. Valores Preditos')
plt.show()



