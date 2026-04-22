import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as pt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sklearn.preprocessing import LabelEncoder



data = pd.read_csv('bank_churn_limpio.csv')

def Categorias(data: pd.DataFrame):

    sb.barplot(data, x='country', y='churn')
    pt.savefig('graficas/barplotCountry.pdf', bbox_inches='tight')
    pt.close()
    print("Gráfico con datos reales (0 y 1) generado con éxito.")
    
    sb.barplot(data, x='gender', y='churn')
    pt.savefig('graficas/barplotGender.pdf', bbox_inches='tight')
    pt.close()
    print("Gráficos churn de genero y pais generados con éxito.")

def calcularVIF(data: pd.DataFrame):
    columnasVIF = ['age', 'credit_score', 'tenure', 'limite_credito', 'products_number', 'estimated_salary', 'NewBalance']
    X = data[columnasVIF].copy()
    X = add_constant(X)
    dataVIF = pd.DataFrame()
    dataVIF["Variable"] = X.columns
    dataVIF["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return dataVIF[dataVIF["Variable"] != 'const'].sort_values(by="VIF", ascending=False)

def MatrizRealcionMultivariable(data: pd.DataFrame):
    data = data.drop(columns='balance')
    columnas_numericas = data.select_dtypes(include=['number'])
    corr_matrix = columnas_numericas.corr()
    pt.figure(figsize=(10, 6))
    sb.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    pt.savefig('graficas/heatmap.pdf', bbox_inches='tight')
    pt.close()
    print("Heatmap de correlaciones generado con éxito.")


def pairplot(data: pd.DataFrame):
    pt.figure(figsize=(10, 6))
    columnasInteres = ['age', 'credit_score', 'NewBalance', 'estimated_salary', 'products_number','tenure', 'churn']
    sb.pairplot(data[columnasInteres], hue='churn', diag_kind='kde', palette='husl', plot_kws={'alpha': 0.65})
    pt.savefig('graficas/pairplot.pdf', bbox_inches='tight')
    pt.close()
    print("Grafico de relacion entre las variables y el churn generado con exito")
    
def analizarLealtad(data):
    pt.figure(figsize=(10, 5))
    sb.kdeplot(data=data[data['churn'] == 0]['tenure'], label='Se quedan', fill=True)
    sb.kdeplot(data=data[data['churn'] == 1]['tenure'], label='Se van', fill=True)
    pt.savefig('graficas/lealtad_tenure.pdf')
    pt.close()
    print("Grafico de relacion entre tenure y churn")

def Churn_Product(data: pd.DataFrame):
    
    sb.countplot(data=data, x='products_number', hue='churn')
    pt.savefig('graficas/productos_vs_churn.pdf')
    sb.catplot(data=data, x='products_number', y='churn', col='country', kind='bar')
    pt.savefig('graficas/productos_pais.pdf')
    pt.close()
    
def edadChurn(data):
    pt.figure(figsize=(10, 6))
    sb.kdeplot(data=data[data['churn'] == 0]['age'], label='Se quedan (Churn 0)', fill=True, color='blue', alpha=0.4)
    sb.kdeplot(data=data[data['churn'] == 1]['age'], label='Se van (Churn 1)', fill=True, color='red', alpha=0.4)
    pt.grid(axis='y', alpha=0.3)
    pt.savefig('graficas/edad_churn.pdf')
    pt.close()
    
def analizarRiquezaEdad(data):

    sb.jointplot(data=data, x='age', y='estimated_salary', hue='churn', kind='kde', palette='RdBu_r', fill=True, alpha=0.5)
    pt.savefig('graficas/riqueza_edad_fuga.pdf')
    pt.close()
    
def DataModelado(data: pd.DataFrame):
    dataCopy = data.copy()
    le = LabelEncoder()
    dataCopy['gender'] = le.fit_transform(dataCopy['gender'])
    dataCopy = pd.get_dummies(dataCopy, columns=['country'], prefix='country')
    dataCopy = dataCopy.drop(columns=['balance'])
    return dataCopy



"""analizarRiquezaEdad(data)
vif_resultados = calcularVIF(data)
print(vif_resultados)
Categorias(data)
MatrizRealcionMultivariable(data)
pairplot(data)
analizarLealtad(data)
Churn_Product(data)
edadChurn(data)"""
DataModelado = DataModelado(data)

DataModelado.to_csv('bank_churn_modelado.csv', index = False)