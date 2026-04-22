import pandas as pd
import matplotlib.pyplot as pt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib as jl

def ModeloCreacion():
    df = pd.read_csv('bank_churn_modelado.csv')
    x = df.drop(columns=['churn', 'customer_id', 'email', 'numero_conta', 'data_nascimento', 'nome', 'cidade'])
    y = df['churn']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10, stratify=y)
    modelo = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    modelo.fit(x_train, y_train)
    y_probs = modelo.predict_proba(x_test)[:, 1]
    y_pred_custom = (y_probs >= 0.3).astype(int)
    reporte = classification_report(y_test, y_pred_custom)
    dict_modelo = {'modelo': modelo,'columnas': x.columns.tolist(),'umbral': 0.3}
    jl.dump(dict_modelo, 'modelo_banco_final.pkl')
    print("\nModelo guardado con éxito como 'modelo_banco_final.pkl'")
    
    return modelo, reporte, x

modelo, reporte, x = ModeloCreacion()

def importanciaVariables():
    
    importancias = pd.Series(modelo.feature_importances_, index=x.columns).sort_values(ascending=False)

    pt.figure(figsize=(10, 6))
    sb.barplot(x=importancias.values, y=importancias.index, palette='viridis')
    pt.savefig('graficas/importancia_variables.pdf', bbox_inches='tight')
    pt.close()

    print("\n--- Importancia de las Variables ---")
    print(importancias)
    
print(reporte)
importanciaVariables()