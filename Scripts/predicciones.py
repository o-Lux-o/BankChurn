import joblib as jl
import pandas as pd

modeloImportado = jl.load('modelo_banco_final.pkl')

modelo = modeloImportado['modelo']
columnas = modeloImportado['columnas']
umbral = modeloImportado['umbral']

def predecir_cliente(datos_cliente: dict):

    dataCliente = pd.DataFrame([datos_cliente])
    dataCliente = dataCliente[columnas]
    prob = modelo.predict_proba(dataCliente)[:, 1][0]
    decision = "Fuga" if prob >= umbral else "Leal"
    
    return {"Probabilidad de fuga": f"{prob * 100:.2f}%", "Recomendación": decision}

nuevo_cliente1 = {
    
    'age': 52,
    'credit_score': 590,
    'tenure': 2,
    'limite_credito': 12000,
    'products_number': 1,
    'estimated_salary': 75000,
    'NewBalance': 0,
    'balance': 0,
    'active_member': 0,
    'gender': 1,         
    'country_France': 0,
    'country_Germany': 1,
    'country_Spain': 0,
    'credit_card': 1
}

nuevo_cliente2 = {
    
    'age': 20,
    'credit_score': 720,
    'tenure': 8,
    'limite_credito': 25000,
    'products_number': 2,
    'estimated_salary': 45000,
    'NewBalance': 15000,
    'balance': 15000,
    'active_member': 1,
    'gender': 1,            
    'country_France': 0,
    'country_Germany': 1,
    'country_Spain': 0,
    'credit_card': 1
}

nuevo_cliente3 = {
    'age': 41,
    'credit_score': 640,
    'tenure': 5,
    'limite_credito': 18000,
    'products_number': 1,
    'estimated_salary': 105000,
    'NewBalance': 8500,
    'balance': 8500,
    'active_member': 0,
    'gender': 1,
    'country_France': 0,
    'country_Germany': 0,
    'country_Spain': 1,
    'credit_card': 0
}

print(predecir_cliente(nuevo_cliente1))
print(predecir_cliente(nuevo_cliente2))
print(predecir_cliente(nuevo_cliente3))