import pandas as pd
import numpy as np

def generarDatosSinteticos(n: int):

    data = {
        'age': np.random.normal(38, 10, n).astype(int).clip(18, 90),
        'credit_score': np.random.normal(650, 100, n).astype(int).clip(350, 850),
        'tenure': np.random.randint(0, 11, n),
        'limite_credito': np.random.uniform(5000, 30000, n).round(2),
        'products_number': np.random.choice([1, 2, 3, 4], n, p=[0.5, 0.45, 0.04, 0.01]),
        'estimated_salary': np.random.uniform(20000, 150000, n).round(2),
        'active_member': np.random.choice([0, 1], n, p=[0.4, 0.6]),
        'gender': np.random.choice([0, 1], n),
        'credit_card': np.random.choice([0, 1], n, p=[0.3, 0.7]),
    }
    
    dataR = pd.DataFrame(data)
    
    paises = ['France', 'Germany', 'Spain']
    for pais in paises:
        dataR[f'country_{pais}'] = 0
        
    for i in range(n):
        paisElegido = np.random.choice(paises)
        dataR.loc[i, f'country_{paisElegido}'] = 1
        
    dataR['balance'] = np.where(np.random.random(n) > 0.3, 
    np.random.uniform(10000, 200000, n), 0).round(2)
    
    dataR['NewBalance'] = dataR['balance']
    
    return dataR

nuevosClientes = generarDatosSinteticos(5)

nuevosClientes.to_csv('nuevosClientes.csv', index=False)