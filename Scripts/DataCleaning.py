import pandas as pd
import numpy as np
import re 
import seaborn as sb
import matplotlib.pyplot as pt

def DataCleaningF(ruta: str):
    data = pd.read_csv(ruta)
    
    def DuplicatedData(data: pd.DataFrame):
        count1 = data.duplicated().sum()
        print(f"\nHay: {count1} valores duplicados")
        data = data.drop_duplicates(keep='first')
        count = data.duplicated().sum()
        print(f"\nAhora hay: {count}\n")
        return data
    
    def DataNull(data: pd.DataFrame):
        count1 = data.isnull().sum()
        print(f"\n{count1}\n\n")
        data = data.dropna(subset=['customer_id'])
        data['email'] = data['email'].fillna('Sin Registro')
        count1 = data.isnull().sum()
        estimatedSalaryMedian = data['estimated_salary'].median()
        data['estimated_salary'] = data['estimated_salary'].fillna(estimatedSalaryMedian)
        count1 = data.isnull().sum()
        print(count1)
        return data
    
    def Formato(data: pd.DataFrame):
        info = data.info()
        print(info)
        data['customer_id'] = data['customer_id'].astype('Int64')
        data['cidade'] = data['cidade'].str.strip().str.title()
        data['nome'] = data['nome'].str.strip().str.title()
        data['country'] = data['country'].str.strip().str.title()
        data['data_nascimento'] = pd.to_datetime(data['data_nascimento'], format='mixed', dayfirst=True, errors='coerce')
        
        patron = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        valido = data['email'].str.match(patron, na=False)
        
        sinRegistro = (data['email'] == 'Sin Registro')
        
        data.loc[~(valido | sinRegistro), 'email'] = 'Formato Inválido'
        
        info = data.info()
        print(info)
        return data
    
    def Outliers(data: pd.DataFrame, columna: pd.Series):
        
        def calcularLimites():
            Q1 = data[columna].quantile(0.25)
            Q3 = data[columna].quantile(0.75)
            IQR = Q3 - Q1
            return (Q1 - 1.5 * IQR), (Q3 + 1.5 * IQR)
        
        if columna == 'age':
            
            limiteInferior, limiteSuperior = calcularLimites()
            
            limiteInferior = 18
            
            cantidadOutliers = len(data[(data[columna] < limiteInferior) | (data[columna] > limiteSuperior)])
        
            print(f"\nLimpieza 'Age'\n")
            print(f"Cantidad de outliers: {cantidadOutliers}\n")
            
            pt.figure(figsize=(10, 6))
            sb.boxplot(data['age'])
            pt.axhline(limiteInferior, color='red', linestyle='--')
            pt.show()
            
            data = data[data['age'] >= 18].copy()
            
            cantidadOutliers = len(data[(data['age'] < limiteInferior)])

            print(f"\nSe eliminaron los valores extremos en 'age', outliers ahora: {cantidadOutliers}\n")
            
            pt.figure(figsize=(10, 6))
            sb.boxplot(data['age'], showfliers=False)
            pt.axhline(limiteInferior, color='red', linestyle='--')
            pt.show()
            
        elif columna == 'credit_score':
            
            limiteInferior, limiteSuperior = calcularLimites()
            
            cantidadOutliers = len(data[(data[columna] < limiteInferior) | (data[columna] > limiteSuperior)])
            
            print(f"\n\nColumna: {columna}\n")
            print(f"Cantidad de outliers inicial: {cantidadOutliers}\n")
            
            pt.figure(figsize=(10, 6))
            sb.boxplot(data[columna])
            pt.axhline(limiteInferior, color='red', linestyle='--')
            pt.axhline(limiteSuperior, color='red', linestyle='--')
            pt.show()
            
            data[columna] = data[columna].clip(lower=limiteInferior, upper=limiteSuperior)
            
            print(f"Ajuste completado, limitando valores entre {limiteInferior:.2f} y {limiteSuperior:.2f}\n")
            
            pt.figure(figsize=(10, 6))
            sb.boxplot(data[columna])
            pt.show()
            
        elif columna == 'tenure':
            
            limiteInferior, limiteSuperior = calcularLimites()
            
            cantidadOutliers = len(data[(data[columna] < limiteInferior) | (data[columna] > limiteSuperior)])
        
            print(f"\n\nColumna: {columna}\n")
            print(f"\nCantidad de outliers: {cantidadOutliers}\n")
            
            pt.figure(figsize=(10, 6))
            sb.boxplot(data['tenure'])
            pt.axhline(limiteInferior, color='red', linestyle='--')
            pt.axhline(limiteSuperior, color='red', linestyle='--')
            pt.show()
            
            print("No encontre outliers")
            
        elif columna == 'balance':
            
            limiteInferior, limiteSuperior = calcularLimites()
            
            cantidadOutliers = len(data[(data[columna] < limiteInferior) | (data[columna] > limiteSuperior)])
        
            print(f"\nColumna: {columna}\n")
            print(f"Cantidad de outliers: {cantidadOutliers}\n")
            
            pt.figure(figsize=(10, 6))
            sb.boxplot(data['balance'])
            pt.axhline(limiteInferior, color='red', linestyle='--')
            pt.axhline(limiteSuperior, color='red', linestyle='--')
            pt.show()
            
            mediana_balance = data.loc[data['balance'] != 999999.99, 'balance'].median()
            data['balance'] = data['balance'].replace(999999.99, mediana_balance)
            
            data['NewBalance'] = np.arcsinh(data['balance'])
            
            NewQ1 = data['NewBalance'].quantile(0.25)
            NewQ3 = data['NewBalance'].quantile(0.75)
            NewIQR = NewQ3 - NewQ1
            
            newLimitInferior = NewQ1 - 1.5 * NewIQR
            newLimitSuperior = NewQ3 + 1.5 * NewIQR
            
            cantidadOutliers = len(data[(data['NewBalance'] < newLimitInferior) | (data['NewBalance'] > newLimitSuperior)])
            print(f"\nLos datos ahora están normalizados usando el seno hiperbólico inverso para no perder informacion: Nueva data guardada en 'NewBalance'\n")


            pt.figure(figsize=(10, 6))
            sb.histplot(data['NewBalance'])
            pt.tight_layout()
            pt.show()
            
        elif columna == 'limite_credito':
            
            limiteInferior, limiteSuperior = calcularLimites()
            
            cantidadOutliers = len(data[(data[columna] < limiteInferior) | (data[columna] > limiteSuperior)])
        
            print(f"\n\nColumna: {columna}\n")
            print(f"\nCantidad de outliers: {cantidadOutliers}\n")
            
            limiteInferior = 0
            
            pt.figure(figsize=(10, 6))
            sb.boxplot(data['limite_credito'])
            pt.axhline(limiteInferior, color='red', linestyle='--')
            pt.axhline(limiteSuperior, color='red', linestyle='--')
            pt.show()
            
            data['limite_credito'] = data['limite_credito'].clip(upper=limiteSuperior)
            
            pt.figure(figsize=(10, 6))
            sb.boxplot(data['limite_credito'])
            pt.axhline(limiteInferior, color='red', linestyle='--')
            pt.axhline(limiteSuperior, color='red', linestyle='--')
            pt.show()
            
            cantidadOutliers = len(data[(data['limite_credito'] < limiteInferior) | (data['limite_credito'] > limiteSuperior)])
            
            print(f"\nCantidad de outliers en 'limite_credito' al ser capeados al valor superior del IQR: {cantidadOutliers}\n")
            
        elif columna == 'products_number': 
            
            limiteInferior, limiteSuperior = calcularLimites()
            
            cantidadOutliers = len(data[(data[columna] < limiteInferior) | (data[columna] > limiteSuperior)])
        
            print(f"\n\nColumna: {columna}\n")
            print(f"\nCantidad de outliers: {cantidadOutliers}\n")
            
            pt.figure(figsize=(10, 6))
            sb.boxplot(data['products_number'])
            pt.axhline(limiteInferior, color='red', linestyle='--')
            pt.show()
            
            print(f'Los outliers en {columna} representan una preciada fuente de informacion, ya que se relaciona con el "churn", así que no se eliminara')
            
        elif columna == 'credit_card': 
            
            pt.figure(figsize=(10, 6))
            sb.boxplot(data['credit_card'])
            pt.show()
            
            print('No tiene Outliers')
        
        elif columna == 'active_member':
            
            pt.figure(figsize=(10, 6))
            sb.boxplot(data['active_member'])
            pt.show()
            
            print('No tiene Outliers')
        
        elif columna == 'estimated_salary':
            
            limiteInferior, limiteSuperior = calcularLimites()
            
            limiteInferior = 0
            
            cantidadOutliers = len(data[(data[columna] < limiteInferior) | (data[columna] > limiteSuperior)])
        
            print(f"\n\nColumna: {columna}\n")
            print(f"\nCantidad de outliers: {cantidadOutliers}\n")
            
            pt.figure(figsize=(10, 6))
            sb.boxplot(data['estimated_salary'])
            pt.axhline(limiteInferior, color='red', linestyle='--')
            pt.show()
            
            print('No tiene Outliers')
        
        elif columna == 'churn':
            
            pt.figure(figsize=(10, 6))
            sb.countplot(x=data['churn'], hue=data['churn'], palette='viridis', legend=False)
            pt.show()
        
        return data
    

    data = DuplicatedData(data)
    data = DataNull(data)
    data = Formato(data)
    data = Outliers(data, 'age')
    data = Outliers(data, 'credit_score')
    data = Outliers(data, 'balance')
    data = Outliers(data, 'limite_credito')
    data = Outliers(data, 'products_number')

    print(data)

    data.to_csv('bank_churn_limpio.csv', index = False)

    return data
 
DataCleaningF('bank_churn.csv')




