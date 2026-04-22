# Predicción de Retención de Clientes: Proyecto Bank Churn

Este proyecto de Data Science se enfoca en predecir la salida de clientes en el sector bancario. El objetivo principal no es solo generar una predicción, sino entender los patrones de comportamiento que llevan a un cliente a dejar la entidad. Al identificar variables críticas como la edad o el volumen de productos contratados, el sistema permite generar estrategias de retención que aportan valor directo al negocio.


## Procesamiento de Datos

La etapa de limpieza fue fundamental para garantizar la fiabilidad de los resultados, se dice que el 80% del trabajo de un data scientist es limpiar los datos, y es cierto, hice varios pasos como:
* Identificación y tratamiento de registros duplicados y gestión de valores nulos: Con estrategias como la eliminacion, etiquetado e imputación por mediana.
* Estandarización de formatos para asegurar la consistencia.
* Análisis y tratamiento de outliers para evitar sesgos en el entrenamiento del modelo: Por ejemplo, el tratamiento que le apliqué a los valores de 'balance' haciéndoles una transformación arcoseno, para estabilizar la varianza y reducir el sesgo de esta variable asimétrica.

## Análisis Exploratorio (EDA)

Durante el análisis, busqué extraer conclusiones accionables más allá de la simple visualización:
* Identifiqué que la cantidad de productos activos es uno de los predictores más fuertes de permanencia; logré ver que, aunque resulta contraintuitivo, a mayor cantidad de productos, mayor es el riesgo de fuga.
* Analicé la relación entre la edad y la tasa de abandono, detectando segmentos de riesgo específicos.
* Las conclusiones obtenidas permitieron refinar las variables para el modelo final.

## Modelado y Metodología

El desarrollo del modelo incluyó pasos técnicos para mejorar el rendimiento:
* Entrené el sistema con un dataset de 9,507 registros, evaluando su desempeño mediante métricas de precisión y capacidad de generalización.
* El análisis de importancia de variables confirmó qué factores pesan más a la hora de predecir la fuga.

## Visualización y Dashboard

Para hacer los datos accesibles, desarrollé un dashboard interactivo en Streamlit. Esta herramienta muestra:
* Un gráfico pairplot para observar la correlación de las variables entre sí y su distribución en la diagonal principal.
* Un gráfico de la importancia de las variables, que muestra el peso que tuvo cada una en el entrenamiento del modelo.
* Una gráfica que muestra el churn relacionado con el país e identifica cuál representa un mayor riesgo de fuga.
* Una sección con datos clave, como el número de clientes analizados, la precisión del modelo y la tasa de fuga histórica.
* Al final, un simulador que le permite al usuario probar el modelo modificando las variables del cliente.
* Una interfaz clara para interpretar los resultados del modelo de forma rápida.

## Herramientas Utilizadas

* Python para todo el flujo de datos.
* Pandas y NumPy para manipulación y limpieza.
* Scikit-learn para el desarrollo del modelo predictivo.
* Matplotlib y Seaborn para el análisis visual.
* Streamlit para la creación del dashboard.

## Conclusiones del Proyecto

Este trabajo me permitió aplicar conocimientos de estadística y programación en un escenario real. Aprendí a transformar datos brutos en una herramienta de alerta temprana y a utilizar técnicas de ingeniería de variables para resolver problemas de distribución en los datos de entrada.
