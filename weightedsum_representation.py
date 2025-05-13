import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

# Cargar el CSV (ajusta el nombre de tu archivo)
df = pd.read_csv('output/WeightedSumAlgorithm_PRUEBA2_results.csv')

print(df.columns)

pesos_cols = df.columns[:3]
objetivos_cols = df.columns[-3:]


# Creamos una etiqueta categórica clara para la combinación de pesos
df['pesos'] = df[pesos_cols].astype(str).agg('-'.join, axis=1)

# Creamos nuevo DataFrame solo con las columnas necesarias (orden importante)
df_plot = df[['pesos'] + list(objetivos_cols)]

# Asegúrate de que todas las columnas (menos 'pesos') sean numéricas
for col in objetivos_cols:
    df_plot[col] = pd.to_numeric(df_plot[col], errors='coerce')

# Dibujo
plt.figure(figsize=(12, 6))
parallel_coordinates(df_plot, class_column='pesos', colormap=plt.cm.viridis, alpha=0.5)
plt.title('Gráfica de coordenadas paralelas - Objetivos vs Pesos')
plt.ylabel('Valor de los objetivos')
plt.xlabel('Objetivos')
plt.legend([],[], frameon=False)  # Puedes quitar la leyenda si hay muchas combinaciones
plt.tight_layout()
plt.show()