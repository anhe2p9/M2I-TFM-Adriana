import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

# Cargar el CSV
df = pd.read_csv('output/Nombre_de_tu_archivo.csv')  # <-- Cambia esto

print(df.columns)

# Suponemos que todas las columnas son objetivos
objetivos_cols = df.columns

# Creamos una columna auxiliar para identificar cada fila si no hay clases (por ejemplo, un índice)
df['id'] = df.index.astype(str)

# Creamos el DataFrame para graficar
df_plot = df[['id'] + list(objetivos_cols)]

# Asegúrate de que las columnas sean numéricas
for col in objetivos_cols:
    df_plot[col] = pd.to_numeric(df_plot[col], errors='coerce')

# Dibujo
plt.figure(figsize=(12, 6))
parallel_coordinates(df_plot, class_column='id', colormap=plt.cm.viridis, alpha=0.7)
plt.title('Gráfica de coordenadas paralelas - Objetivos')
plt.ylabel('Valor de los objetivos')
plt.xlabel('Objetivos')
plt.legend([], [], frameon=False)  # Quita la leyenda si hay muchas soluciones
plt.tight_layout()
plt.show()