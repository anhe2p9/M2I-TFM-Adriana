import os.path

import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
from ILP_CC_reducer.models import MultiobjectiveILPmodel
import re

plt.rcParams['text.usetex'] = True

model = MultiobjectiveILPmodel()

csv_path = 'output/pruebas/EpsilonConstraintAlgorithm_small_test/EpsilonConstraintAlgorithm_small_test_results.csv'

# Cargar el CSV
df = pd.read_csv(csv_path)

match = re.search(r'EpsilonConstraintAlgorithm_(.+?)_results\.csv$', os.path.basename(csv_path))
if match:
    method_name = match.group(1)

print(df.columns)

# Suponemos que todas las columnas son objetivos
objetivos_cols = df.columns

objective_map = {
    model.sequences_objective.__name__: "SEQUENCES",
    model.cc_difference_objective.__name__: "CC",
    model.loc_difference_objective.__name__: "LOC"
}

# Creamos una columna auxiliar para identificar cada fila si no hay clases (por ejemplo, un índice)
df['id'] = df.index.astype(str)

# Creamos el DataFrame para graficar
df_plot = df[['id'] + list(objetivos_cols)]


df_plot_renamed = df_plot.rename(columns=objective_map)

# Asegúrate de que las columnas sean numéricas
for col in objetivos_cols:
    df_plot[col] = pd.to_numeric(df_plot[col], errors='coerce')

# Dibujo
plt.figure(figsize=(8, 6))
parallel_coordinates(df_plot_renamed, class_column='id', colormap=plt.cm.viridis, alpha=0.7)
plt.title(f'Multiobjective ILP solutions for {method_name}')
plt.ylabel('Objectives values')
plt.xlabel('Objetives to minimize')
plt.legend([], [], frameon=False)  # Quita la leyenda si hay muchas soluciones
plt.tight_layout()
plt.show()