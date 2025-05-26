import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Necesario para 3D plots
import seaborn as sns
import ternary

from ILP_CC_reducer.models import MultiobjectiveILPmodel

model = MultiobjectiveILPmodel()

# Cargar CSV
df = pd.read_csv('output/bytecode-viewer/EpsilonConstraintAlgorithm_renameClassNode/EpsilonConstraintAlgorithm_renameClassNode_results.csv')


objective_map = {
    model.sequences_objective.__name__: "SEQUENCES",
    model.cc_difference_objective.__name__: "CC",
    model.loc_difference_objective.__name__: "LOC"
}


# Detectar automáticamente las 3 últimas columnas como los objetivos
objetivo1, objetivo2, objetivo3 = df.columns[-3:]

# Extraer coordenadas
x = df[objetivo1]
y = df[objetivo2]
z = df[objetivo3]



""" Figura 3D """
# Crear la figura y los ejes 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Dibujar los puntos
ax.scatter(x, y, z, c='blue', marker='o', alpha=0.8)


# Anotar cada punto con sus coordenadas
for xi, yi, zi in zip(x, y, z):
    ax.text(xi, yi, zi, f'({xi:.2f}, {yi:.2f}, {zi:.2f})', fontsize=8, color='black')


# Etiquetas
ax.set_xlabel(f"{objective_map[objetivo1]}")
ax.set_ylabel(f"{objective_map[objetivo2]}")
ax.set_zlabel(f"{objective_map[objetivo3]}")
ax.set_title('Soluciones no dominadas en espacio objetivo')

plt.tight_layout()
plt.show()




""" Matriz de dispersión """

# Selección de objetivos
objectives = df[[objetivo1, objetivo2, objetivo3]]

# Pairplot (matriz de dispersión)
sns.pairplot(objectives)
plt.suptitle('Proyecciones 2D entre objetivos')
plt.tight_layout()
plt.show()




""" Gráfico de burbujas """

plt.figure(figsize=(8, 6))
plt.scatter(df[objetivo1], df[objetivo2], c=df[objetivo3], cmap='viridis', s=60)
plt.xlabel(objetivo1)
plt.ylabel(objetivo2)
plt.title('Frente de Pareto: color representa ' + objetivo3)
plt.colorbar(label=objetivo3)
plt.grid(True)
plt.show()