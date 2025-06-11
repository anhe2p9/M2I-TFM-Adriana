import os.path

import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
from ILP_CC_reducer.models import MultiobjectiveILPmodel
import re

plt.rcParams['text.usetex'] = True

model = MultiobjectiveILPmodel()

# csv_path = 'output/pruebas/EpsilonConstraintAlgorithm_small_test/EpsilonConstraintAlgorithm_small_test_results.csv'

# csv_path = 'output/bytecode-viewer/EpsilonConstraintAlgorithm_renameMethodNode/EpsilonConstraintAlgorithm_renameMethodNode_results.csv'
# csv_path = 'output/bytecode-viewer/EpsilonConstraintAlgorithm_getTreeCellRendererComponent/EpsilonConstraintAlgorithm_getTreeCellRendererComponent_results.csv'
# csv_path = 'output/bytecode-viewer/EpsilonConstraintAlgorithm_compile/EpsilonConstraintAlgorithm_compile_results.csv'
# csv_path = 'output/bytecode-viewer/EpsilonConstraintAlgorithm_downloadZipsOnly/EpsilonConstraintAlgorithm_downloadZipsOnly_results.csv'
# csv_path = 'output/bytecode-viewer/EpsilonConstraintAlgorithm_parseCommandLine/EpsilonConstraintAlgorithm_parseCommandLine_results.csv'
# csv_path = 'output/bytecode-viewer/EpsilonConstraintAlgorithm_doSaveJarDecompiled/EpsilonConstraintAlgorithm_doSaveJarDecompiled_results.csv'
# csv_path = 'output/bytecode-viewer/EpsilonConstraintAlgorithm_FileDrop/EpsilonConstraintAlgorithm_FileDrop_results.csv'
# csv_path = 'output/bytecode-viewer/EpsilonConstraintAlgorithm_renameClassNode/EpsilonConstraintAlgorithm_renameClassNode_results.csv'
# csv_path = 'output/bytecode-viewer/EpsilonConstraintAlgorithm_renameFieldNode/EpsilonConstraintAlgorithm_renameFieldNode_results.csv'
# csv_path = "C:/Users/X1502/Adriana/LAB334/resultados_multiobj/prueba_FileDrop.csv"
# csv_path = "C:/Users/X1502/Adriana/LAB334/resultados_multiobj/bytecode-viewer/EpsilonConstraintAlgorithm_compile/EpsilonConstraintAlgorithm_compile_results.csv"
# csv_path = "C:/Users/X1502/Adriana/LAB334/resultados_multiobj/bytecode-viewer/EpsilonConstraintAlgorithm_renameMethodNode/EpsilonConstraintAlgorithm_renameMethodNode_results.csv"
# csv_path = "C:/Users/X1502/Adriana/LAB334/resultados_multiobj/bytecode-viewer/EpsilonConstraintAlgorithm_getTreeCellRendererComponent/EpsilonConstraintAlgorithm_getTreeCellRendererComponent_results.csv"
# csv_path = "C:/Users/X1502/Adriana/LAB334/resultados_multiobj/bytecode-viewer/EpsilonConstraintAlgorithm_downloadZipsOnly/EpsilonConstraintAlgorithm_downloadZipsOnly_results.csv"
# csv_path = "C:/Users/X1502/Adriana/LAB334/resultados_multiobj/bytecode-viewer/EpsilonConstraintAlgorithm_parseCommandLine/EpsilonConstraintAlgorithm_parseCommandLine_results.csv"
# csv_path = "C:/Users/X1502/Adriana/LAB334/resultados_multiobj/bytecode-viewer/EpsilonConstraintAlgorithm_doSaveJarDecompiled/EpsilonConstraintAlgorithm_doSaveJarDecompiled_results.csv"
# csv_path = "C:/Users/X1502/Adriana/LAB334/resultados_multiobj/bytecode-viewer/EpsilonConstraintAlgorithm_renameClassNode/EpsilonConstraintAlgorithm_renameClassNode_results.csv"
# csv_path = "C:/Users/X1502/Adriana/LAB334/resultados_multiobj/bytecode-viewer/EpsilonConstraintAlgorithm_renameFieldNode/EpsilonConstraintAlgorithm_renameFieldNode_results.csv"



def generate_graph(csv_path, output_pdf_path):
    linestyles = ['-', '--', '-.', ':']
    markers = ['o', 's', 'D', '^', 'v', '*', 'x', '+', 'p', 'h', '1', '2', '3', '4', '|', '_']
    colors = plt.cm.tab20.colors  # hasta 20 colores distintos


    # Cargar el CSV
    df = pd.read_csv(csv_path)

    print(df.columns)

    # Suponemos que todas las columnas son objetivos
    objetivos_cols = df.columns

    objective_map = {
        model.sequences_objective.__name__: r"SEQUENCES$_{sum}$",
        model.cc_difference_objective.__name__: r"CC$_{diff}$",
        model.loc_difference_objective.__name__: r"LOC$_{diff}$"
    }


    # Crear nombres s1, s2, ..., sN
    df['id'] = [f's{i+1}' for i in range(len(df))]

    # DataFrame para graficar
    df_plot = df[['id'] + list(objetivos_cols)]


    # Asegúrate de que las columnas sean numéricas
    for col in objetivos_cols:
        df_plot[col] = pd.to_numeric(df_plot[col], errors='coerce')

    x = range(len(objetivos_cols))

    for i, row in df.iterrows():
        style = linestyles[i % len(linestyles)]
        marker = markers[i % len(markers)]
        color = colors[i % len(colors)]
        y = [row[col] for col in objetivos_cols]
        plt.plot(x, y, label=row['id'], linestyle=style, marker=marker, color=color, alpha=0.8, linewidth=3)

    # Mapeo de nombres de objetivos a nombres legibles
    x_labels = [objective_map.get(col, col) for col in objetivos_cols]

    # Dibujo
    plt.xticks(ticks=x, labels=x_labels, fontsize=18)
    plt.ylabel(r'Objectives values', fontsize=18)
    plt.xlabel(r'Objetives to minimize', fontsize=18)
    plt.legend([], [], frameon=False)  # Quita la leyenda si hay muchas soluciones
    plt.legend(title=r'Solution', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.grid(True)
    plt.tight_layout()

    # Guardar como PDF
    plt.savefig(output_pdf_path, format='pdf')
    plt.close()


def recorrer_y_graficar(ruta_base):
    carpeta_graficas = os.path.join(ruta_base, "GRÁFICAS")
    os.makedirs(carpeta_graficas, exist_ok=True)

    for proyecto in os.listdir(ruta_base):
        ruta_proyecto = os.path.join(ruta_base, proyecto)
        if not os.path.isdir(ruta_proyecto) or proyecto == "GRÁFICAS":
            continue

        # Carpeta del proyecto dentro de GRÁFICAS
        carpeta_salida_proyecto = os.path.join(carpeta_graficas, f"{proyecto}_gráficas")
        os.makedirs(carpeta_salida_proyecto, exist_ok=True)

        for carpeta_clase_metodo in os.listdir(ruta_proyecto):
            ruta_metodo = os.path.join(ruta_proyecto, carpeta_clase_metodo)
            if not os.path.isdir(ruta_metodo):
                continue

            for archivo in os.listdir(ruta_metodo):
                if archivo.endswith("_results.csv"):
                    ruta_csv = os.path.join(ruta_metodo, archivo)
                    salida_pdf = os.path.join(carpeta_salida_proyecto, f"{carpeta_clase_metodo}.pdf")
                    print(f"Generando gráfica para: {ruta_csv}")
                    generate_graph(ruta_csv, salida_pdf)


# Llama a la función indicando el directorio raíz donde están los proyectos
base_path = "C:/Users/X1502/Adriana/LAB334/resultados_multiobj/DOCKER_output"
recorrer_y_graficar(base_path)