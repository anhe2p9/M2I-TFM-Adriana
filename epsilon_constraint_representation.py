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
    # Cargar el CSV
    df = pd.read_csv(csv_path)

    match = re.search(r'EpsilonConstraintAlgorithm_(.+?)_results\.csv$', os.path.basename(csv_path))
    if match:
        method_name = match.group(1)
    else:
        method_name = os.path.basename(csv_path)

    print(df.columns)

    # Suponemos que todas las columnas son objetivos
    objetivos_cols = df.columns

    objective_map = {
        model.sequences_objective.__name__: "SEQUENCES",
        model.cc_difference_objective.__name__: "CC",
        model.loc_difference_objective.__name__: "LOC"
    }


    # Crear nombres s1, s2, ..., sN
    df['id'] = [f's{i+1}' for i in range(len(df))]

    # DataFrame para graficar
    df_plot = df[['id'] + list(objetivos_cols)]

    # Renombrar columnas con nombres amigables (si coinciden)
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
    plt.legend(title='Solution', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    # plt.show()

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