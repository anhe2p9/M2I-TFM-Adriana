import pandas as pd

from pathlib import Path

import matplotlib.pyplot as plt
import plotly.graph_objects as go

import numpy as np
from pymoo.indicators.hv import HV
from matplotlib.patches import Rectangle

import csv
import os
import re
import os.path

from ILP_CC_reducer.models.ILPmodel import GeneralILPmodel
model = GeneralILPmodel(active_objectives=["extractions", "cc", "loc"])



def generate_2DPF_plot(results_path, output_pdf_path):
    df = pd.read_csv(results_path)

    # Validar que hay datos y al menos 3 columnas numéricas
    if df.shape[0] == 0:
        print("Empty file.")
    else:
        columnas_numericas = df.select_dtypes(include=[np.number])
        if columnas_numericas.shape[1] < 2:
            print("It is not possible to represent 2D PF because there is less than 2 numeric columns.")
        else:
            # Seleccionar primeras 3 columnas numéricas
            objetivos = columnas_numericas.iloc[:, :2].values
            nombres_objetivos = columnas_numericas.columns[:2]

            objetivo1 = df.iloc[:, 0]  # Todas las filas, columna 0
            objetivo2 = df.iloc[:, 1]

            if objetivos.size == 0:
                print("Without numeric values.")
            else:

                o1 = np.array(objetivo1)
                o2 = np.array(objetivo2)

                sorted_indices = np.argsort(o1)
                o1 = o1[sorted_indices]
                o2 = o2[sorted_indices]

                # Crear la figura
                plt.figure(figsize=(8, 6))
                fig, ax = plt.subplots()

                plot_colors = {
                    'lavanda': "#9B8FC6",
                    'naranja': "#E07B39",
                    'verde': "#C1FFC1"
                }

                for i in range(len(o1) - 1):
                    # Línea horizontal hacia el siguiente x
                    plt.plot([o1[i], o1[i + 1]], [o2[i], o2[i]], color=plot_colors['lavanda'],
                             linewidth=2, zorder=2)
                    # Línea vertical hasta el siguiente y
                    plt.plot([o1[i + 1], o1[i + 1]], [o2[i], o2[i + 1]], color=plot_colors['naranja'],
                             linewidth=2, zorder=2)

                for i in range(len(o1)):
                    width = max(o1) + 1 - o1[i]
                    height = max(o2) + 1 - o2[i]
                    rect = Rectangle((o1[i], o2[i]), width, height, facecolor=plot_colors['verde'])
                    ax.add_patch(rect)


                plt.plot([o1[0], o1[0]], [max(o2)+1, o2[0]], color=plot_colors['naranja'],
                         linewidth=2, zorder=2)
                plt.plot([o1[-1], max(o1)+1], [o2[-1], o2[-1]], color=plot_colors['lavanda'],
                         linewidth=2, zorder=2)

                # Dibujar puntos
                plt.scatter(o1, o2, color='black', s=100, zorder=3)

                x_min, x_max = plt.xlim()
                y_min, y_max = plt.ylim()

                dx = (x_max - x_min) * 0.04
                dy = (y_max - y_min) * 0.02

                for idx, (x, y) in enumerate(zip(o1, o2), start=1):
                    plt.text(x + dx, y + dy, f's{idx}', ha='center', fontsize=20, color='black', zorder=4)

                # Etiquetas y título
                objective_map = {
                    model.extractions_objective.__name__: r"$EXTRACTIONS$",
                    model.cc_difference_objective.__name__: r"$CC_{diff}$",
                    model.loc_difference_objective.__name__: r"$LOC_{diff}$"
                }

                x_label = objective_map.get(nombres_objetivos[0])
                y_label = objective_map.get(nombres_objetivos[1])

                plt.tick_params(axis='both', which='major', labelsize=12)

                plt.xlabel(x_label, fontsize=14)
                plt.ylabel(y_label, fontsize=14)
                plt.grid(True, zorder=1)

                # Guardar como PDF
                plt.savefig(output_pdf_path, format='pdf')
                plt.close()
                print(f"2D PF plot saved in {output_pdf_path}.")








def generate_parallel_coordinates_plot(csv_path, output_pdf_path):
    linestyles = ['-', '--', '-.', ':']
    markers = ['o', 's', 'D', '^', 'v', '*', 'x', '+', 'p', 'h', '1', '2', '3', '4', '|', '_']
    colors = plt.cm.tab20.colors  # hasta 20 colores distintos


    # Cargar el CSV
    df = pd.read_csv(csv_path)

    print(df.columns)

    # Suponemos que todas las columnas son objetivos
    objetivos_cols = df.columns

    objective_map = {
        model.extractions_objective.__name__: r"EXTRACTIONS",
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
    print(f"Plot saved in {output_pdf_path}.")













def generate_3DPF_plot(results_path, output_html_path):

    df = pd.read_csv(results_path)

    # Validar que hay datos y al menos 3 columnas numéricas
    if df.shape[0] == 0:
        print("Empty file.")
    else:
        columnas_numericas = df.select_dtypes(include=[np.number])
        if columnas_numericas.shape[1] < 3:
            print("It is not possible to represent 3D PF because there is less than 3 numeric columns.")
        else:
            # Seleccionar primeras 3 columnas numéricas
            objetivos = columnas_numericas.iloc[:, :3].values
            nombres_objetivos = columnas_numericas.columns[:3]

            if objetivos.size == 0:
                print("Without numeric values.")
            else:
                # Calculate nadir directly from Pareto front
                nadir = np.max(objetivos, axis=0)
                ref_point = nadir + 1

                fig = go.Figure()

                n1, n2, n3 = ref_point

                solutions = []
                with open(results_path, newline='') as csvfile:
                    lector = csv.reader(csvfile)
                    next(lector)  # Saltar la cabecera
                    for row in lector:
                        sol = tuple(map(int, row))  # Convierte los strings a enteros
                        solutions.append(sol)

                parallel_face_colors = {
                    'top_bottom': "#E6E6FA",
                    'front_back': "#FFDAB9",
                    'left_right': "#C1FFC1"
                }

                # Caras agrupadas por paralelismo (cada par son dos triángulos)
                parallel_faces = [
                    # Inferior (z=c) y superior (z=10)
                    {
                        'faces': [(0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7)],
                        'color': parallel_face_colors['top_bottom']
                    },
                    # Frontal (y=b) y trasera (y=15)
                    {
                        'faces': [(0, 1, 5), (0, 5, 4), (2, 3, 7), (2, 7, 6)],
                        'color': parallel_face_colors['front_back']
                    },
                    # Derecha (x=20) e izquierda (x=a)
                    {
                        'faces': [(1, 2, 6), (1, 6, 5), (0, 3, 7), (0, 7, 4)],
                        'color': parallel_face_colors['left_right']
                    }
                ]

                for sol in solutions:
                    a, b, c = sol[0], sol[1], sol[2]

                    # Coordenadas de los 8 vértices del cubo
                    x = [a, n1, n1, a, a, n1, n1, a]
                    y = [b, b, n2, n2, b, b, n2, n2]
                    z = [c, c, c, c, n3, n3, n3, n3]

                    # Añadir una traza Mesh3d por cada par de caras con mismo color
                    for group in parallel_faces:
                        color = group['color']
                        face_tris = group['faces']

                        i_vals = [f[0] for f in face_tris]
                        j_vals = [f[1] for f in face_tris]
                        k_vals = [f[2] for f in face_tris]

                        fig.add_trace(go.Mesh3d(
                            x=x, y=y, z=z,
                            i=i_vals, j=j_vals, k=k_vals,
                            color=color,
                            opacity=1,
                            flatshading=True,
                            showscale=False
                        ))

                # Dibujar puntos
                f1, f2, f3 = zip(*solutions)
                fig.add_trace(go.Scatter3d(
                    x=f1, y=f2, z=f3,
                    mode='markers+text',
                    marker=dict(size=10, color='black'),
                    text=[f's{idx+1}' for idx in range(len(solutions))],
                    textposition='top center',
                    textfont=dict(color='black', size = 18),
                    name='Solutions'
                ))

                objective_map = {
                    model.extractions_objective.__name__: r"EXTRACTIONS",
                    model.cc_difference_objective.__name__: r"CC<sub>diff</sub>",
                    model.loc_difference_objective.__name__: r"LOC<sub>diff</sub>"
                }

                x_label = objective_map.get(nombres_objetivos[0])
                y_label = objective_map.get(nombres_objetivos[1])
                z_label = objective_map.get(nombres_objetivos[2])

                fig.update_layout(scene=dict(xaxis=dict(title=dict(text=x_label, font=dict(size=25))),
                                             yaxis=dict(title=dict(text=y_label, font=dict(size=25))),
                                             zaxis=dict(title=dict(text=z_label, font=dict(size=25))),
                                             aspectmode='data'))
                fig.write_html(output_html_path)
                print(f"3D PF saved in {output_html_path}.")


def traverse_and_PF_plot(input_path, output_path):
    carpeta_graficas = os.path.join(output_path, "PF3D")
    os.makedirs(carpeta_graficas, exist_ok=True)

    for proyecto in os.listdir(input_path):
        ruta_proyecto = os.path.join(input_path, proyecto)
        if not os.path.isdir(ruta_proyecto) or proyecto == "PF3D":
            continue

        # Carpeta del proyecto dentro de GRÁFICAS
        carpeta_salida_proyecto = os.path.join(carpeta_graficas, f"{proyecto}_PF_3D")
        os.makedirs(carpeta_salida_proyecto, exist_ok=True)

        for carpeta_clase_metodo in os.listdir(ruta_proyecto):
            if carpeta_clase_metodo.startswith('HybridMethodForThreeObj'):
                ruta_metodo = os.path.join(ruta_proyecto, carpeta_clase_metodo)
                if not os.path.isdir(ruta_metodo):
                    continue

                for archivo in os.listdir(ruta_metodo):
                    if archivo.endswith("_results.csv"):
                        ruta_csv = os.path.join(ruta_metodo, archivo)
                        salida_html = os.path.join(carpeta_salida_proyecto, f"{carpeta_clase_metodo}_3DPF.html")
                        print(f"Generating 3D PF for: {ruta_csv}")
                        generate_3DPF_plot(ruta_csv, salida_html)

            if carpeta_clase_metodo.startswith('EpsilonConstraintAlgorithm'):
                continue





def traverse_and_plot(input_path: str, output_path: str):
    carpeta_graficas = os.path.join(output_path, "plots")
    os.makedirs(carpeta_graficas, exist_ok=True)

    for proyecto in os.listdir(input_path):
        ruta_proyecto = os.path.join(input_path, proyecto)
        if not os.path.isdir(ruta_proyecto) or proyecto == "plots":
            continue

        # Carpeta del proyecto dentro de parallel_coordinates_plots
        carpeta_salida_proyecto = os.path.join(carpeta_graficas, f"{proyecto}_plots")
        os.makedirs(carpeta_salida_proyecto, exist_ok=True)

        for carpeta_clase_metodo in os.listdir(ruta_proyecto):
                ruta_metodo = os.path.join(ruta_proyecto, carpeta_clase_metodo)
                if not os.path.isdir(ruta_metodo):
                    continue

                for archivo in os.listdir(ruta_metodo):
                    if archivo.endswith("_results.csv"):
                        ruta_csv = os.path.join(ruta_metodo, archivo)
                        if carpeta_clase_metodo.startswith('HybridMethodForThreeObj'):
                            #  Parallel coordinates plots
                            salida_pdf = os.path.join(carpeta_salida_proyecto,
                                                      f"{carpeta_clase_metodo}_parallel_coordinates_plot.pdf")
                            print(f"Generating parallel coordinates plot for: {ruta_csv}")
                            generate_parallel_coordinates_plot(ruta_csv, salida_pdf)

                            #  3D PF plots
                            salida_html = os.path.join(carpeta_salida_proyecto, f"{carpeta_clase_metodo}_3DPF.html")
                            print(f"Generating 3D PF for: {ruta_csv}")
                            generate_3DPF_plot(ruta_csv, salida_html)
                        if carpeta_clase_metodo.startswith('EpsilonConstraintAlgorithm'):
                            salida_pdf = os.path.join(carpeta_salida_proyecto,
                                                      f"{carpeta_clase_metodo}_2DPF_plot.pdf")
                            print(f"Generating 2D PF plot for: {ruta_csv}")
                            generate_2DPF_plot(ruta_csv, salida_pdf)





def generate_statistics_obj(
    results_path: str,
    complete_data_path: str,
    output_path: str,
    proyecto: str,
    clase_metodo: str,
    num_obj: int
):
    """
    Calcula estadísticas para un archivo CSV con num_obj objetivos.
    Devuelve un diccionario con los resultados o None si ocurre un error.
    """
    try:
        df = pd.read_csv(results_path)
        execution_time_average = ""
        total_execution_time = ""
        if complete_data_path:
            compl_dat = pd.read_csv(complete_data_path)
            if not compl_dat.empty:
                execution_time_average = compl_dat["executionTime"].mean()

        if output_path:
            with open(output_path, "r") as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip()  # última línea
                    # Extraemos el número usando regex
                    match = re.search(r"Total execution time:\s*([0-9.]+)", last_line)
                    if match:
                        time_value = float(match.group(1))
                        total_execution_time = time_value

        if df.empty:
            raise ValueError("Archivo vacío")

        columnas_numericas = df.select_dtypes(include=[np.number])
        if columnas_numericas.shape[1] < num_obj:
            raise ValueError(f"Menos de {num_obj} columnas numéricas")

        objetivos = columnas_numericas.iloc[:, :num_obj].values
        nombres_objetivos = columnas_numericas.columns[:num_obj]

        nadir = np.max(objetivos, axis=0)
        ref_point = nadir + 1

        hv = HV(ref_point=ref_point)
        hipervolumen = hv.do(objetivos)

        ideal = np.min(objetivos, axis=0)
        hv_max = hv.do(ideal)
        hv_normalized = hipervolumen / hv_max

        if "_" in clase_metodo:
            clase, metodo = clase_metodo.rsplit("_", 1)
            parts = clase.split("_", 2)

            algorithm = parts[0]  # antes del primer "_"
            clase = parts[2]
        else:
            clase, metodo = clase_metodo, ""

        medias = np.round(np.mean(objetivos, axis=0), 2)
        desvios = np.round(np.std(objetivos, axis=0), 2)
        medianas = np.median(objetivos, axis=0)
        iqr = np.percentile(objetivos, 75, axis=0) - np.percentile(objetivos, 25, axis=0)

        res = {
            "project": proyecto,
            "class": clase,
            "method": metodo,
            "algorithm": algorithm,
            "num_solutions": objetivos.shape[0],
            "nadir": f"({', '.join(str(int(x)) for x in ref_point)})",
            "hypervolume": hipervolumen,
            "normalized_hypervolume": np.round(hv_normalized, 2),
            "execution_time_average": execution_time_average,
            "total_execution_time": total_execution_time
        }

        for i, nombre in enumerate(nombres_objetivos):
            res[f"avg_{nombre}"] = medias[i]
            res[f"std_{nombre}"] = desvios[i]
            res[f"median_{nombre}"] = medianas[i]
            res[f"iqr_{nombre}"] = iqr[i]

        return res

    except Exception as e:
        return {"error": str(e)}


def generate_statistics(input_path: str, output_path: str):
    resultados_2obj = []
    resultados_3obj = []
    archivos_invalidos = []

    for proyecto in os.listdir(input_path):
        ruta_proyecto = os.path.join(input_path, proyecto)
        if not os.path.isdir(ruta_proyecto):
            continue

        for clase_metodo in os.listdir(ruta_proyecto):
            ruta_clase = os.path.join(ruta_proyecto, clase_metodo)
            if not os.path.isdir(ruta_clase):
                continue

            results_file = next(
                (f for f in os.listdir(ruta_clase) if f.endswith("_results.csv")),
                None
            )
            if results_file is None:
                continue

            complete_data_path, execution_time_path = None, None

            complete_data_file = next(
                (f for f in os.listdir(ruta_clase) if f.endswith("_complete_data.csv")),
                None
            )

            execution_time_file = next(
                (f for f in os.listdir(ruta_clase) if f.endswith("_output.txt")),
                None
            )

            results_path = os.path.join(ruta_clase, results_file)

            if complete_data_file:
                complete_data_path = os.path.join(ruta_clase, complete_data_file)
            if execution_time_file:
                execution_time_path = os.path.join(ruta_clase, execution_time_file)

            if ('_extractions-cc_' in clase_metodo
                  or '_cc-extractions_' in clase_metodo
                  or '_loc-cc_' in clase_metodo
                  or '_cc-loc_' in clase_metodo
                  or '_extractions-loc_' in clase_metodo
                  or '_loc-extractions_' in clase_metodo):
                resultado = generate_statistics_obj(
                    results_path, complete_data_path, execution_time_path, proyecto, clase_metodo, num_obj=2
                )
                if resultado is None or 'error' in resultado:
                    archivos_invalidos.append({
                        "project": proyecto,
                        "class_method": clase_metodo,
                        "archivo": results_path,
                        "error": resultado.get('error', 'Error desconocido')
                    })
                else:
                    resultados_2obj.append(resultado)

            elif ('extractions-cc-loc' in clase_metodo
                  or 'extractions-loc-cc' in clase_metodo
                  or 'loc-extractions-cc' in clase_metodo
                  or 'cc-extractions-loc' in clase_metodo
                  or 'cc-loc-extractions' in clase_metodo
                  or 'loc-cc-extractions' in clase_metodo):
                resultado = generate_statistics_obj(
                    results_path, complete_data_path, execution_time_path, proyecto, clase_metodo, num_obj=3
                )
                if resultado is None or 'error' in resultado:
                    archivos_invalidos.append({
                        "project": proyecto,
                        "class_method": clase_metodo,
                        "archivo": results_path,
                        "error": resultado.get('error', 'Error desconocido')
                    })
                else:
                    resultados_3obj.append(resultado)

    # Guardar resultados
    if resultados_2obj:
        df_2obj = pd.DataFrame(resultados_2obj)
        ruta_2obj = os.path.join(output_path, "hipervolumen_2objs_summary.csv")
        df_2obj.to_csv(ruta_2obj, index=False)
        print(f"\n✅ Resumen 2 objetivos guardado en: {ruta_2obj}")

    if resultados_3obj:
        df_3obj = pd.DataFrame(resultados_3obj)
        ruta_3obj = os.path.join(output_path, "hipervolumen_3objs_summary.csv")
        df_3obj.to_csv(ruta_3obj, index=False)
        print(f"\n✅ Resumen 3 objetivos guardado en: {ruta_3obj}")

    if archivos_invalidos:
        df_invalidos = pd.DataFrame(archivos_invalidos)
        ruta_invalidos = os.path.join(output_path, "archivos_invalidos.csv")
        df_invalidos.to_csv(ruta_invalidos, index=False)
        print(f"⚠️  Archivos con errores guardados en: {ruta_invalidos}")


def analyze_model_data(method_path):
    # Main path containing the subfolders
    main_path = Path(method_path)

    # Dictionary to store the three DataFrames
    dataframes = {}
    row_counts = {}

    # Get all CSV files in the folder
    csv_files = list(method_path.glob("*.csv"))

    # Read each CSV file and store it in the dictionary
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        # Extract the part after the last underscore and before ".csv"
        suffix = csv_file.stem.split("_")[-1]
        dataframes[suffix] = df  # Use the suffix as key

        # Count the number of rows (excluding header)
        row_counts[suffix] = len(df)

    # Example: sum rows from specific CSVs
    keys_to_sum = ["sequences", "nested"]
    variables = sum(row_counts[key] for key in keys_to_sum if key in row_counts)

    keys_to_sum = ["sequences", "nested", "conflict"]
    constraints = sum(row_counts[key] for key in keys_to_sum if key in row_counts) + 1 # sum 1 for the x_0 == 1 constraint

    return variables, constraints