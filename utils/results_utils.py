import pandas as pd

import matplotlib.pyplot as plt
import plotly.graph_objects as go

import numpy as np
from pymoo.indicators.hv import HV

import csv
import os
import os.path

from ILP_CC_reducer.models.ILPmodelRsain import ILPmodelRsain
model = ILPmodelRsain()



def generate_PF_plot(results_path, output_html_path):

    df = pd.read_csv(results_path)

    # Validar que hay datos y al menos 3 columnas numéricas
    if df.shape[0] == 0:
        print("Empty file.")
    else:
        columnas_numericas = df.select_dtypes(include=[np.number])
        if columnas_numericas.shape[1] < 3:
            print("Less than 3 numeric columns.")
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
                    'top_bottom': 'purple',
                    'front_back': 'orange',
                    'left_right': 'cyan'
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
                    mode='markers',
                    marker=dict(size=4, color='black'),
                    name='Soluciones'
                ))

                objective_map = {
                    model.sequences_objective.__name__: r"SEQUENCES<sub>sum</sub>",
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
            ruta_metodo = os.path.join(ruta_proyecto, carpeta_clase_metodo)
            if not os.path.isdir(ruta_metodo):
                continue

            for archivo in os.listdir(ruta_metodo):
                if archivo.endswith("_results.csv"):
                    ruta_csv = os.path.join(ruta_metodo, archivo)
                    salida_html = os.path.join(carpeta_salida_proyecto, f"{carpeta_clase_metodo}_3DPF.html")
                    print(f"Generating Pf 3D for: {ruta_csv}")
                    generate_PF_plot(ruta_csv, salida_html)


def generate_plot(csv_path, output_pdf_path):
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
    print(f"Plot saved in {output_pdf_path}.")


def traverse_and_plot(input_path: str, output_path: str):
    carpeta_graficas = os.path.join(output_path, "GRÁFICAS")
    os.makedirs(carpeta_graficas, exist_ok=True)

    for proyecto in os.listdir(input_path):
        ruta_proyecto = os.path.join(input_path, proyecto)
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
                    salida_pdf = os.path.join(carpeta_salida_proyecto, f"{carpeta_clase_metodo}_parallel_coordinates_plot.pdf.pdf")
                    print(f"Generando gráfica para: {ruta_csv}")
                    generate_plot(ruta_csv, salida_pdf)

def generate_statistics(input_path: str, output_path: str):
    # Inicializar listas
    resultados = []
    archivos_invalidos = []

    # Recorrer todos los proyectos
    for proyecto in os.listdir(input_path):
        ruta_proyecto = os.path.join(input_path, proyecto)
        if not os.path.isdir(ruta_proyecto):
            continue

        for clase_metodo in os.listdir(ruta_proyecto):
            ruta_clase = os.path.join(ruta_proyecto, clase_metodo)
            if not os.path.isdir(ruta_clase):
                continue

            results_file = next((f for f in os.listdir(ruta_clase) if f.endswith("_results.csv")), None)
            if results_file is None:
                continue

            results_path = os.path.join(ruta_clase, results_file)

            try:
                df = pd.read_csv(results_path)

                # Validar que hay datos y al menos 3 columnas numéricas
                if df.shape[0] == 0:
                    raise ValueError("Archivo vacío")

                columnas_numericas = df.select_dtypes(include=[np.number])
                if columnas_numericas.shape[1] < 3:
                    raise ValueError("Menos de 3 columnas numéricas")

                # Seleccionar primeras 3 columnas numéricas
                objetivos = columnas_numericas.iloc[:, :3].values
                nombres_objetivos = columnas_numericas.columns[:3]

                if objetivos.size == 0:
                    raise ValueError("Sin datos numéricos")

                # Calculate nadir directly from Pareto front
                nadir = np.max(objetivos, axis=0)
                ref_point = nadir + 1

                # Punto de referencia y cálculo de hipervolumen
                hv = HV(ref_point=ref_point)
                hipervolumen = hv.do(objetivos)

                # Cálculo del hipervolumen de la caja total
                ideal = np.min(objetivos, axis=0)
                hv_max = hv.do(ideal)

                # Nomalize PF
                hv_normalized = hipervolumen / hv_max


                # Dividir clase_metodo en clase y método
                if "_" in clase_metodo:
                    clase, metodo = clase_metodo.rsplit("_", 1)
                else:
                    clase, metodo = clase_metodo, ""

                # Estadísticas
                medias = np.round(np.mean(objetivos, axis=0), 2)
                desvios = np.round(np.std(objetivos, axis=0), 2)
                medianas = np.median(objetivos, axis=0)
                iqr = np.percentile(objetivos, 75, axis=0) - np.percentile(objetivos, 25, axis=0)

                resultados.append({
                    "project": proyecto,
                    "class": clase,
                    "method": metodo,
                    "num_solutions": objetivos.shape[0],
                    "nadir": f"({', '.join(str(int(x)) for x in ref_point)})",
                    "hypervolume": hipervolumen,
                    "normalized_hypervolume": np.round(hv_normalized,2),
                    f"avg_{nombres_objetivos[0]}": medias[0],
                    f"std_{nombres_objetivos[0]}": desvios[0],
                    f"median_{nombres_objetivos[0]}": medianas[0],
                    f"iqr_{nombres_objetivos[0]}": iqr[0],
                    f"avg_{nombres_objetivos[1]}": medias[1],
                    f"std_{nombres_objetivos[1]}": desvios[1],
                    f"median_{nombres_objetivos[1]}": medianas[1],
                    f"iqr_{nombres_objetivos[1]}": iqr[1],
                    f"avg_{nombres_objetivos[2]}": medias[2],
                    f"std_{nombres_objetivos[2]}": desvios[2],
                    f"median_{nombres_objetivos[2]}": medianas[2],
                    f"iqr_{nombres_objetivos[2]}": iqr[2],
                })


            except Exception as e:
                archivos_invalidos.append({
                    "proyecto": proyecto,
                    "clase_metodo": clase_metodo,
                    "archivo": results_path,
                    "error": str(e)
                })

    # Guardar resumen principal
    df_resultados = pd.DataFrame(resultados)
    ruta_salida = os.path.join(output_path, "hipervolumen_summary.csv")
    df_resultados.to_csv(ruta_salida, index=False)
    print(f"\n✅ Resumen guardado en: {ruta_salida}")

    # Guardar archivos inválidos (opcional)
    if archivos_invalidos:
        df_invalidos = pd.DataFrame(archivos_invalidos)
        ruta_invalidos = os.path.join(output_path, "archivos_invalidos.csv")
        df_invalidos.to_csv(ruta_invalidos, index=False)
        print(f"⚠️  Algunos archivos fueron descartados por errores. Detalles en: {ruta_invalidos}")
