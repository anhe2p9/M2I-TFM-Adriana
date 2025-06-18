import numpy as np
import math

import os
import os.path
from typing import Any
import csv
from pathlib import Path
import pandas as pd

import pyomo.environ as pyo # ayuda a definir y resolver problemas de optimización
import pyomo.dataportal as dp # permite cargar datos para usar en esos modelos de optimización

import matplotlib.pyplot as plt
from ILP_CC_reducer.models import MultiobjectiveILPmodel

import numpy as np
from pymoo.indicators.hv import HV

plt.rcParams['text.usetex'] = True
model = MultiobjectiveILPmodel()


def modify_component(mobj_model: pyo.AbstractModel, component: str, new_value: pyo.Any) -> None:
    """ Modify a given component of a model to avoid construct warnings """
    
    if hasattr(mobj_model, component):
        mobj_model.del_component(component)
    mobj_model.add_component(component, new_value)

def concrete_and_solve_model(mobj_model: pyo.AbstractModel, instance: dp.DataPortal):
    """ Generates a Concrete Model for a given model instance and solves it using CPLEX solver """
    
    concrete = mobj_model.create_instance(instance)
    solver = pyo.SolverFactory('cplex')
    result = solver.solve(concrete)
    return concrete, result

def write_output_to_files(csv_info: list, concrete: pyo.ConcreteModel, project_name: str, class_name: str,
                          method_name: str, algorithm: str, output_data: list=None, complete_data: list=None,
                          nadir: list= None):

    result_name = f"{algorithm}_{class_name}_{method_name}"
    base_path = f"output/results/{project_name}/{result_name}"
    method_path = f"{base_path}/{method_name}"

    if not os.path.exists(base_path):
        os.makedirs(base_path)

    # Save model in a LP file
    if concrete:
        concrete.write(f'{method_path}.lp',
                       io_options={'symbolic_solver_labels': True})
        print("Model correctly saved in a LP file.")

    # Save data in a CSV file
    filename = f"{method_path}_results.csv"
            
    if os.path.exists(filename):
        os.remove(filename)
            
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(csv_info)
        print("CSV file correctly created.")
    
    # Save output in a TXT file
    if output_data:
        with open(f"{method_path}_output.txt", "w") as f:
            for linea in output_data:
                f.write(linea + "\n")
            print("Output correctly saved in a TXT file.")

    # Save output in a TXT file
    if complete_data:
        with open(f"{method_path}_complete_data.csv",
                  mode="w", newline="", encoding="utf-8") as complete_csv:
            writer = csv.writer(complete_csv)
            writer.writerows(complete_data)
            print("Complete CSV file correctly created.")

    # # Save nadir point in a csv file
    # if nadir:
    #     with open(f"{method_path}_nadir.csv",
    #               mode="w", newline="", encoding="utf-8") as nadir_csv:
    #         writer = csv.writer(nadir_csv)
    #         writer.writerows(nadir)
    #         print("Nadir CSV file correctly created.")



def print_result_and_sequences(concrete: pyo.ConcreteModel, solver_status: str, newrow: list, obj2: str=None):
    """ Print results and a vertical list of sequences selected """
    
    print('===============================================================================')
    if (solver_status == 'ok'):
        if obj2: # TODO: poner un for cada objetivo porque tiene que ser lo más general posible
            print(f'Objective SEQUENCES: {newrow[0]}')
            print(f'Objective {obj2}: {newrow[1]}')
        else:
            print(f'Objective SEQUENCES: {newrow[0]}')
            print(f'Objective CC_diff: {newrow[1]}')
            print(f'Objective LOC_diff: {newrow[2]}')
        print('Sequences selected:')
        for s in concrete.S:
            print(f"x[{s}] = {concrete.x[s].value}")
    print('===============================================================================')




def add_info_to_list(concrete: pyo.ConcreteModel, output_data: list, solver_status: str, obj1: str, obj2: str, newrow: list):
    """ Write results and a vertical list of selected sequences in a given file """
    
    
    if (solver_status == 'ok'):
        output_data.append(f'{obj1.__name__}: {newrow[0]}')
        output_data.append(f'{obj2.__name__}: {newrow[1]}')
        output_data.append('Sequences selected:')
        for s in concrete.S:
            output_data.append(f"x[{s}] = {concrete.x[s].value}")
    output_data.append('===============================================================================')








def generate_three_weights(n_divisions=6, theta_index=0, phi_index=0) -> tuple[int, int, int]:
    """
    Generates subdivisions in spherical coordinates for an octant.
        
    Args:
        n_divisions (int): Number of divisions in each plane (XY, XZ, YZ).
        
    Returns:
        dict: Dictionary with subdivisions in spherical coordinates.
    """
    # Crear ángulos según las divisiones
    angles = np.linspace(0, np.pi/2, n_divisions + 1)  # divisiones del plano
    subdivisions = {i: angles[i] for i in range(n_divisions+1)}
    
    w1, w2, w3 =  [math.sin(subdivisions[theta_index])*math.cos(subdivisions[phi_index]),
                  math.sin(subdivisions[theta_index])*math.sin(subdivisions[phi_index]),
                  math.cos(subdivisions[theta_index])]
    
    return w1, w2, w3


def generate_two_weights(n_divisions=6, theta_index=0) -> tuple[int, int, int]:
    """
    Generates subdivisions in polar coordinates for a cuadrant.
        
    Args:
        n_divisions (int): Number of divisions in each plane (XY, XZ, YZ).
        
    Returns:
        dict: Dictionary with subdivisions in spherical coordinates.
    """
    # Crear ángulos según las divisiones
    angles = np.linspace(0, np.pi/2, n_divisions + 1)  # divisiones del plano
    subdivisions = {i: angles[i] for i in range(n_divisions+1)}
    
    w1, w2 =  [math.sin(subdivisions[theta_index]), math.cos(subdivisions[theta_index])]
    
    return w1, w2


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
                    salida_pdf = os.path.join(carpeta_salida_proyecto, f"{carpeta_clase_metodo}.pdf")
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

