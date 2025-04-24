import numpy as np
import math

import sys
import os
from typing import Any
import csv
from pathlib import Path
import pandas as pd

import pyomo.environ as pyo # ayuda a definir y resolver problemas de optimización
import pyomo.dataportal as dp # permite cargar datos para usar en esos modelos de optimización

from ILP_CC_reducer.models.multiobjILPmodel import MultiobjectiveILPmodel


def modify_component(mobj_model: pyo.AbstractModel, component: str, new_value: pyo.Any):
    if hasattr(mobj_model.model, component):
        mobj_model.model.del_component(component)
    mobj_model.model.add_component(component, new_value)

def concrete_and_solve_model(mobj_model: pyo.AbstractModel, instance: dp.DataPortal):
    concrete = mobj_model.model.create_instance(instance)
    solver = pyo.SolverFactory('cplex')
    result = solver.solve(concrete)
    return concrete, result



def calculate_results(concrete_model: pyo.ConcreteModel, obj: str): # TODO: esto hay que generalizarlo para que sirva para cualquiera de los tres objetivos, o para dos en caso de que sean dos
    sequences_sum = sum(concrete_model.x[i].value for i in concrete_model.S)
            
    if obj == 'LOC':
        max_xLOC = max(concrete_model.loc[i] for i in concrete_model.S if concrete_model.x[i].value == 1)
        max_zLOC = max(concrete_model.loc[j] for j,ii in concrete_model.N if concrete_model.z[j,ii].value == 1)
        
        max_selected = abs(max_xLOC - max_zLOC)
        min_selected = min(concrete_model.loc[i] for i in concrete_model.S if concrete_model.x[i].value == 1)
    
    elif obj == 'CC':

        max_xCC = max(concrete_model.nmcc[i] for i in concrete_model.S if concrete_model.x[i].value == 1)
        max_zCC = max(concrete_model.ccr[j,ii] for j,ii in concrete_model.N if concrete_model.z[j,ii].value == 1)
        
        
        max_selected = abs(max_xCC - max_zCC)
        min_selected = min(concrete_model.nmcc[i] for i in concrete_model.S if concrete_model.x[i].value == 1)
        
    else:
        sys.exit("Second objective parameter must be one between 'LOC' and 'CC'.")
        
    obj2_dif = abs(max_selected - min_selected)
    
    newrow = [sequences_sum, obj2_dif]
    
    return newrow


def print_result_and_sequences(solver_status: str, newrow: list, obj2: str, concrete: pyo.ConcreteModel):
    print('===============================================================================')
    if (solver_status == 'ok'):
        print(f'Objective SEQUENCES: {newrow[0]}')
        print(f'Objective {obj2}: {newrow[1]}')
        print('Sequences selected:')
        for s in concrete.S:
            print(f"x[{s}] = {concrete.x[s].value}")
    print('===============================================================================')


def write_results_and_sequences_to_file(file: str, solver_status: str, newrow: list, obj2: str, concrete: pyo.ConcreteModel):
    file.write('-------------------------------------------------------------------------------\n')
    if (solver_status == 'ok'):
        file.write(f'Objective SEQUENCES: {newrow[0]}\n')
        file.write(f'Second objective value ({obj2}): {newrow[1]}\n')
        file.write('Sequences selected:\n')
        for s in concrete.S:
            file.write(f"x[{s}] = {concrete.x[s].value}\n")
    file.write('===============================================================================\n')
    








def extract_optimal_tuples(file_path):
    sheets = ["General", "MaxTimeSOLVED"]
    result_set = set()
    
    for sheet in sheets:
        df = pd.read_excel(file_path, sheet_name=sheet, engine='openpyxl')
        
        if df.shape[1] < 33:  # Verifica que haya al menos 33 columnas (AG es la columna 33)
            print(f"Advertencia: La hoja '{sheet}' no tiene suficientes columnas.")
            continue
        
        df = df.dropna(subset=['terminationCondition'])  # Elimina filas donde 'AG' es NaN
        
        for _, row in df.iterrows():
            if str(row['terminationCondition']).strip().lower() == 'optimal':
                result_set.add((row.iloc[0], row.iloc[1], row.iloc[2]))
    
    return result_set




def generate_weights(n_divisions=6, theta_index=0, phi_index=0) -> tuple[int, int, int]:
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


def generate_weights_2obj(n_divisions=6, theta_index=0) -> tuple[int, int, int]:
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



def process_weighted_model(model: pyo.AbstractModel, data: dp.DataPortal, w1 ,w2, w3):
    
    multiobj_model = MultiobjectiveILPmodel()
    
    if hasattr(model, 'obj'):
        model.del_component('obj')
    model.add_component('obj', pyo.Objective(rule=lambda m: multiobj_model.weightedSum(m, w1, w2, w3)))
    
    
    
    concrete = model.create_instance(data) # para crear una instancia de modelo y hacerlo concreto
    solver = pyo.SolverFactory('cplex')
    results = solver.solve(concrete)
    # solver.solve(concrete)
    
    # print(results)
    # num_variables = sum(len(variable) for variable in concrete.component_objects(pyo.Var, active=True))
    # print(f"There are {num_variables} variables\n")
    # print("==========================================================================================================\n")

    # print(f"Sequences values: {[concrete.x[i].value for i in concrete.S if i != 0]}")
    sequences_sum = sum(concrete.x[i].value for i in concrete.S if i != 0)
    
    xLOC = [concrete.loc[i] for i in concrete.S if concrete.x[i].value == 1]
    zLOC = [concrete.loc[j] for j,ii in concrete.N if concrete.z[j,ii].value == 1]
    
    maxLOCselected = abs(max(xLOC) - max(zLOC))
    minLOCselected = min(concrete.loc[i] for i in concrete.S if concrete.x[i].value == 1)
    LOCdif = abs(maxLOCselected - minLOCselected)
    
    xCC = [concrete.nmcc[i] for i in concrete.S if concrete.x[i].value == 1]
    zCC = [concrete.ccr[j,ii] for j,ii in concrete.N if concrete.z[j,ii].value == 1]
    
    
    maxCCselected = abs(max(xCC) - max(zCC))
    minCCselected = min(concrete.nmcc[i] for i in concrete.S if concrete.x[i].value == 1)
    CCdif = abs(maxCCselected - minCCselected)
    
    
    newrow = [round(w1,2),round(w2,2),round(w3,2),sequences_sum,LOCdif,CCdif]
    
    # TODO: añadir generación de CSVs con los resultados (hay algún método ya hecho creo que sería solo llamarlo)
    
    return concrete, newrow, results







def generate_results_csv_file(instance: str, concrete: pyo.ConcreteModel, results: Any, csv_list: list[Any]):
    
    data_row = []

    _, folder_name = os.path.split(instance)
    class_name, method_name = folder_name.split('_')
    data_row.append(class_name)
    data_row.append(method_name)
            
    initial_complexity = concrete.nmcc[0]
    data_row.append(initial_complexity)
    
    solution = [concrete.x[s].index() for s in concrete.S if concrete.x[s].value == 1 and concrete.x[s].index() != 0]
    data_row.append(solution)
    
    extractions = len(solution)
    data_row.append(extractions)
    
    reduction_complexity = sum(concrete.ccr[j,i] for j,i in concrete.N if concrete.z[j,i].value == 1)
    data_row.append(reduction_complexity)
    
    final_complexity = initial_complexity - reduction_complexity
    data_row.append(final_complexity)
    
    
    LOC_for_each_sequence = [concrete.loc[s] for s in concrete.S if concrete.x[s].value == 1 and concrete.x[s].index() != 0]
    if len(LOC_for_each_sequence) > 0:
        minExtractedLOC = min(LOC_for_each_sequence)
        data_row.append(minExtractedLOC)
        maxExtractedLOC = max(LOC_for_each_sequence)
        data_row.append(maxExtractedLOC)
        meanExtractedLOC = float(np.mean(LOC_for_each_sequence))
        data_row.append(meanExtractedLOC)
        totalExtractedLOC = sum(LOC_for_each_sequence)
        data_row.append(totalExtractedLOC)
    else:
        for i in range(4):
            data_row.append("")
    
    
    CC_for_each_sequence = [concrete.nmcc[s] for s in concrete.S if concrete.x[s].value == 1 and concrete.x[s].index() != 0]
    if len(CC_for_each_sequence) > 0:
        minExtractedCC = min(CC_for_each_sequence)
        data_row.append(minExtractedCC)
        maxExtractedCC = max(CC_for_each_sequence)
        data_row.append(maxExtractedCC)
        meanExtractedCC = float(np.mean(CC_for_each_sequence))
        data_row.append(meanExtractedCC)
        totalExtractedCC = sum(CC_for_each_sequence)
        data_row.append(totalExtractedCC)
    else:
        for i in range(4):
            data_row.append("")
    
    data_row.append(str(results.solver.status))
    data_row.append(results.solver.time)
           
    
    print(data_row)
    csv_list.append(data_row)
    
    print("============================================================================================================")
    
    
    return csv_list




def create_csv_from_results(concrete: pyo.ConcreteModel, results: Any):
    
    csv_data = [["class", "method", "initialComplexity", "solution", "extractions", 
             "reductionComplexity", "finalComplexity",
             "minExtractedLOC", "maxExtractedLOC", "meanExtractedLOC", "totalExtractedLOC", 
             "minReductionOfCC", "maxReductionOfCC", "meanReductionOfCC", "totalReductionOfCC", 
             "modelStatus", "executionTime"]]


    instance_folder = "original_code_data"
    
    for subfolder in sorted(os.listdir(instance_folder)):
        subfolder_path = os.path.join(instance_folder, subfolder)
        if os.path.isdir(subfolder_path):
            print(f"Processing Class_Method: {subfolder}")
            results_csv = generate_results_csv_file(Path(subfolder_path), concrete, results, csv_data)
        
    
    # Escribir datos en un archivo CSV
    with open("results.csv", mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(results_csv)
    
    print("Archivo CSV creado correctamente.")
