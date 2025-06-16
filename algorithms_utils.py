import numpy as np
import math

# import sys
import os
from typing import Any
import csv
from pathlib import Path

import pyomo.environ as pyo # ayuda a definir y resolver problemas de optimización
import pyomo.dataportal as dp # permite cargar datos para usar en esos modelos de optimización

# from ILP_CC_reducer.models.multiobjILPmodel import MultiobjectiveILPmodel


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

    if not os.path.exists(f"output/{project_name}/{result_name}"):
        os.makedirs(f"output/{project_name}/{result_name}")

    # Save model in a LP file
    if concrete:
        concrete.write(f'output/{project_name}/{result_name}/{method_name}.lp',
                       io_options={'symbolic_solver_labels': True})
        print("Model correctly saved in a LP file.")

    # Save data in a CSV file
    filename = f"output/{project_name}/{result_name}/{method_name}_results.csv"
            
    if os.path.exists(filename):
        os.remove(filename)
            
    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(csv_info)
        print("CSV file correctly created.")
    
    # Save output in a TXT file
    if output_data:
        with open(f"output/{project_name}/{result_name}/{method_name}_output.txt", "w") as f:
            for linea in output_data:
                f.write(linea + "\n")
            print("Output correctly saved in a TXT file.")

    # Save output in a TXT file
    if complete_data:
        with open(f"output/{project_name}/{result_name}/{method_name}_complete_data.csv",
                  mode="w", newline="", encoding="utf-8") as complete_csv:
            writer = csv.writer(complete_csv)
            writer.writerows(complete_data)
            print("Complete CSV file correctly created.")

    # Save nadir point in a csv file
    if nadir:
        with open(f"output/{project_name}/{result_name}/{method_name}_nadir.csv",
                  mode="w", newline="", encoding="utf-8") as nadir_csv:
            writer = csv.writer(nadir_csv)
            writer.writerows(nadir)
            print("Nadir CSV file correctly created.")



def calculate_results(concrete_model: pyo.ConcreteModel, obj_selected: str = None):
    """ Calculate sequences selected, and the selected second objective difference """
    sequences_sum = sum(concrete_model.x[i].value for i in concrete_model.S)
            
    if obj_selected is not None:
        obj_diff = obtain_obj_diff(concrete_model, obj_selected)
        newrow = [sequences_sum, obj_diff]
    else:
        cc_diff = obtain_obj_diff(concrete_model, 'CC')
        loc_diff = obtain_obj_diff(concrete_model, 'LOC')
        newrow = [sequences_sum, cc_diff, loc_diff]   
    return newrow


def obtain_obj_diff(concrete_model: pyo.ConcreteModel, obj: str):
    """ Obtain LOC diff in case that objective is selected, or CC diff in case that objective is selected. """
    
    # Define parameters
    paremeter_selected_x = concrete_model.loc if obj == 'LOC' else concrete_model.nmcc
    paremeter_selected_z = concrete_model.loc if obj == 'LOC' else concrete_model.ccr
    
    # Select Z index depending on the objective
    get_z_index = (lambda j, ii: paremeter_selected_z[j]) if obj == 'LOC' else (lambda j, ii: paremeter_selected_z[j, ii])
    
    # Obtain max
    max_xLOC = max(paremeter_selected_x[i] for i in concrete_model.S if concrete_model.x[i].value == 1)
    max_zLOC = max(get_z_index(j,ii) for j,ii in concrete_model.N if concrete_model.z[j,ii].value == 1)
    max_selected = abs(max_xLOC - max_zLOC) # Obtain the max extracted property
    
    # Obtain min
    min_selected = min(paremeter_selected_x[i] for i in concrete_model.S if concrete_model.x[i].value == 1)
    
    # Obtain diff between max and min
    obj_diff = abs(max_selected - min_selected)
    
    return obj_diff


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


def write_results_and_sequences_to_file(concrete: pyo.ConcreteModel, file: str, solver_status: str, newrow: list, obj2: str=None):
    """ Write results and a vertical list of selected sequences in a given file """
    
    file.write('-------------------------------------------------------------------------------\n')
    if (solver_status == 'ok'):
        if obj2: # TODO: aquí también poner un for para los objetivos
            file.write(f'Objective SEQUENCES: {newrow[0]}\n')
            file.write(f'Objective {obj2}: {newrow[1]}\n')
        else:
            file.write(f'Objective SEQUENCES: {newrow[0]}\n')
            file.write(f'Objective CC_diff: {newrow[1]}\n')
            file.write(f'Objective LOC_diff: {newrow[2]}\n')
        file.write('Sequences selected:\n')
        for s in concrete.S:
            file.write(f"x[{s}] = {concrete.x[s].value}\n")
    file.write('===============================================================================\n')







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
