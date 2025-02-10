import numpy as np
import math

import sys

import pyomo.environ as pyo # ayuda a definir y resolver problemas de optimización
import pyomo.dataportal as dp # permite cargar datos para usar en esos modelos de optimización

from ILP_CC_reducer.models.multiobjILPmodel import MultiobjectiveILPmodel


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
    # results = solver.solve(concrete)
    solver.solve(concrete)
    
    # print(results)
    # num_variables = sum(len(variable) for variable in concrete.component_objects(pyo.Var, active=True))
    # print(f"There are {num_variables} variables\n")
    # print("==========================================================================================================\n")


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
    
    return concrete, newrow



def process_weighted_model_2obj(model: pyo.AbstractModel, data: dp.DataPortal, w1 ,w2, obj_selected):
    
    multiobj_model = MultiobjectiveILPmodel()
    
    if hasattr(model, 'obj'):
        model.del_component('obj')
    model.add_component('obj', pyo.Objective(rule=lambda m: multiobj_model.weightedSum2obj(m, w1, w2, obj_selected)))
    
    
    
    concrete = model.create_instance(data) # para crear una instancia de modelo y hacerlo concreto
    solver = pyo.SolverFactory('cplex')
    results = solver.solve(concrete)
    # solver.solve(concrete)
    
    # print(results)
    # num_variables = sum(len(variable) for variable in concrete.component_objects(pyo.Var, active=True))
    # print(f"There are {num_variables} variables\n")
    # print("==========================================================================================================\n")


    sequences_sum = sum(concrete.x[i].value for i in concrete.S if i != 0)
    
    if obj_selected == 'LOC':
        xLOC = [concrete.loc[i] for i in concrete.S if concrete.x[i].value == 1]
        zLOC = [concrete.loc[j] for j,ii in concrete.N if concrete.z[j,ii].value == 1]
        
        max_selected = abs(max(xLOC) - max(zLOC))
        min_selected = min(concrete.loc[i] for i in concrete.S if concrete.x[i].value == 1)
    
    elif obj_selected == 'CC':

        xCC = [concrete.nmcc[i] for i in concrete.S if concrete.x[i].value == 1]
        zCC = [concrete.ccr[j,ii] for j,ii in concrete.N if concrete.z[j,ii].value == 1]
        
        
        max_selected = abs(max(xCC) - max(zCC))
        min_selected = min(concrete.nmcc[i] for i in concrete.S if concrete.x[i].value == 1)
        
    else:
        sys.exit("Second objective parameter must be one between 'LOC' and 'CC'.")
        
    obj2_dif = abs(max_selected - min_selected)
    
    newrow = [round(w1,2),round(w2,2),sequences_sum,obj2_dif]
    
    print('===============================================================================')
    if (results.solver.status == 'ok'):
        print('Objective SEQUENCES: ', sequences_sum +1)
        print(f'Objective {obj_selected}: {obj2_dif}')
        print('Sequences selected:')
        for s in concrete.S:
            print(f"x[{s}] = {concrete.x[s].value}")
    print('===============================================================================')
    
    return concrete, newrow
