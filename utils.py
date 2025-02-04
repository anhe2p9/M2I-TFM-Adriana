import numpy as np
import math

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



def process_weighted_model(model: pyo.AbstractModel, data: dp.DataPortal, w1 ,w2, w3):
    
    multiobj_model = MultiobjectiveILPmodel()
    
    if hasattr(model, 'obj'):
        model.del_component('obj')  # Eliminar el componente existente
        model.add_component('obj', pyo.Objective(rule=lambda m: multiobj_model.weightedSum(m, w1, w2, w3)))
    else:
        model.obj = pyo.Objective(rule=lambda m: multiobj_model.weightedSum(m,w1, w2, w3))
    
    concrete = model.create_instance(data) # para crear una instancia de modelo y hacerlo concreto
    solver = pyo.SolverFactory('cplex')
    # results = solver.solve(concrete)
    solver.solve(concrete)
    

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





