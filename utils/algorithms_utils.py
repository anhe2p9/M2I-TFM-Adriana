import math

import pyomo.environ as pyo # ayuda a definir y resolver problemas de optimización
import pyomo.dataportal as dp # permite cargar datos para usar en esos modelos de optimización

import matplotlib.pyplot as plt
from ILP_CC_reducer.models import MultiobjectiveILPmodel

import numpy as np

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
    angles = np.linspace(0.1, np.pi/2, n_divisions + 1)  # divisiones del plano
    subdivisions = {i: angles[i] for i in range(n_divisions+1)}
    
    w1, w2, w3 =  [math.sin(subdivisions[theta_index])*math.cos(subdivisions[phi_index]),
                  math.sin(subdivisions[theta_index])*math.sin(subdivisions[phi_index]),
                  math.cos(subdivisions[theta_index])]
    
    return w1, w2, w3


def generate_two_weights(n_divisions=6, theta_index=0) -> tuple[int, int, int]:
    """
    Generates subdivisions in polar coordinates for a cuadrant.
        
    Args:
        n_divisions (int): Number of divisions in each axe (X, Y).
        
    Returns:
        dict: Dictionary with subdivisions in spherical coordinates.
    """
    # Crear ángulos según las divisiones
    angles = np.linspace(0.1, np.pi/2, n_divisions + 1)  # divisiones del plano
    subdivisions = {i: angles[i] for i in range(n_divisions+1)}
    
    w1, w2 =  [math.sin(subdivisions[theta_index]), math.cos(subdivisions[theta_index])]
    
    return w1, w2