import math

import pyomo.environ as pyo # ayuda a definir y resolver problemas de optimización
import pyomo.dataportal as dp # permite cargar datos para usar en esos modelos de optimización

import matplotlib.pyplot as plt
from ILP_CC_reducer.models import MultiobjectiveILPmodel

import numpy as np
import pandas as pd

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








def generate_three_weights(n_divisions=6, theta_index=0, phi_index=0) -> tuple[float, float, float]:
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


def generate_two_weights(n_divisions=6, theta_index=0) -> tuple[float, float]:
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


def write_complete_info(concrete: pyo.ConcreteModel, results, data):
    """ Completes a csv with all solution data """

    complete_data_row = []

    objective_map = {
        'extractions': model.extractions_objective,
        'cc': model.cc_difference_objective,
        'loc': model.loc_difference_objective
    }

    """ Number of sequences """
    num_sequences = len([s for s in concrete.S])
    complete_data_row.append(num_sequences)

    """ Number of variables """
    num_vars_utilizadas = results.Problem[0].number_of_variables
    complete_data_row.append(num_vars_utilizadas)

    """ Number of constraints """
    num_constraints = sum(len(constraint) for constraint in concrete.component_objects(pyo.Constraint, active=True))
    complete_data_row.append(num_constraints)

    """ Initial complexity """
    initial_complexity = concrete.nmcc[0]
    complete_data_row.append(initial_complexity)

    if (results.solver.status == 'ok') and (results.solver.termination_condition == 'optimal'):
        """ Solution """
        solution = [concrete.x[s].index() for s in concrete.S if concrete.x[s].value == 1 and s != 0]
        complete_data_row.append([(concrete.x[s].index(),
                                   round(pyo.value(concrete.nmcc[s] - sum(concrete.ccr[j, s] * concrete.z[j, s]
                                                                          for j,k in concrete.N if k == s))),
                                   round(pyo.value(concrete.loc[s] - sum((concrete.loc[j] - 1) * concrete.z[j, k]
                                                                         for j,k in concrete.N if k == s))))
                                  for s in concrete.S if concrete.x[s].value == 1])

        """ Offsets """
        df_csv = pd.read_csv(data["offsets"], header=None, names=["index", "start", "end"])

        # Filter by index in solution str list and obtain values
        solution_str = [str(i) for i in solution]
        offsets_list = df_csv[df_csv["index"].isin(solution_str)][["start", "end"]].values.tolist()

        offsets_list = [[int(start), int(end)] for start, end in offsets_list]
        complete_data_row.append(offsets_list)

        """ Extractions """
        extractions = round(pyo.value(objective_map['extractions'](concrete)))
        complete_data_row.append(extractions)

        """ Not nested solution """
        not_nested_solution = [concrete.x[s].index() for s, k in concrete.N if k == 0 and concrete.z[s, k].value != 0]
        complete_data_row.append(not_nested_solution)

        """ Not nested extractions """
        not_nested_extractions = len(not_nested_solution)
        complete_data_row.append(not_nested_extractions)

        """ Nested solution """
        nested_solution = {}

        for s, k in concrete.N:
            if concrete.z[s, k].value != 0 and k in solution:
                if k not in nested_solution:
                    nested_solution[k] = []  # Crear una nueva lista para cada k
                nested_solution[k].append(concrete.x[s].index())

        if len(nested_solution) != 0:
            complete_data_row.append(nested_solution)
        else:
            complete_data_row.append("")


        """ Nested extractions """
        nested_extractions = sum(len(v) for v in nested_solution.values())
        complete_data_row.append(nested_extractions)


        """ Reduction of complexity """
        cc_reduction = [(concrete.ccr[j, k] * concrete.z[j, k].value) for j, k in concrete.N if
                        k == 0 and concrete.z[j, k].value != 0]

        reduction_complexity = sum(cc_reduction)
        complete_data_row.append(reduction_complexity)


        """ Final complexity """
        final_complexity = initial_complexity - reduction_complexity
        complete_data_row.append(final_complexity)

        """ Minimum extracted LOC, Maximum extracted LOC, Mean extracted LOC, Total extracted LOC, Nested LOC """
        loc_for_each_sequence = [(concrete.loc[j] * concrete.z[j, k].value) for j, k in concrete.N if
                                 k == 0 and concrete.z[j, k].value != 0]
        if len(loc_for_each_sequence) > 0:
            min_extracted_loc = min(loc_for_each_sequence)
            complete_data_row.append(min_extracted_loc)
            max_extracted_loc = max(loc_for_each_sequence)
            complete_data_row.append(max_extracted_loc)
            mean_extracted_loc = round(float(np.mean(loc_for_each_sequence)))
            complete_data_row.append(mean_extracted_loc)
            total_extracted_loc = sum(loc_for_each_sequence)
            complete_data_row.append(total_extracted_loc)
            # NESTED LOC
            nested_loc = {}
            for v in nested_solution.values():
                for n in v:
                    nested_loc[n] = concrete.loc[n]
            if len(nested_loc) > 0:
                complete_data_row.append(nested_loc)
            else:
                complete_data_row.append("")
        else:
            for _ in range(5):
                complete_data_row.append("")



        """ Min reduction of CC, Max reduction of CC, Mean reduction of CC, Total reduction of CC, Nested CC """
        if len(cc_reduction) > 0:
            min_extracted_cc = min(cc_reduction)
            complete_data_row.append(min_extracted_cc)
            max_extracted_cc = max(cc_reduction)
            complete_data_row.append(max_extracted_cc)
            mean_extracted_cc = round(float(np.mean(cc_reduction)))
            complete_data_row.append(mean_extracted_cc)
            total_extracted_cc = initial_complexity - final_complexity
            complete_data_row.append(total_extracted_cc)
            # NESTED CC
            nested_cc = {}
            for v in nested_solution.values():
                for n in v:
                    nested_cc[n] = concrete.nmcc[n]
            if len(nested_cc) > 0:
                complete_data_row.append(nested_cc)
            else:
                complete_data_row.append("")
        else:
            for _ in range(5):
                complete_data_row.append("")

        """ Min extracted Params, Max extracted Params, Mean extracted Params, Total extracted Params """
        params_for_each_sequence = [(concrete.params[j] * concrete.z[j, k].value) for j, k in concrete.N if
                                    concrete.z[j, k].value != 0]
        if len(params_for_each_sequence) > 0:
            min_extracted_params = min(params_for_each_sequence)
            complete_data_row.append(min_extracted_params)
            max_extracted_params = max(params_for_each_sequence)
            complete_data_row.append(max_extracted_params)
            mean_extracted_params = round(float(np.mean(params_for_each_sequence)))
            complete_data_row.append(mean_extracted_params)
            total_extracted_params = sum(params_for_each_sequence)
            complete_data_row.append(total_extracted_params)
        else:
            for _ in range(4):
                complete_data_row.append("")
    else:
        for _ in range(23):
            complete_data_row.append("")

    """ Termination condition """
    complete_data_row.append(str(results.solver.termination_condition))

    """ Execution time """
    complete_data_row.append(results.solver.time)

    return complete_data_row