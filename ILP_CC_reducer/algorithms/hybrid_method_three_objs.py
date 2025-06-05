import pyomo.environ as pyo # ayuda a definir y resolver problemas de optimización
import pyomo.dataportal as dp # permite cargar datos para usar en esos modelos de optimización

from typing import List, Tuple, Optional

import time
import numpy as np
import pandas as pd


from ILP_CC_reducer.algorithm.algorithm import Algorithm
from ILP_CC_reducer.models import MultiobjectiveILPmodel
from ILP_CC_reducer.models import ILPmodelRsain

import algorithms_utils


Box3D = Tuple[float, float, float]
Point3D = Tuple[float, float, float]
multiobjective_model = MultiobjectiveILPmodel()
uniobjective_model = ILPmodelRsain()



class HybridMethodForThreeObj(Algorithm):

    @staticmethod
    def get_name() -> str:
        return 'Epsilon Constraint algorithm with 3 obj'
    
    @staticmethod
    def get_description() -> str:
        return "It obtains supported and non-supported ILP solutions."

    @staticmethod
    def execute(data_dict: dict, tau: int, objectives_list: list= None):

        data = data_dict['data']

        start_total = time.time()

        if not objectives_list:  # if there is no order, the order will be [SEQ,CC,LOC]
            objectives_list = [multiobjective_model.sequences_objective,
                               multiobjective_model.cc_difference_objective,
                               multiobjective_model.loc_difference_objective]

        add_items_to_multiobjective_model(tau)
        concrete = uniobjective_model.create_instance(data)

        nadir_dict = {multiobjective_model.sequences_objective: len(concrete.S)+1,
                      multiobjective_model.cc_difference_objective: concrete.nmcc[0]+1 ,
                      multiobjective_model.loc_difference_objective: concrete.loc[0]+1}

        nadir_point = []
        for obj in objectives_list:
            nadir_point.append(nadir_dict[obj])

        initial_box = tuple(nadir_point)

        solutions_set, concrete, output_data, complete_data = epsilon_constraint_with_full_p_split(data_dict, objectives_list,
                                                                                     initial_box, max_solutions=20)

        csv_data = [[obj.__name__ for obj in objectives_list]]

        for sol in solutions_set:
            csv_data.append(list(sol))

        end_total = time.time()

        output_data.append("==========================================================================================")
        output_data.append("==========================================================================================")
        output_data.append(f"Total execution time: {end_total - start_total:.2f}")

        return csv_data, concrete, output_data, complete_data


def add_items_to_multiobjective_model(tau: int):
    multiobjective_model.tau = pyo.Param(within=pyo.NonNegativeReals, initialize=tau, mutable=False)  # Threshold MO
    uniobjective_model.tau = pyo.Param(within=pyo.NonNegativeReals, initialize=tau, mutable=False)  # Threshold UO

    multiobjective_model.lambda2 = pyo.Param(initialize=0.001, mutable=True)  # Lambda parameter
    multiobjective_model.lambda3 = pyo.Param(initialize=0.001, mutable=True)  # Lambda parameter


def solve_epsilon_constraint(data: dp.DataPortal, objectives_list: list, box: tuple):
    output_data = []

    obj1, obj2, obj3 = objectives_list

    algorithms_utils.modify_component(multiobjective_model, 'obj', pyo.Objective(
        rule=lambda m: multiobjective_model.epsilon_objective(m, obj1, obj2, obj3)))  # min f1(x) - (lambda2 * l2 + lambda3 * l3)

    add_boxes_constraints(box, objectives_list)

    concrete, result = algorithms_utils.concrete_and_solve_model(multiobjective_model, data)

    prefijo = "boxes_constraint"

    for name, constraint in concrete.component_map(pyo.Constraint, active=True).items():
        if name.startswith(prefijo) or name.startswith('epsilonConstraint'):
            print(f"Restricción: {name}")
            for index in constraint:
                print(f"  {index}: {constraint[index].expr}")


    if (result.solver.status == 'ok') and (result.solver.termination_condition == 'optimal') :
        newrow = tuple(round(pyo.value(obj(concrete))) for obj in objectives_list)  # Results for CSV file
        ordered_newrow = tuple(round(pyo.value(obj(concrete))) for obj in [multiobjective_model.sequences_objective,
                                                                         multiobjective_model.cc_difference_objective,
                                                                         multiobjective_model.loc_difference_objective])

        output_data.append('===============================================================================')
        if result.solver.status == 'ok':
            for i, obj in enumerate(objectives_list):
                output_data.append(f'Objective {obj.__name__}: {newrow[i]}')
            output_data.append('Sequences selected:')
            for s in concrete.S:
                output_data.append(f"x[{s}] = {concrete.x[s].value}")

    else:
        newrow = None
        ordered_newrow = None

    cplex_time = result.solver.time

    return newrow, concrete, result, cplex_time, output_data, ordered_newrow


def add_boxes_constraints(box: tuple, objectives_list: list):
    obj1, obj2, obj3 = objectives_list

    for i in range(3):
        if hasattr(multiobjective_model, f'box_u{i+1}_constraint'):
            multiobjective_model.del_component(f'box_u{i+1}_constraint')

    multiobjective_model.box_u1_constraint = pyo.Constraint(rule=lambda m: obj1(m) <= box[0] - 1)
    multiobjective_model.box_u2_constraint = pyo.Constraint(rule=lambda m: obj2(m) <= box[1] - 1)
    multiobjective_model.box_u3_constraint = pyo.Constraint(rule=lambda m: obj3(m) <= box[2] - 1)


def epsilon_constraint_with_full_p_split(data_dict, objectives_list, initial_box: tuple, max_solutions=100):
    output_data = []

    complete_data = [["numberOfSequences", "numberOfVariables", "numberOfConstraints",
                      "initialComplexity", "solution (index,CC,LOC)", "offsets", "extractions",
                      "not_nested_solution", "not_nested_extractions",
                      "nested_solution", "nested_extractions",
                      "reductionComplexity", "finalComplexity",
                      "minExtractedLOC", "maxExtractedLOC", "meanExtractedLOC", "totalExtractedLOC", "nestedLOC",
                      "minReductionOfCC", "maxReductionOfCC", "meanReductionOfCC", "totalReductionOfCC", "nestedCC",
                      "minExtractedParams", "maxExtractedParams", "meanExtractedParams", "totalExtractedParams",
                      "terminationCondition", "executionTime"]]

    solutions_set = set()  # Non-dominated solutions set
    s_ordered = set()

    boxes = [initial_box]  # tuple list (u1, u2, u3)

    while boxes and len(solutions_set) < max_solutions:

        print(
            "=============================================================================================================")

        print(f"Processing e-constraint with boxes: {boxes}.")

        actual_box = boxes.pop(0)

        # Usar e[1] y e[2] como valores de epsilon para restricciones f2 y f3
        solution, concrete, result, cplex_time, new_output_data, ordered_newrow = solve_epsilon_constraint(data_dict['data'],
                                                                                           objectives_list, actual_box)

        if solution:
            solutions_set.add(solution)  # Add solution to solutions set
            s_ordered.add(ordered_newrow)

            output_data = output_data + new_output_data
            output_data.append(f"CPLEX time: {cplex_time}.")

            complete_data_new_row = write_complete_info(concrete, result, data_dict)
            complete_data.append(complete_data_new_row)

            print(f"New solution: {solution}.")

            # Partimos la caja con respecto a la solución real obtenida
            new_boxes = full_p_split_3d(actual_box, solution)

            for i, box in enumerate(new_boxes):
                if box is not None:
                    boxes.append(box)

            for i,box in enumerate(boxes):
                if inside(solution,box):
                    boxes.pop(i)
                    new_boxes = full_p_split_3d(box, solution)
                    for j, new_box in enumerate(new_boxes):
                        if box is not None:
                            boxes.append(new_box)

            print(f"GENERAL BOXES LIST: {boxes}.")
            non_dominated_boxes = filter_contained_boxes(boxes)
            print(f"NON DOMINATED BOXES: {non_dominated_boxes}.")
            boxes = non_dominated_boxes

        else:
            # No se encontró solución → caja descartada
            output_data.append("======================================================================================")
            output_data.append(f"SOLUTION NOT FOUND, CPLEX TIME: {cplex_time}")
            output_data.append("======================================================================================")

            continue

    print(f"Solution set: {s_ordered}.")

    return solutions_set, concrete, output_data, complete_data


def full_p_split_3d(box: Box3D, z: tuple) -> List[Optional[Box3D]]:
    l = (0,0,0)
    u = box  # original box: l = (l1, l2, l3), u = (u1, u2, u3)
    boxes = []

    for i in range(3):  # i = 0 (x), 1 (y), 2 (z)

        new_u = list(u)
        for j in range(i):
            new_u[j] = u[j]
        new_u[i] = max(z[i], l[i])  # cut by x_i < z_i

        is_empty = any(u[k] < z[k] for k in range(i+1, 3)) or new_u[i] <= l[i]

        boxes.append(None if is_empty else tuple(new_u))

    return boxes


def dominates(a: tuple, b: tuple) -> bool:
    """
    Returns True if point a dominates b.
    """
    return all(a[i] <= b[i] for i in range(3)) and any(a[i] < b[i] for i in range(3))


def inside(a: tuple, b: tuple) -> bool:
    """
    Returns True if point 'a' is completely inside 'b'.
    """
    return all(a[i] < b[i] for i in range(3))


def filter_contained_boxes(boxes: List[Point3D]) -> List[Point3D]:
    """
    Elimina las cajas cuyo punto superior está contenido (componente a componente)
    en otra caja de la lista.
    """
    non_dominated = []

    for i, box_i in enumerate(boxes):
        dominated = False
        for j, box_j in enumerate(boxes):
            if i == j:
                continue
            # Si box_i está estrictamente contenida en box_j
            if dominates(box_i, box_j):
                dominated = True
                print(f"Discarded box: {box_i} because it is inside box {box_j}.")
                break
        if not dominated:
            non_dominated.append(box_i)

    return non_dominated


def write_complete_info(concrete: pyo.ConcreteModel, results, data):
    """ Completes a csv with all solution data """

    complete_data_row = []

    objective_map = {
        'sequences': multiobjective_model.sequences_objective,
        'cc': multiobjective_model.cc_difference_objective,
        'loc': multiobjective_model.loc_difference_objective
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
        extractions = round(pyo.value(objective_map['sequences'](concrete)))
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