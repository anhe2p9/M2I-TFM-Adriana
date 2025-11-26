import pyomo.environ as pyo # ayuda a definir y resolver problemas de optimización
import pyomo.dataportal as dp # permite cargar datos para usar en esos modelos de optimización

from typing import List, Tuple, Optional

import time
import sys

import utils.algorithms_utils as algorithms_utils

from ILP_CC_reducer.algorithm.algorithm import Algorithm
from ILP_CC_reducer.model import GeneralILPmodel


PointND = Tuple[float, ...]
BoxND = Tuple[float, ...]

class HybridMethodAlgorithm(Algorithm):

    @staticmethod
    def get_name() -> str:
        return 'Hybrid Method algorithm'
    
    @staticmethod
    def get_description() -> str:
        return "It obtains supported and non-supported ILP solutions."

    @staticmethod
    def execute(data_dict: dict, tau: int, info_dict: dict):

        num_of_objectives = info_dict.get("num_of_objectives")
        objectives_names = info_dict.get("objectives_list")
        model = GeneralILPmodel(active_objectives=objectives_names)
        objectives_list = algorithms_utils.organize_objectives(model, objectives_names)

        if num_of_objectives == 2:
            if not objectives_list:  # if there is no order, the order will be [EXTRACTIONS,CC]
                objectives_list = [model.extractions_objective,
                                   model.cc_difference_objective]
        elif num_of_objectives == 3:
            if not objectives_list:  # if there is no order, the order will be [EXTRACTIONS,CC,LOC]
                objectives_list = [model.extractions_objective,
                                   model.cc_difference_objective,
                                   model.loc_difference_objective]
        else:
            sys.exit("Number of objectives for hybrid method algorithm must be 2 or 3.")


        start_total = time.time()

        solutions_set, concrete, output_data, complete_data, reference_point = initialize_hybrid_method(model,
                                                                                                        objectives_list,
                                                                                                        tau,
                                                                                                        data_dict)

        objectives_names = [obj.__name__ for obj in objectives_list]

        csv_data = [objectives_names]

        for sol in solutions_set:
            csv_data.append(list(sol))

        end_total = time.time()

        total_time = end_total - start_total

        output_data.append("==========================================================================================")
        output_data.append("==========================================================================================")
        output_data.append(f"Total execution time: {total_time:.2f}")

        return csv_data, concrete, output_data, complete_data, [objectives_names, reference_point]




def initialize_hybrid_method(model: pyo.AbstractModel, objectives_list: list, tau: int, data_dict: dict):
    data = data_dict['data']

    model.tau = pyo.Param(within=pyo.NonNegativeReals, initialize=tau, mutable=False)  # Threshold MO

    concrete = model.create_instance(data)

    reference_point = algorithms_utils.obtaint_reference_point(concrete, objectives_list)

    initial_box = tuple(reference_point)

    solutions_set, concrete, output_data, complete_data = hybrid_method_with_full_p_split(model, data_dict,
                                                                                          objectives_list,
                                                                                          initial_box,
                                                                                          max_solutions=20)

    return solutions_set, concrete, output_data, complete_data, reference_point



def solve_hybrid_method(model: pyo.AbstractModel, data: dp.DataPortal, objectives_list: list, box: tuple):
    output_data = []

    algorithms_utils.modify_component(model, 'obj', pyo.Objective(
        rule=lambda m: sum(obj(m) for obj in objectives_list)))

    add_boxes_constraints(model, box, objectives_list)

    concrete, result = algorithms_utils.concrete_and_solve_model(model, data)

    if (result.solver.status == 'ok') and (result.solver.termination_condition == 'optimal') :
        newrow = tuple(round(pyo.value(obj(concrete))) for obj in objectives_list)  # Results for CSV file


        desired_order_for_objectives = ['extractions_objective', 'cc_difference_objective', 'loc_difference_objective']
        objectives_dict = {obj.__name__: obj for obj in objectives_list}
        ordered_objectives = [objectives_dict[name] for name in desired_order_for_objectives if name in objectives_dict]
        ordered_newrow = tuple(round(pyo.value(obj(concrete))) for obj in ordered_objectives)

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


def add_boxes_constraints(model: pyo. AbstractModel, box: tuple, objectives_list: list):
    for i in range(len(objectives_list)):
        if hasattr(model, f'box_u{i+1}_constraint'):
            model.del_component(f'box_u{i+1}_constraint')

    for i, obj_func in enumerate(objectives_list):
        attr_name = f'box_u{i + 1}_constraint'
        rhs = box[i] - 1

        def make_rule(obj, r):
            return lambda m: obj(m) <= r

        setattr(
            model,
            attr_name,
            pyo.Constraint(rule=make_rule(obj_func, rhs))
        )


def hybrid_method_with_full_p_split(model: pyo.AbstractModel, data_dict, objectives_list,
                                    initial_box: tuple, max_solutions=100):
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

    boxes = [initial_box]  # tuple list (u_1, ..., u_n)

    concrete = None

    while boxes and len(solutions_set) < max_solutions:

        print(
            "=============================================================================================================")

        print(f"Processing hybrid method with boxes: {boxes}.")

        actual_box = boxes.pop(0)

        solution, concrete, result, cplex_time, new_output_data, ordered_newrow = solve_hybrid_method(model,
                                                                                                      data_dict['data'],
                                                                                                      objectives_list,
                                                                                                      actual_box)

        if solution:
            solutions_set.add(solution)  # Add solution to solutions set
            s_ordered.add(ordered_newrow)

            output_data = output_data + new_output_data
            output_data.append(f"CPLEX time: {cplex_time}.")

            complete_data_new_row = algorithms_utils.write_complete_info(concrete, result, data_dict)
            complete_data.append(complete_data_new_row)

            print(f"New solution: {solution}.")

            # Partimos la caja con respecto a la solución real obtenida
            boxes = full_p_split(actual_box, solution, boxes)

            for i,box in enumerate(boxes):
                if inside(solution,box):
                    boxes.pop(i)
                    boxes = full_p_split(box, solution, boxes)

            print(f"GENERAL BOXES LIST: {boxes}.")
            non_dominated_boxes = filter_contained_boxes(boxes)
            print(f"NON DOMINATED BOXES: {non_dominated_boxes}.")
            boxes = non_dominated_boxes

        else:
            # No se encontró solución → caja descartada
            output_data.append("======================================================================================")
            output_data.append(f"SOLUTION NOT FOUND, CPLEX TIME: {cplex_time}")
            output_data.append("======================================================================================")


    print(f"Solution set: {s_ordered}.")

    return solutions_set, concrete, output_data, complete_data


def full_p_split(box: BoxND, z: tuple, boxes: list) -> List[Optional[BoxND]]:
    dimensions = len(box)

    l = (0,) * dimensions
    u = box  # original box: l = (l1, ..., l_n), u = (u1, ..., u_n)
    new_boxes = []

    for i in range(dimensions):  # i = 0 (x), 1 (y), 2 (z)

        new_u = list(u)
        for j in range(i):
            new_u[j] = u[j]
        new_u[i] = max(z[i], l[i])  # cut by x_i < z_i

        is_empty = any(u[k] < z[k] for k in range(i+1, dimensions)) or new_u[i] <= l[i]

        new_boxes.append(None if is_empty else tuple(new_u))

    for i, box in enumerate(new_boxes):
        if box is not None:
            boxes.append(box)

    return boxes


def inside(a: tuple, b: tuple) -> bool:
    """
    Returns True if point 'a' is completely inside 'b'.
    """
    return all(a[i] < b[i] for i in range(len(a)))


def filter_contained_boxes(boxes: List[PointND]) -> List[PointND]:
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
            # If box_i dominates box_j
            if algorithms_utils.dominates(box_i, box_j):
                dominated = True
                print(f"Discarded box: {box_i} because it is inside box {box_j}.")
                break
        if not dominated:
            non_dominated.append(box_i)

    return non_dominated


