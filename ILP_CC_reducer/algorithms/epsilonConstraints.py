import pyomo.environ as pyo # ayuda a definir y resolver problemas de optimización
import pyomo.dataportal as dp # permite cargar datos para usar en esos modelos de optimización

from typing import List, Tuple, Optional, Set

import time


from ILP_CC_reducer.algorithm.algorithm import Algorithm
from ILP_CC_reducer.models import MultiobjectiveILPmodel
from ILP_CC_reducer.models import ILPmodelRsain

import algorithms_utils


Box3D = Tuple[Tuple[float, float, float], Tuple[float, float, float]]
Point3D = Tuple[float, float, float]
multiobjective_model = MultiobjectiveILPmodel()
uniobjective_model = ILPmodelRsain()



class EpsilonConstraintAlgorithm(Algorithm):

    @staticmethod
    def get_name() -> str:
        return 'Epsilon Constraint algorithm with 3 obj'
    
    @staticmethod
    def get_description() -> str:
        return ("It obtains supported and non-supported ILP solutions.")

    @staticmethod
    def execute(data: dp.DataPortal, tau: int, objectives_list: list= None):

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

        initial_box = ((0,0,0),tuple(nadir_point))

        add_b0_constraints(initial_box, objectives_list)

        S, concrete, output_data = epsilon_constraint_with_ppartition(data, objectives_list, initial_box, max_solutions=20)

        csv_data = [[obj.__name__ for obj in objectives_list]]

        for sol in S:
            csv_data.append(list(sol))

        end_total = time.time()

        output_data.append("==========================================================================================")
        output_data.append("==========================================================================================")
        output_data.append(f"Total execution time: {end_total - start_total:.2f}")

        return csv_data, concrete, output_data


def add_items_to_multiobjective_model(tau: int):
    multiobjective_model.tau = pyo.Param(within=pyo.NonNegativeReals, initialize=tau, mutable=False)  # Threshold MO
    uniobjective_model.tau = pyo.Param(within=pyo.NonNegativeReals, initialize=tau, mutable=False)  # Threshold UO

    multiobjective_model.lambda2 = pyo.Param(initialize=0.5, mutable=True)  # Lambda parameter
    multiobjective_model.lambda3 = pyo.Param(initialize=0.5, mutable=True)  # Lambda parameter

    multiobjective_model.sl2 = pyo.Var(within=pyo.NonNegativeReals)  # sl2 = epsilon2 - f2(x)
    multiobjective_model.sl3 = pyo.Var(within=pyo.NonNegativeReals)  # sl3 = epsilon3 - f3(x)



def add_b0_constraints(initial_box: tuple, objectives_list: list):
    l,u = initial_box
    obj1, obj2, obj3 = objectives_list

    multiobjective_model.b0l1_constraint = pyo.Constraint(rule=lambda m: obj1(m) >= l[0])
    multiobjective_model.b0l2_constraint = pyo.Constraint(rule=lambda m: obj2(m) >= l[1])
    multiobjective_model.b0l3_constraint = pyo.Constraint(rule=lambda m: obj3(m) >= l[2])

    multiobjective_model.b0u1_constraint = pyo.Constraint(rule=lambda m: obj1(m) <= u[0])
    multiobjective_model.b0u2_constraint = pyo.Constraint(rule=lambda m: obj2(m) <= u[1])
    multiobjective_model.b0u3_constraint = pyo.Constraint(rule=lambda m: obj3(m) <= u[2])



def solve_epsilon_constraint(data: dp.DataPortal, objectives_list: list, box: tuple, eps2: float, eps3: float):
    output_data = []

    obj1, obj2, obj3 = objectives_list

    algorithms_utils.modify_component(multiobjective_model, 'obj', pyo.Objective(
        rule=lambda m: multiobjective_model.epsilon_objective(m, obj1)))  # min f1(x) - (lambda2 * l2 + lambda3 * l3)

    add_epsilon_constraints(obj2, obj3, eps2, eps3)
    add_boxes_constraints(box, objectives_list)

    concrete, result = algorithms_utils.concrete_and_solve_model(multiobjective_model, data)

    if (result.solver.status == 'ok') and (result.solver.termination_condition == 'optimal') :
        newrow = tuple(round(pyo.value(obj(concrete))) for obj in objectives_list)  # Results for CSV file

        output_data.append('===============================================================================')
        if result.solver.status == 'ok':
            for i, obj in enumerate(objectives_list):
                output_data.append(f'Objective {obj.__name__}: {newrow[i]}')
            output_data.append('Sequences selected:')
            for s in concrete.S:
                output_data.append(f"x[{s}] = {concrete.x[s].value}")

    else:
        newrow = None

    print(f"Newrow: {newrow}.")

    cplex_time = result.solver.time

    return newrow, concrete, cplex_time, output_data


def add_epsilon_constraints(obj2: pyo.Objective, obj3: pyo.Objective, eps2: int, eps3: int):
    algorithms_utils.modify_component(
        multiobjective_model, 'epsilonConstraint2', pyo.Constraint(
            rule=lambda m: obj2(m) + m.sl2 == eps2))  # subject to f2(x) + l2 = epsilon2
    algorithms_utils.modify_component(
        multiobjective_model, 'epsilonConstraint3', pyo.Constraint(
            rule=lambda m: obj3(m) + m.sl3 == eps3))  # subject to f3(x) + l3 = epsilon3


def add_boxes_constraints(box_info: tuple, objectives_list: list):
    _, solution, direction = box_info
    if solution:
        for i in range(4):
            if hasattr(multiobjective_model, f'boxes_constraint_{i}'):
                multiobjective_model.del_component(f'boxes_constraint_{i}')
        algorithms_utils.modify_component(
            multiobjective_model, f'boxes_constraint_{direction}', pyo.Constraint(
                rule=lambda m: objectives_list[direction - 1](m) <= solution[direction - 1] - 1))
        for i in range(3-direction):
            algorithms_utils.modify_component(
                multiobjective_model, f'boxes_constraint_{direction+i+1}', pyo.Constraint(
                    rule=lambda m: objectives_list[direction+i](m) >= solution[direction+i]))


def epsilon_constraint_with_ppartition(data, objectives_list, initial_box: Box3D, max_solutions=100) -> Set[Point3D]:
    output_data = []

    S = set()  # Conjunto de soluciones no dominadas
    boxes = [(initial_box, None, None)]  # lista de tuplas (caja, z, direction)

    while boxes and len(S) < max_solutions:

        # print(f"Solutions set: {S}.")
        # print(f"BOXES: {boxes}.")

        print(f"Processing e-constraint with boxes: {boxes}.")

        box_info = boxes.pop(0)
        B = box_info[0]

        l, u = B

        # Usar e[1] y e[2] como valores de epsilon para restricciones f2 y f3
        solution, concrete, cplex_time, new_output_data = solve_epsilon_constraint(data, objectives_list,
                                                                               box_info, eps2=u[1], eps3=u[2])

        if solution:
            # Si ya tenemos exactamente esa solución, no la volvemos a usar
            if any(approx_equal(solution, s) for s in S):
                continue  # solución repetida, descartar caja

            # Si está dominada, tampoco sirve
            if any(dominates(other, solution) for other in S):
                continue  # caja dominada, no se parte

            # Solución válida y no dominada → la guardamos
            S.add(solution)
            output_data = output_data + new_output_data
            output_data.append(f"CPLEX time: {cplex_time}.")

            print(f"New solution: {solution}.")

            # Partimos la caja con respecto a la solución real obtenida
            z = solution
            new_boxes = p_partition_3d(B, z)

            for i, box in enumerate(new_boxes):
                if box is not None:
                    boxes.append((box, solution, i+1))

        else:
            # No se encontró solución → caja descartada
            output_data.append("======================================================================================")
            output_data.append(f"SOLUTION NOT FOUND, CPLEX TIME: {cplex_time}")
            output_data.append("======================================================================================")

            continue

    print(f"Solution set: {S}.")

    return S, concrete, output_data


def p_partition_3d(B: Box3D, z: Point3D) -> List[Optional[Box3D]]:
    l, u = B
    boxes = []

    for i in range(3):
        new_l = list(l)
        for j in range(i + 1, 3):
            new_l[j] = max(z[j], l[j])

        new_u = list(u)
        for j in range(i):
            new_u[j] = u[j]
        new_u[i] = max(z[i], l[i])

        is_empty = any(new_l[k] >= new_u[k] for k in range(3))
        boxes.append(None if is_empty else (tuple(new_l), tuple(new_u)))

    return boxes


def dominates(a: Point3D, b: Point3D) -> bool:
    """
    Returns True if point a dominates b.
    """
    return all(a[i] <= b[i] for i in range(3)) and any(a[i] < b[i] for i in range(3))

def approx_equal(a: Point3D, b: Point3D, tol=1e-4):
    return all(abs(ai - bi) < tol for ai, bi in zip(a, b))