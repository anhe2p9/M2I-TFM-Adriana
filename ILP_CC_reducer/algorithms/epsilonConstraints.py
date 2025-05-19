import pyomo.environ as pyo # ayuda a definir y resolver problemas de optimización
import pyomo.dataportal as dp # permite cargar datos para usar en esos modelos de optimización

from typing import List, Tuple, Optional, Set

import heapq

# from typing import Any
# import sys

# import os
# import csv

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

        S, concrete = epsilon_constraint_with_ppartition(data, objectives_list, initial_box, max_solutions=20)

        print(f"SOLUTION: {S}")

        csv_data = [[obj.__name__ for obj in objectives_list]]

        for sol in S:
            csv_data.append(list(sol))

        return csv_data, concrete, None

def add_items_to_multiobjective_model(tau: int):
    multiobjective_model.tau = pyo.Param(within=pyo.NonNegativeReals, initialize=tau, mutable=False)  # Threshold MO
    uniobjective_model.tau = pyo.Param(within=pyo.NonNegativeReals, initialize=tau, mutable=False)  # Threshold UO

    multiobjective_model.lambda2 = pyo.Param(initialize=0.5, mutable=True)  # Lambda parameter
    multiobjective_model.lambda3 = pyo.Param(initialize=0.5, mutable=True)  # Lambda parameter

    multiobjective_model.l2 = pyo.Var(within=pyo.NonNegativeReals)  # l2 = epsilon2 - f2(x)
    multiobjective_model.l3 = pyo.Var(within=pyo.NonNegativeReals)  # l3 = epsilon3 - f3(x)




def solve_epsilon_constraint(data: dp.DataPortal, objectives_list: list, solution, direction: int, eps2: float, eps3: float):

    obj1, obj2, obj3 = objectives_list

    algorithms_utils.modify_component(multiobjective_model, 'obj', pyo.Objective(
        rule=lambda m: multiobjective_model.epsilon_objective(m, obj1)))  # min f1(x) - (lambda2 * l2 + lambda3 * l3)

    algorithms_utils.modify_component(
        multiobjective_model, 'epsilonConstraint2', pyo.Constraint(
            rule=lambda m: obj2(m) + m.l2 == eps2))  # subject to f2(x) + l2 = epsilon2
    algorithms_utils.modify_component(
        multiobjective_model, 'epsilonConstraint3', pyo.Constraint(
            rule=lambda m: obj3(m) + m.l3 == eps3))  # subject to f3(x) + l3 = epsilon3

    print(f"SOLUTION: {solution}")

    if solution:
        for i in range(direction):
            if i == 0:
                algorithms_utils.modify_component(
                    multiobjective_model, f'boxes_constraint_{i}', pyo.Constraint(
                        rule=lambda m: objectives_list[i](m) <= solution[i] - 1))
            else:
                algorithms_utils.modify_component(
                    multiobjective_model, f'boxes_constraint_{i}', pyo.Constraint(
                        rule=lambda m: objectives_list[i](m) >= solution[i]))

    concrete, result = algorithms_utils.concrete_and_solve_model(multiobjective_model, data)

    # concrete.pprint()

    if (result.solver.status == 'ok') and (result.solver.termination_condition == 'optimal') :
        newrow = tuple(round(pyo.value(obj(concrete))) for obj in objectives_list)  # Results for CSV file
    else:
        newrow = None

    print(F"NEWROW: {newrow}")

    return newrow, concrete


def epsilon_constraint_with_ppartition(data, objectives_list, initial_box: Box3D, max_solutions=100) -> Set[Point3D]:
    S = set()  # Conjunto de soluciones no dominadas
    boxes = [(initial_box, None)]  # lista de tuplas (caja, z)

    solution = None

    while boxes and len(S) < max_solutions:
        print(f"BOX: {boxes}")

        B, direction = boxes.pop(0)
        l, u = B

        # Punto z intermedio para partir (punto central)
        # z = tuple((l[i] + u[i]) / 2 for i in range(3))


        # Usar e[1] y e[2] como valores de epsilon para restricciones f2 y f3
        solution, concrete = solve_epsilon_constraint(data, objectives_list, solution, direction, eps2=u[1], eps3=u[2])

        print("Intentando con eps2 =", u[1], "eps3 =", u[2])
        print("Solución obtenida:", solution)

        if solution is not None:
            # Si ya tenemos exactamente esa solución, no la volvemos a usar
            if any(approx_equal(solution, s) for s in S):
                continue  # solución repetida, descartar caja

            # Si está dominada, tampoco sirve
            if any(dominates(other, solution) for other in S):
                continue  # caja dominada, no se parte

            # Solución válida y no dominada → la guardamos
            S.add(solution)

            # Partimos la caja con respecto a la solución real obtenida
            z = solution
            new_boxes = p_partition_3d(B, z)

            for i, box in enumerate(new_boxes):
                if box is not None:
                    boxes.append((box, i))
        else:
            # No se encontró solución → caja descartada
            continue

    return S, concrete


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
    print(f"VALOR B: {a}")
    print(f"VALOR B: {b}")

    return all(a[i] <= b[i] for i in range(3)) and any(a[i] < b[i] for i in range(3))

def approx_equal(a: Point3D, b: Point3D, tol=1e-4):
    return all(abs(ai - bi) < tol for ai, bi in zip(a, b))