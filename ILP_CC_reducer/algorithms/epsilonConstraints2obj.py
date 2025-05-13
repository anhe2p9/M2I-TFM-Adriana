import pyomo.environ as pyo  # ayuda a definir y resolver problemas de optimización
import pyomo.dataportal as dp  # permite cargar datos para usar en esos modelos de optimización

# import sys
# import csv
# import os

from ILP_CC_reducer.algorithm.algorithm import Algorithm
from ILP_CC_reducer.models import MultiobjectiveILPmodel

import algorithms_utils


class EpsilonConstraintAlgorithm2obj(Algorithm):

    @staticmethod
    def get_name() -> str:
        return 'Epsilon Constraint algorithm'

    @staticmethod
    def get_description() -> str:
        return "It obtains supported and non-supported ILP solutions."

    @staticmethod
    def execute(data: dp.DataPortal, tau: int, objectives_list: list) -> tuple:

        multiobjective_model: MultiobjectiveILPmodel = MultiobjectiveILPmodel()

        obj1, obj2 = objectives_list[:2]

        output_data = []

        csv_data = [[obj.__name__ for obj in objectives_list]]

        multiobjective_model.tau = pyo.Param(within=pyo.NonNegativeReals, initialize=tau, mutable=False)  # Threshold

        multiobjective_model.obj = pyo.Objective(rule=lambda m: obj2(m))  # Objective {min f2}

        """ z <- Solve {min f2(x) subject to x in X} """
        concrete, result = algorithms_utils.concrete_and_solve_model(multiobjective_model, data)

        if (result.solver.status == 'ok') and (result.solver.termination_condition == 'optimal'):

            """ z <- Solve {min f1(x) subject to f2(x) <= f2(z)} """
            f2z = round(pyo.value(obj2(concrete)))  # f2(z) := f2z

            print(
                "==================================================================================================\n")
            print(f"min f2(x), {obj2.__name__}, subject to x in X: {f2z}")

            output_data.append(
                "==================================================================================================\n")
            output_data.append(f"min f2(x), {obj2.__name__}, subject to x in X: {f2z}\n")

            multiobjective_model.f2z = pyo.Param(
                initialize=f2z)  # new parameter f2(z) to implement new constraint f2(x) <= f2(z)
            multiobjective_model.f2Constraint = pyo.Constraint(
                rule=lambda m: multiobjective_model.second_obj_diff_constraint(m, obj2))  # new constraint: f2(x) <= f2(z)
            algorithms_utils.modify_component(multiobjective_model, 'obj',
                                              pyo.Objective(rule=lambda m: obj1(m)))  # new objective: min f1(x)

            """ Solve {min f1(x) subject to f2(x) <= f2(z)} """
            concrete, result = algorithms_utils.concrete_and_solve_model(multiobjective_model, data)
            multiobjective_model.del_component('f2Constraint')  # delete f2(x) <= f2(z) constraint

            f1z = round(pyo.value(obj1(concrete)))  # f1(z) := f1z

            print(f"min f1(x), sequences, subject to f2(x) <= f2(z): {round(f1z)}\n")
            output_data.append(f"min f1(x), sequences, subject to f2(x) <= f2(z): {round(f1z)}\n")

            """ FP <- {z} (add z to Pareto front) """
            new_row = [round(pyo.value(obj(concrete))) for obj in objectives_list]  # Results for CSV file
            csv_data.append(new_row)

            output_data.append('===============================================================================')
            if result.solver.status == 'ok':
                for i, obj in enumerate(objectives_list):
                    output_data.append(f'Objective {obj.__name__}: {new_row[i]}')
                output_data.append('Sequences selected:')
                for s in concrete.S:
                    output_data.append(f"x[{s}] = {concrete.x[s].value}")
            output_data.append('===============================================================================')

            """ epsilon <- f1(z) - 1 """
            multiobjective_model.epsilon = pyo.Param(initialize=f1z - 1, mutable=True)  # Epsilon parameter

            multiobjective_model.l = pyo.Var(within=pyo.NonNegativeReals)  # l = epsilon - f1(x)

            u1 = f1z - 1  # lower bound for f1(x)
            multiobjective_model.lambda_value = pyo.Param(initialize=(1/(f1z-u1)), mutable=True)  # Lambda parameter

            solution_found = (result.solver.status == 'ok') and (
                    result.solver.termination_condition == 'optimal')  # while loop control

            """ While exists x in X that makes f1(x) <= epsilon do """
            while solution_found:
                """ estimate a lambda value > 0 """
                algorithms_utils.modify_component(multiobjective_model, 'lambda_value',
                                                  pyo.Param(initialize=(1/(f1z-u1)), mutable=True))

                """ z <- solve {min f2(x) - lambda * l, subject to f1(x) + l = epsilon} """
                algorithms_utils.modify_component(multiobjective_model, 'obj', pyo.Objective(
                    rule=lambda m: multiobjective_model.epsilon_objective_2obj(m, obj2)))  # min f2(x) - lambda * l
                algorithms_utils.modify_component(multiobjective_model, 'epsilonConstraint', pyo.Constraint(
                    rule=lambda m: multiobjective_model.epsilon_constraint_2obj(m, obj1)))  # f1(x) + l = epsilon

                concrete, result = algorithms_utils.concrete_and_solve_model(multiobjective_model, data)  # Solve problem

                """ Checks if exists x in X that makes f1(x) <= epsilon (if exists solution) """
                if (result.solver.status == 'ok') and (result.solver.termination_condition == 'optimal'):

                    print(f"l value: {round(concrete.l.value)}")
                    output_data.append(f"slack variable l value: {concrete.l.value}\n")

                    """ PF = PF U {z} """
                    f1z = round(pyo.value(obj1(concrete)))

                    new_row = [round(pyo.value(obj(concrete))) for obj in objectives_list]
                    # new_row = [f1z,f2z]  # Results for CSV file
                    csv_data.append(new_row)

                    """ epsilon = f1(z) - 1 """
                    algorithms_utils.modify_component(multiobjective_model, 'epsilon',
                                                      pyo.Param(initialize=f1z - 1, mutable=True))

                    output_data.append(f"f1z: {round(f1z)}\n")

                    # lower bound for f1(x) (it has to decrease with f1z)
                    u1 = f1z - 1

                    output_data.append(f"epsilon: {round(concrete.epsilon.value)}\n")
                    output_data.append(f"u1: {round(u1)}\n")
                    output_data.append(f"lambda: {round(concrete.lambda_value.value)}\n")
                    output_data.append(f"comprobación: {round(f1z)} <= {round(concrete.epsilon.value)}\n")
                    output_data.append('===============================================================================')
                    if result.solver.status == 'ok':
                        for i, obj in enumerate(objectives_list):
                            output_data.append(f'Objective {obj.__name__}: {new_row[i]}')
                        output_data.append('Sequences selected:')
                        for s in concrete.S:
                            output_data.append(f"x[{s}] = {concrete.x[s].value}")
                    output_data.append('===============================================================================')

                solution_found = (result.solver.status == 'ok') and (result.solver.termination_condition == 'optimal')

        return csv_data, concrete, output_data
