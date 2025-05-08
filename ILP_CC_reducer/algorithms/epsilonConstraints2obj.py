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

        csv_data = [[f"{obj1.__name__}", f"{obj2.__name__}"]]

        # Define threshold
        multiobjective_model.model.tau = pyo.Param(within=pyo.NonNegativeReals, initialize=tau, mutable=False)  # Threshold

        # Solve {min f2}
        multiobjective_model.model.obj = pyo.Objective(rule=lambda m: obj2(m))
        concrete, result = algorithms_utils.concrete_and_solve_model(multiobjective_model, data)

        if (result.solver.status == 'ok') and (result.solver.termination_condition == 'optimal'):

            """ z <- Solve {min f1(x) subject to f2(x) <= f2(z)} """
            # f2(z) := f2z
            f2z = pyo.value(concrete.obj)

            output_data.append(
                "==================================================================================================\n")
            objective_handlers = {
                'LOCdifferenceObjective': lambda v: v.tmax.value - v.tmin.value,
                'CCdifferenceObjective': lambda v: v.cmax.value - v.cmin.value,
                'sequencesObjective': lambda v: sum(v.x[i].value for i in v.S),
            }

            value_f1 = objective_handlers[obj1.__name__](concrete)
            value_f2 = objective_handlers[obj2.__name__](concrete)

            print(f"min f2(x), {obj2.__name__}, subject to x in X: {f2z}\n")
            print(f"Valores en el primer paso: {value_f1, value_f2}\n")

            output_data.append(f"min f2(x), {obj2.__name__}, subject to x in X: {f2z}\n")
            output_data.append(f"Valores en el primer paso: {value_f1, value_f2}\n")

            # new parameter f2(z) to implement new constraint f2(x) <= f2(z)
            multiobjective_model.model.f2z = pyo.Param(initialize=f2z)
            # new constraint f2(x) <= f2(z)
            multiobjective_model.model.f2Constraint = pyo.Constraint(
                rule=lambda m: multiobjective_model.second_obj_diff_constraint(m, obj2))
            # new objective: min f1(x)
            algorithms_utils.modify_component(multiobjective_model, 'obj',
                                              pyo.Objective(rule=lambda m: multiobjective_model.sequencesObjective(m)))
            # Solve model
            concrete, result = algorithms_utils.concrete_and_solve_model(multiobjective_model, data)
            # z <- Solve {min f1(x) subject to f2(x) <= f2(z)}
            f1z = pyo.value(concrete.obj)

            print(f"min f1(x), sequences, subject to f2(x) <= f2(z): {f1z}\n")
            output_data.append(f"min f1(x), sequences, subject to f2(x) <= f2(z): {f1z}\n")

            """ FP <- {z} (add z to Pareto front) """
            new_row = [sum(concrete.x[i].value for i in concrete.S),
                      concrete.cmax.value - concrete.cmin.value]  # Calculate results for CSV file
            csv_data.append(new_row)

            # algorithms_utils.write_results_and_sequences_to_file(concrete, f, result.solver.status, new_row, obj2)

            output_data.append('===============================================================================')
            if result.solver.status == 'ok':
                output_data.append(f'Objective SEQUENCES: {new_row[0]}')
                output_data.append(f'Objective {obj2}: {new_row[1]}')
                output_data.append('Sequences selected:')
                for s in concrete.S:
                    output_data.append(f"x[{s}] = {concrete.x[s].value}")
            output_data.append('===============================================================================')

            # epsilon <- f1(z) - 1
            multiobjective_model.model.epsilon = pyo.Param(initialize=f1z - 1, mutable=True)

            # lower bound for f1(x)
            u1 = f1z - 1

            # l = epsilon - f1(x)
            multiobjective_model.model.l = pyo.Var(within=pyo.NonNegativeReals)

            solution_found = (result.solver.status == 'ok') and (
                        result.solver.termination_condition == 'optimal')  # while loop control
            multiobjective_model.model.del_component('f2Constraint')  # delete f2(x) <= f2(z) constraint

            multiobjective_model.model.lambda_value = pyo.Param(initialize=(1/(f1z-u1)), mutable=True)

            while solution_found:

                # estimate a lambda value > 0
                multiobjective_model.lambda_value = 1/(f1z - u1)

                """ Solve {min f2(x) - lambda * l, subject to f1(x) + l = epsilon} """
                # min f2(x) - lambda * l
                algorithms_utils.modify_component(multiobjective_model, 'obj', pyo.Objective(
                    rule=lambda m: multiobjective_model.epsilonObjective_2obj(m, obj2)))
                # subject to f1(x) + l = epsilon
                algorithms_utils.modify_component(multiobjective_model, 'epsilonConstraint', pyo.Constraint(
                    rule=lambda m: multiobjective_model.epsilonConstraint_2obj(m, obj2)))
                # Solve
                concrete, result = algorithms_utils.concrete_and_solve_model(multiobjective_model, data)

                """ While exists x in X that makes f1(x) < epsilon do """
                output_data.append(f"slack variable l value: {concrete.l.value}\n")

                # z <- solve {min f2(x) - lambda * l, subject to f1(x) + l = epsilon}

                """ PF = PF U {z} """
                new_row = [sum(concrete.x[i].value for i in concrete.S),
                          concrete.cmax.value - concrete.cmin.value]  # Calculate results for CSV file
                csv_data.append(new_row)

                """ epsilon = f1(z) - 1 """
                f1z = pyo.value(obj1(concrete))

                # New epsilon value
                algorithms_utils.modify_component(multiobjective_model, 'epsilon',
                                                  pyo.Param(initialize=f1z - 1, mutable=True))

                output_data.append(f"f1z: {f1z}\n")

                # lower bound for f1(x) (it has to decrease with f1z)
                u1 = f1z - 1

                print(f"epsilon: {concrete.epsilon.value}\n")
                print(f"u1: {u1}\n")
                print(f"lambda: {concrete.lambda_value.value}\n")
                print(f"comprobación: {f1z} <= {concrete.epsilon.value}\n")

                output_data.append(f"epsilon: {concrete.epsilon.value}\n")
                output_data.append(f"u1: {u1}\n")
                output_data.append(f"lambda: {concrete.lambda_value.value}\n")
                output_data.append(f"comprobación: {f1z} <= {concrete.epsilon.value}\n")

                # algorithms_utils.write_results_and_sequences_to_file(concrete, f, result.solver.status, new_row, obj2)

                output_data.append('===============================================================================')
                if result.solver.status == 'ok':
                    output_data.append(f'Objective SEQUENCES: {new_row[0]}')
                    output_data.append(f'Objective {obj2}: {new_row[1]}')
                    output_data.append('Sequences selected:')
                    for s in concrete.S:
                        output_data.append(f"x[{s}] = {concrete.x[s].value}")
                output_data.append('===============================================================================')

                solution_found = (result.solver.status == 'ok') and (result.solver.termination_condition == 'optimal')

        return csv_data, concrete, output_data
