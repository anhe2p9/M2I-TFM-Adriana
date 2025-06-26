import pyomo.environ as pyo  # ayuda a definir y resolver problemas de optimización
import pyomo.dataportal as dp  # permite cargar datos para usar en esos modelos de optimización

import utils.algorithms_utils as algorithms_utils

from ILP_CC_reducer.algorithm.algorithm import Algorithm
from ILP_CC_reducer.models import MultiobjectiveILPmodel

multiobjective_model = MultiobjectiveILPmodel()


class EpsilonConstraintAlgorithm(Algorithm):

    @staticmethod
    def get_name() -> str:
        return 'Epsilon Constraint algorithm'

    @staticmethod
    def get_description() -> str:
        return "It obtains supported and non-supported ILP solutions."

    @staticmethod
    def execute(data_dict: dict, tau: int, objectives_list: list) -> tuple:

        data = data_dict['data']

        if not objectives_list:  # if there is no order, the order will be [EXTRACTIONS,CC]
            objectives_list = [multiobjective_model.extractions_objective,
                               multiobjective_model.cc_difference_objective]

        obj1, obj2 = objectives_list[:2]
        output_data = []
        csv_data = [[obj.__name__ for obj in objectives_list]]

        multiobjective_model.tau = pyo.Param(within=pyo.NonNegativeReals, initialize=tau, mutable=False)  # Threshold
        multiobjective_model.obj = pyo.Objective(rule=lambda m: obj2(m))  # Objective {min f2}

        """ z <- Solve {min f2(x) subject to x in X} """
        concrete, result = algorithms_utils.concrete_and_solve_model(multiobjective_model, data)

        if (result.solver.status == 'ok') and (result.solver.termination_condition == 'optimal'):
            f1z = solve_min_f1_subject_to_f2x_lower_than_f2z(obj1, obj2, concrete, output_data, data)
            new_row = add_solution_to_csv_data(concrete, objectives_list, csv_data)
            add_result_to_output_data(concrete, objectives_list, new_row, output_data, result)

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

                """ Solve epsilon constraint problem """
                concrete, result = solve_min_f2_subject_to_epsilon_constraint(obj1, obj2, data)

                """ Checks if exists x in X that makes f1(x) <= epsilon (if exists solution) """
                if (result.solver.status == 'ok') and (result.solver.termination_condition == 'optimal'):
                    u1 = update_pareto_front_and_params(concrete, obj1, objectives_list, csv_data)
                    add_data_to_output(output_data, concrete, objectives_list, new_row, result)
                solution_found = (result.solver.status == 'ok') and (result.solver.termination_condition == 'optimal')

        return csv_data, concrete, output_data, None, None



def solve_min_f1_subject_to_f2x_lower_than_f2z(obj1: pyo.Objective, obj2: pyo.Objective,
                                               concrete: pyo.ConcreteModel, output_data: list, data: dp.DataPortal):
    """
    z <- Solve {min f1(x) subject to f2(x) <= f2(z)}
    """
    f2z = round(pyo.value(obj2(concrete)))  # f2(z) := f2z

    print(
        "==================================================================================================\n")
    print(f"min f2(x), {obj2.__name__}, subject to x in X: {f2z}")

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

    print(f"min f1(x), extractions, subject to f2(x) <= f2(z): {round(f1z)}\n")

    return f1z


def add_solution_to_csv_data(concrete: pyo.ConcreteModel, objectives_list: list, csv_data: list):
    """ FP <- {z} (add z to Pareto front) """
    new_row = [round(pyo.value(obj(concrete))) for obj in objectives_list]  # Results for CSV file
    csv_data.append(new_row)

    return new_row


def add_result_to_output_data(concrete: pyo.ConcreteModel, objectives_list: list,
                              new_row: list, output_data: list, result):
    output_data.append('===============================================================================')
    if result.solver.status == 'ok':
        for i, obj in enumerate(objectives_list):
            output_data.append(f'Objective {obj.__name__}: {new_row[i]}')
        output_data.append('Sequences selected:')
        for s in concrete.S:
            output_data.append(f"x[{s}] = {concrete.x[s].value}")
    output_data.append('===============================================================================')


def solve_min_f2_subject_to_epsilon_constraint(obj1: pyo.Objective, obj2: pyo.Objective, data: dp.DataPortal):
    """
    z <- solve {min f2(x) - lambda * l, subject to f1(x) + l = epsilon}
    """
    algorithms_utils.modify_component(multiobjective_model, 'obj', pyo.Objective(
        rule=lambda m: multiobjective_model.epsilon_objective_2obj(m, obj2)))  # min f2(x) - lambda * l
    algorithms_utils.modify_component(multiobjective_model, 'epsilonConstraint', pyo.Constraint(
        rule=lambda m: multiobjective_model.epsilon_constraint_2obj(m, obj1)))  # f1(x) + l = epsilon

    concrete, result = algorithms_utils.concrete_and_solve_model(multiobjective_model, data)  # Solve problem

    return concrete,result


def add_data_to_output(output_data: list, concrete: pyo.ConcreteModel, objectives_list: list, new_row: list, result):
    output_data.append('===============================================================================')
    if result.solver.status == 'ok':
        for i, obj in enumerate(objectives_list):
            output_data.append(f'Objective {obj.__name__}: {new_row[i]}')
        output_data.append('Sequences selected:')
        for s in concrete.S:
            output_data.append(f"x[{s}] = {concrete.x[s].value}")
    output_data.append('===============================================================================')


def update_pareto_front_and_params(concrete: pyo.ConcreteModel, obj1: pyo.Objective,
                                   objectives_list: list, csv_data: list):
    """ PF = PF U {z} """
    f1z = round(pyo.value(obj1(concrete)))
    new_row = [round(pyo.value(obj(concrete))) for obj in objectives_list]
    csv_data.append(new_row)

    """ epsilon = f1(z) - 1 """
    algorithms_utils.modify_component(multiobjective_model, 'epsilon',
                                      pyo.Param(initialize=f1z - 1, mutable=True))

    # lower bound for f1(x) (it has to decrease with f1z)
    u1 = f1z - 1

    return u1