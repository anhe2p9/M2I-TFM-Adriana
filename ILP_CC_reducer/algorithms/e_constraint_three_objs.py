import pyomo.environ as pyo  # ayuda a definir y resolver problemas de optimización
import pyomo.dataportal as dp  # permite cargar datos para usar en esos modelos de optimización

import utils.algorithms_utils as algorithms_utils

from ILP_CC_reducer.algorithm.algorithm import Algorithm
from ILP_CC_reducer.models import MultiobjectiveILPmodel

multiobjective_model = MultiobjectiveILPmodel()


class EpsilonConstraintForThreeObjsAlgorithm(Algorithm):

    @staticmethod
    def get_name() -> str:
        return 'Epsilon Constraint algorithm'

    @staticmethod
    def get_description() -> str:
        return "It obtains supported and non-supported ILP solutions for two objectives using e-constraint algorithm."

    @staticmethod
    def execute(data_dict: dict, tau: int, objectives_list: list) -> tuple:

        data = data_dict['data']

        if not objectives_list:  # if there is no order, the order will be [EXTRACTIONS,CC]
            objectives_list = [multiobjective_model.extractions_objective,
                               multiobjective_model.cc_difference_objective,
                               multiobjective_model.loc_difference_objective]

        obj1, obj2, obj3 = objectives_list
        output_data = []
        results_csv = [[obj.__name__ for obj in objectives_list]]

        multiobjective_model.tau = pyo.Param(within=pyo.NonNegativeReals, initialize=tau, mutable=False)  # Threshold

        """ Obtain payoff table by the lexicographic optimization of the objective functions """
        multiobjective_model.obj = pyo.Objective(rule=lambda m: obj1(m))
        for i, objective in enumerate(objectives_list):
            algorithms_utils.modify_component(multiobjective_model, 'obj',
                                              pyo.Objective(rule=lambda m: objective(m)))  # new objective: min f2(x)

            concrete, result = algorithms_utils.concrete_and_solve_model(multiobjective_model, data)

            if (result.solver.status == 'ok') and (result.solver.termination_condition == 'optimal'):
                f_nz = round(pyo.value(objective(concrete)))  # f_n(z) := f_nz

                print(
                    "==================================================================================================\n")
                print(f"min f{i}(x), {objective.__name__}, subject to x in X. Result obtained: f{i}(z) = {round(f_nz)}.")

                attr_name = f'f{i}z'

                def make_rule(obj, r):
                    return lambda m: obj(m) <= r

                setattr(
                    multiobjective_model,
                    attr_name,
                    pyo.Constraint(rule=make_rule(objective, f_nz))
                )

-----------------------------------------------------------------------------------------------

        """ z <- Solve {min f2(x) subject to x in X} """

        if (result.solver.status == 'ok') and (result.solver.termination_condition == 'optimal'):
            """
            z <- Solve {min f2(x) subject to f3(x) <= f3(z)}
            """
            f3z = round(pyo.value(obj3(concrete)))  # f3(z) := f3z

            print(
                "==================================================================================================\n")
            print(f"min f3(x), {obj3.__name__}, subject to x in X. Result obtained: f3(z) = {round(f3z)}.")

            multiobjective_model.f3z = pyo.Param(
                initialize=f3z)  # new parameter f3(z) to implement new constraint f3(x) <= f3(z)
            multiobjective_model.f3Constraint = pyo.Constraint(
                rule=lambda m: obj3(m) <= m.f3z)  # new constraint: f3(x) <= f3(z)
            algorithms_utils.modify_component(multiobjective_model, 'obj',
                                              pyo.Objective(rule=lambda m: obj2(m)))  # new objective: min f2(x)

            """ Solve {min f2(x) subject to f3(x) <= f3(z)} """
            concrete, result = algorithms_utils.concrete_and_solve_model(multiobjective_model, data)


            """
            z <- Solve {min f1(x) subject to f2(x) <= f2(z), f3(x) <= f3(z)}
            """
            f2z = round(pyo.value(obj2(concrete)))  # f2(z) := f2z

            print(f"min f2(x), {obj2.__name__}, subject to x in X. Result obtained: f2(z) = {round(f2z)}.")

            multiobjective_model.f2z = pyo.Param(
                initialize=f2z)  # new parameter f2(z) to implement new constraint f2(x) <= f2(z)
            multiobjective_model.f2Constraint = pyo.Constraint(
                rule=lambda m: obj2(m) <= m.f2z)  # new constraint: f2(x) <= f2(z)
            algorithms_utils.modify_component(multiobjective_model, 'obj',
                                              pyo.Objective(rule=lambda m: obj1(m)))  # new objective: min f1(x)



            """ Solve {min f1(x) subject to f2(x) <= f2(z)} """
            concrete, result = algorithms_utils.concrete_and_solve_model(multiobjective_model, data)
            multiobjective_model.del_component('f3Constraint')  # delete f3(x) <= f3(z) constraint
            multiobjective_model.del_component('f2Constraint')  # delete f2(x) <= f2(z) constraint

            """ FP <- {z} (add z to Pareto front) """
            new_row = [round(pyo.value(obj(concrete))) for obj in objectives_list]  # Results for CSV file
            results_csv.append(new_row)
            print(f"New solution: {new_row}.")

            add_result_to_output_data_file(concrete, objectives_list, new_row, output_data, result)

            f1z = new_row[0]


            """ epsilon1 <- f1(z) - 1 """
            multiobjective_model.epsilon1 = pyo.Param(initialize=f1z - 1, mutable=True)  # Epsilon parameter
            multiobjective_model.s1 = pyo.Var(within=pyo.NonNegativeReals)  # s1 = epsilon - f1(x)
            l1 = f1z - 1  # lower bound for f1(x)
            multiobjective_model.lambda1_value = pyo.Param(initialize=(1/((f1z-l1)*10**3)), mutable=True)  # Lambda1 parameter

            """ epsilon2 <- f2(z) - 1 """
            multiobjective_model.epsilon2 = pyo.Param(initialize=f2z - 1, mutable=True)  # Epsilon parameter
            multiobjective_model.s2 = pyo.Var(within=pyo.NonNegativeReals)  # s2 = epsilon - f2(x)
            l2 = f2z - 1  # lower bound for f2(x)
            multiobjective_model.lambda2_value = pyo.Param(initialize=(1 / ((f2z - l2) * 10 ** 3)),
                                                          mutable=True)  # Lambda2 parameter

            solution_found = (result.solver.status == 'ok') and (
                    result.solver.termination_condition == 'optimal')  # while loop control

            """ While exists x in X that makes f1(x) <= epsilon1 and f2(x) <= epsilon2 do """
            while solution_found:
                """ estimate a lambda value > 0 """
                algorithms_utils.modify_component(multiobjective_model, 'lambda1_value',
                                                  pyo.Param(initialize=(1 / ((f1z - l1) * 10 ** 3)), mutable=True))
                algorithms_utils.modify_component(multiobjective_model, 'lambda2_value',
                                                  pyo.Param(initialize=(1 / ((f2z - l2) * 10 ** 3)), mutable=True))

                """ Solve epsilon constraint problem """
                """ z <- solve {min f2(x) - lambda1 * l1 + lambda2 * l2, subject to f1(x) + s1 = epsilon1, f2(x) + s2 = epsilon2} """
                algorithms_utils.modify_component(multiobjective_model, 'obj', pyo.Objective(
                    rule=lambda m: multiobjective_model.epsilon_objective_3obj(m, obj2)))  # min f3(x) - lambda1 * s1 - lambda2 * s2
                algorithms_utils.modify_component(multiobjective_model, 'epsilonConstraint1', pyo.Constraint(
                    rule=lambda m: obj1(m) + m.s1 == m.epsilon1))  # f1(x) + s1 = epsilon1
                algorithms_utils.modify_component(multiobjective_model, 'epsilonConstraint2', pyo.Constraint(
                    rule=lambda m: obj2(m) + m.s2 == m.epsilon2))  # f2(x) + s2 = epsilon2

                concrete, result = algorithms_utils.concrete_and_solve_model(multiobjective_model,
                                                                             data)  # Solve problem

                """ Checks if exists x in X that makes f1(x) <= epsilon (if exists solution) """
                if (result.solver.status == 'ok') and (result.solver.termination_condition == 'optimal'):
                    """ PF = PF U {z} """
                    f1z = pyo.value(obj1(concrete))
                    f2z = pyo.value(obj2(concrete))
                    new_row = [round(pyo.value(obj(concrete))) for obj in objectives_list]
                    results_csv.append(new_row)
                    print(f"New solution: {new_row}.")

                    """ epsilon1 = f1(z) - 1 """
                    algorithms_utils.modify_component(multiobjective_model, 'epsilon1',
                                                      pyo.Param(initialize=f1z - 1, mutable=True))
                    """ epsilon2 = f2(z) - 1 """
                    algorithms_utils.modify_component(multiobjective_model, 'epsilon2',
                                                      pyo.Param(initialize=f2z - 1, mutable=True))

                    # lower bound for f1(x) (it has to decrease with f1z)
                    l1 = f1z - 1
                    # lower bound for f2(x) (it has to decrease with f2z)
                    l2 = f2z - 1
                    add_result_to_output_data_file(concrete, objectives_list, new_row, output_data, result)
                solution_found = (result.solver.status == 'ok') and (result.solver.termination_condition == 'optimal')

        return results_csv, concrete, output_data, None, None



def add_result_to_output_data_file(concrete: pyo.ConcreteModel, objectives_list: list,
                                   new_row: list, output_data: list, result):
    output_data.append('===============================================================================')
    if result.solver.status == 'ok':
        for i, obj in enumerate(objectives_list):
            output_data.append(f'Objective {obj.__name__}: {new_row[i]}')
        output_data.append('Sequences selected:')
        for s in concrete.S:
            output_data.append(f"x[{s}] = {concrete.x[s].value}")
    output_data.append('===============================================================================')
