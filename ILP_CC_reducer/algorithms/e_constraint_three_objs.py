import pyomo.environ as pyo  # ayuda a definir y resolver problemas de optimización
import pyomo.dataportal as dp  # permite cargar datos para usar en esos modelos de optimización

import utils.algorithms_utils as algorithms_utils

import sys

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
    def execute(data_dict: dict, tau: int, num_of_objectives: int, objectives_list: list) -> tuple:

        data = data_dict['data']

        if num_of_objectives == 2:
            if not objectives_list:  # if there is no order, the order will be [EXTRACTIONS,CC]
                objectives_list = [multiobjective_model.extractions_objective,
                                   multiobjective_model.cc_difference_objective]
        elif num_of_objectives == 3:
            if not objectives_list:  # if there is no order, the order will be [EXTRACTIONS,CC,LOC]
                objectives_list = [multiobjective_model.extractions_objective,
                                   multiobjective_model.cc_difference_objective,
                                   multiobjective_model.loc_difference_objective]
        else:
            sys.exit("Number of objectives for hybrid method algorithm must be 2 or 3.")

        p = len(objectives_list)

        output_data = []
        results_csv = [[obj.__name__ for obj in objectives_list]]

        multiobjective_model.tau = pyo.Param(within=pyo.NonNegativeReals, initialize=tau, mutable=False)  # Threshold

        """ Obtain payoff table by the lexicographic optimization of the objective functions """
        opt_lex_list = []
        opt_lex_table = []

        multiobjective_model.obj = pyo.Objective(rule=lambda m: objectives_list[0](m))

        print(f"--------------------------------------------------------------------------------")
        for i, objective in enumerate(objectives_list):
            algorithms_utils.modify_component(multiobjective_model, 'obj',
                                              pyo.Objective(rule=lambda m: objective(m)))  # new objective: min f2(x)

            concrete, result = algorithms_utils.concrete_and_solve_model(multiobjective_model, data)

            if (result.solver.status == 'ok') and (result.solver.termination_condition == 'optimal'):
                opt_lex = [round(pyo.value(opt(concrete))) for opt in objectives_list]

                opt_lex_table.append(opt_lex)

                f_nz = round(pyo.value(objective(concrete)))  # f_n(z) := f_nz

                opt_lex_list.append(f_nz)

                print(f"min f{i+1}(x), {objective.__name__}. Result obtained: f{i+1}(z) = {round(f_nz)}.")

                attr_name = f'f{i+1}z_constraint'

                def make_rule(obj, r):
                    return lambda m: obj(m) <= r

                setattr(
                    multiobjective_model,
                    attr_name,
                    pyo.Constraint(rule=make_rule(objective, f_nz))
                )

        opt_lex_point = tuple(opt_lex_list)
        print(f"--------------------------------------------------------------------------------")
        print(f"Lexicographic optima list: {opt_lex_point}.")

        for i in range(p):
            multiobjective_model.del_component(f'f{i+1}z_constraint')  # delete f_n(x) <= f_n(z) constraint

        """ Set upper bounds ub_k for k=2...p """
        ub_dict = {multiobjective_model.extractions_objective: len(concrete.S) + 1,
                      multiobjective_model.cc_difference_objective: concrete.nmcc[0] + 1,
                      multiobjective_model.loc_difference_objective: concrete.loc[0] + 1}

        ub_point = []
        for obj in objectives_list:
            ub_point.append(ub_dict[obj])

        ub = tuple(ub_point)

        print(f"Upper bounds: {ub}.")

        """ Calculate ranges (r_1, ..., r_p) """
        ranges = tuple(u - l for u, l in zip(ub, opt_lex_point))
        print(f"Ranges: {ranges}.")


        """ Set number of gridpoints g_k (k=2...p) for the p-1 obj.functions' ranges """
        grid_points = ranges[1:]
        print(f"Grid points: {grid_points}.")

        neff = 0
        multiobjective_model.s = pyo.Var(range(p),
                                         domain=pyo.NonNegativeReals)


        for i in range(ub_point[1], 0, -1):
            for j in range(ub_point[2], 0, -1):
                concrete, result, feasible = solve_e_constraint(objectives_list, ub_point[1:], data)

                # concrete.pprint()

                if feasible:
                    new_sol = [round(pyo.value(obj(concrete))) for obj in objectives_list]

                    desired_order_for_objectives = ['extractions_objective', 'cc_difference_objective',
                                                    'loc_difference_objective']
                    objectives_dict = {obj.__name__: obj for obj in objectives_list}
                    ordered_objectives = [objectives_dict[name] for name in desired_order_for_objectives if
                                          name in objectives_dict]
                    ordered_newrow = tuple(round(pyo.value(obj(concrete))) for obj in ordered_objectives)

                    print(f"New solution found: {ordered_newrow}.")

                    neff += 1

                    results_csv.append(new_sol)
                    print(f"Registering solution #{neff}")
                else:
                    print(f"Infeasible.")
                    break


        print(f"Total number of efficient solutions found: {neff}")
        print(f"Solutions: {results_csv}.")

        return results_csv, concrete, output_data, None, None


def solve_e_constraint(objectives_list: list, ub, data):
    eps = 1 / (10 ** 3)

    def make_objective(obj):
        return lambda m: obj(m) - eps * sum(m.s[i] for i in range(1, len(objectives_list)))

    algorithms_utils.modify_component(multiobjective_model, 'obj',
                                      pyo.Objective(rule=make_objective(objectives_list[0])))

    for k, objective in enumerate(objectives_list[1:]):
        attr_name = f'f{k + 1}z_constraint_eps_problem'

        def make_rule(itr, obj, ep):
            return lambda m: obj(m) + m.s[itr] == ep

        print(
            f"ub[{k}]: {ub[k]} - (it[{k}] * ranges[{k}]) / grid_points[{k}]: ({it[k]} * {ranges[k]}) / {grid_points[k]} = {ub[k]} - {(it[k] * ranges[k]) / grid_points[k]} = {ub[k] - (it[k] * ranges[k]) / grid_points[k]}.")

        algorithms_utils.modify_component(multiobjective_model, attr_name, pyo.Constraint(
            rule=make_rule(k, objective, ub[k] - (it[k] * ranges[k]) / grid_points[k])))

    concrete, result = algorithms_utils.concrete_and_solve_model(multiobjective_model, data)

    # concrete.pprint()

    solution_found = (result.solver.status == 'ok') and (result.solver.termination_condition == 'optimal')

    return concrete, result, solution_found




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
