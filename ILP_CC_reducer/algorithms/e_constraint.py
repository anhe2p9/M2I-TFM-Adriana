import pyomo.environ as pyo  # ayuda a definir y resolver problemas de optimización

import utils.algorithms_utils as algorithms_utils
import sys
import time

from ILP_CC_reducer.algorithm.algorithm import Algorithm
from ILP_CC_reducer.model import GeneralILPmodel


class EpsilonConstraintAlgorithm(Algorithm):

    @staticmethod
    def get_name() -> str:
        return 'Epsilon Constraint algorithm'

    @staticmethod
    def get_description() -> str:
        return "It obtains supported and non-supported ILP solutions for two objectives using e-constraint algorithm."

    @staticmethod
    def execute(data_dict: dict, tau: int, info_dict: dict) -> tuple:

        num_of_objectives = info_dict.get("num_of_objectives")
        objectives_names = info_dict.get("objectives_list")
        model = GeneralILPmodel(active_objectives=objectives_names)
        objectives_list = algorithms_utils.organize_objectives(model, objectives_names)

        start_total = time.time()

        if num_of_objectives == 2:
            if not objectives_list:  # if there is no order, the order will be [EXTRACTIONS,CC]
                objectives_list = [model.extractions_objective,
                                   model.cc_difference_objective]
            results_csv, concrete, output_data, complete_data = e_constraint_2objs(data_dict, tau,
                                                                                   objectives_list, model)
        elif num_of_objectives == 3:
            if not objectives_list:  # if there is no order, the order will be [EXTRACTIONS,CC,LOC]
                objectives_list = [model.extractions_objective,
                                   model.cc_difference_objective,
                                   model.loc_difference_objective]
            results_csv, concrete, output_data, complete_data = e_constraint_3objs(data_dict, tau,
                                                                                   objectives_list, model)
        else:
            sys.exit("Number of objectives for augmented e-constraint algorithm must be 2 or 3.")

        objectives_names = [obj.__name__ for obj in objectives_list]

        reference_point = algorithms_utils.obtaint_reference_point(concrete, objectives_list)

        end_total = time.time()

        total_time = end_total - start_total

        output_data.append("==========================================================================================")
        output_data.append("==========================================================================================")
        output_data.append(f"Total execution time: {total_time:.2f}")


        return results_csv, concrete, output_data, complete_data, [objectives_names, reference_point]


def e_constraint_2objs(data_dict: dict, tau: int, objectives_list: list, model: pyo.AbstractModel):
    data = data_dict['data']

    if not objectives_list:  # if there is no order, the order will be [EXTRACTIONS,CC]
        objectives_list = [model.extractions_objective, model.cc_difference_objective]

    obj1, obj2 = objectives_list
    output_data = []
    results_csv = [[obj.__name__ for obj in objectives_list]]

    complete_data = [["numberOfSequences", "numberOfVariables", "numberOfConstraints",
                      "initialComplexity", "solution (index,CC,LOC)", "offsets", "extractions",
                      "not_nested_solution", "not_nested_extractions",
                      "nested_solution", "nested_extractions",
                      "reductionComplexity", "finalComplexity",
                      "minExtractedLOC", "maxExtractedLOC", "meanExtractedLOC", "totalExtractedLOC", "nestedLOC",
                      "minReductionOfCC", "maxReductionOfCC", "meanReductionOfCC", "totalReductionOfCC", "nestedCC",
                      "minExtractedParams", "maxExtractedParams", "meanExtractedParams", "totalExtractedParams",
                      "terminationCondition", "executionTime"]]

    model.tau = pyo.Param(within=pyo.NonNegativeReals, initialize=tau, mutable=False)  # Threshold

    """ z <- Solve {min f2(x) subject to x in X} """
    model.obj = pyo.Objective(rule=lambda m: obj2(m))  # Objective {min f2}
    concrete, result = algorithms_utils.concrete_and_solve_model(model, data)

    if (result.solver.status == 'ok') and (result.solver.termination_condition == 'optimal'):
        """
        z <- Solve {min f1(x) subject to f2(x) <= f2(z)}
        """
        f2z = round(pyo.value(obj2(concrete)))  # f2(z) := f2z

        print(
            "==================================================================================================\n")
        print(f"min f2(x), {obj2.__name__}, subject to x in X. Result obtained: f2(z) = {round(f2z)}.")

        model.f2z = pyo.Param(
            initialize=f2z)  # new parameter f2(z) to implement new constraint f2(x) <= f2(z)
        model.f2Constraint = pyo.Constraint(
            rule=lambda m: model.second_obj_diff_constraint(m,
                                                                           obj2))  # new constraint: f2(x) <= f2(z)
        algorithms_utils.modify_component(model, 'obj',
                                          pyo.Objective(rule=lambda m: obj1(m)))  # new objective: min f1(x)

        """ Solve {min f1(x) subject to f2(x) <= f2(z)} """
        concrete, result = algorithms_utils.concrete_and_solve_model(model, data)
        model.del_component('f2Constraint')  # delete f2(x) <= f2(z) constraint

        """ FP <- {z} (add z to Pareto front) """
        new_row = [round(pyo.value(obj(concrete))) for obj in objectives_list]  # Results for CSV file
        results_csv.append(new_row)
        print(f"New solution: {new_row}.")

        add_result_to_output_data_file(concrete, objectives_list, new_row, output_data, result)

        f1z = new_row[0]

        """ epsilon <- f1(z) - 1 """
        model.epsilon = pyo.Param(initialize=f1z - 1, mutable=True)  # Epsilon parameter
        model.s = pyo.Var(within=pyo.NonNegativeReals)  # s = epsilon - f1(x)
        l1 = f1z - 1  # lower bound for f1(x)
        model.lambda_value = pyo.Param(initialize=(1 / ((f1z - l1) * 10 ** 3)),
                                                      mutable=True)  # Lambda parameter

        solution_found = (result.solver.status == 'ok') and (
                result.solver.termination_condition == 'optimal')  # while loop control

        """ While exists x in X that makes f1(x) <= epsilon do """
        while solution_found:
            """ estimate a lambda value > 0 """
            algorithms_utils.modify_component(model, 'lambda_value',
                                              pyo.Param(initialize=(1 / ((f1z - l1) * 10 ** 3)), mutable=True))

            """ Solve epsilon constraint problem """
            """ z <- solve {min f2(x) - lambda * l, subject to f1(x) + l = epsilon} """
            algorithms_utils.modify_component(model, 'obj', pyo.Objective(
                rule=lambda m: model.epsilon_objective_2obj(m, obj2)))  # min f2(x) - lambda * l
            algorithms_utils.modify_component(model, 'epsilonConstraint', pyo.Constraint(
                rule=lambda m: model.epsilon_constraint_2obj(m, obj1)))  # f1(x) + l = epsilon

            concrete, result = algorithms_utils.concrete_and_solve_model(model,
                                                                         data)  # Solve problem

            """ Checks if exists x in X that makes f1(x) <= epsilon (if exists solution) """
            if (result.solver.status == 'ok') and (result.solver.termination_condition == 'optimal'):
                """ PF = PF U {z} """
                f1z = pyo.value(obj1(concrete))
                new_row = [round(pyo.value(obj(concrete))) for obj in objectives_list]
                results_csv.append(new_row)
                print(f"New solution: {new_row}.")

                complete_data_new_row = algorithms_utils.write_complete_info(concrete, result, data_dict)
                complete_data.append(complete_data_new_row)

                """ epsilon = f1(z) - 1 """
                algorithms_utils.modify_component(model, 'epsilon',
                                                  pyo.Param(initialize=f1z - 1, mutable=True))

                # lower bound for f1(x) (it has to decrease with f1z)
                l1 = f1z - 1
                add_result_to_output_data_file(concrete, objectives_list, new_row, output_data, result)

                output_data.append('===============================================================================')
                if result.solver.status == 'ok':
                    for i, obj in enumerate(objectives_list):
                        output_data.append(f'Objective {obj.__name__}: {complete_data_new_row[i]}')
                    output_data.append('Sequences selected:')
                    for s in concrete.S:
                        output_data.append(f"x[{s}] = {concrete.x[s].value}")

            solution_found = (result.solver.status == 'ok') and (result.solver.termination_condition == 'optimal')

    return results_csv, concrete, output_data, complete_data



def e_constraint_3objs(data_dict: dict, tau: int, objectives_list: list, model: pyo.AbstractModel):
    data = data_dict['data']

    complete_data = [["numberOfSequences", "numberOfVariables", "numberOfConstraints",
                      "initialComplexity", "solution (index,CC,LOC)", "offsets", "extractions",
                      "not_nested_solution", "not_nested_extractions",
                      "nested_solution", "nested_extractions",
                      "reductionComplexity", "finalComplexity",
                      "minExtractedLOC", "maxExtractedLOC", "meanExtractedLOC", "totalExtractedLOC", "nestedLOC",
                      "minReductionOfCC", "maxReductionOfCC", "meanReductionOfCC", "totalReductionOfCC", "nestedCC",
                      "minExtractedParams", "maxExtractedParams", "meanExtractedParams", "totalExtractedParams",
                      "terminationCondition", "executionTime"]]

    p = len(objectives_list)

    output_data = []
    results_csv = [[obj.__name__ for obj in objectives_list]]

    model.tau = pyo.Param(within=pyo.NonNegativeReals, initialize=tau, mutable=False)  # Threshold

    """ Obtain payoff table by the lexicographic optimization of the objective functions """
    opt_lex_list = []
    opt_lex_table = []

    model.obj = pyo.Objective(rule=lambda m: objectives_list[0](m))

    concrete = None

    print(f"--------------------------------------------------------------------------------")
    for i, objective in enumerate(objectives_list):
        algorithms_utils.modify_component(model, 'obj',
                                          pyo.Objective(rule=lambda m: objective(m)))  # new objective: min f2(x)

        concrete, result = algorithms_utils.concrete_and_solve_model(model, data)

        if (result.solver.status == 'ok') and (result.solver.termination_condition == 'optimal'):
            opt_lex = [round(pyo.value(opt(concrete))) for opt in objectives_list]

            opt_lex_table.append(opt_lex)

            f_nz = round(pyo.value(objective(concrete)))  # f_n(z) := f_nz

            opt_lex_list.append(f_nz)

            print(f"min f{i + 1}(x), {objective.__name__}. Result obtained: f{i + 1}(z) = {round(f_nz)}.")

            attr_name = f'f{i + 1}z_constraint'

            def make_rule(obj, r):
                return lambda m: obj(m) <= r

            setattr(
                model,
                attr_name,
                pyo.Constraint(rule=make_rule(objective, f_nz))
            )

    opt_lex_point = tuple(opt_lex_list)
    print(f"--------------------------------------------------------------------------------")
    print(f"Lexicographic optima list: {opt_lex_point}.")

    for i in range(p):
        model.del_component(f'f{i + 1}z_constraint')  # delete f_n(x) <= f_n(z) constraint

    """ Set upper bounds ub_k for k=2...p """
    ub_dict = {model.extractions_objective: len(concrete.S) + 1,
               model.cc_difference_objective: concrete.nmcc[0] + 1,
               model.loc_difference_objective: concrete.loc[0] + 1}

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

    model.s = pyo.Var(range(p), domain=pyo.NonNegativeReals)

    solutions_set = set()

    for i in range(ub_point[1], 0, -1):
        j = ub_point[2]
        # for j in range(ub_point[2], 0, -1):
        while j > 0:
            print("==================================")
            print(f"[i,j] for e-constraint: [{i},{j}].")
            e_const = [i, j]
            concrete, result, feasible = solve_e_constraint(objectives_list, model, e_const, data)

            cplex_time = result.solver.time

            # concrete.pprint()

            if feasible:
                new_sol = [round(pyo.value(obj(concrete))) for obj in objectives_list]

                desired_order_for_objectives = ['extractions_objective', 'cc_difference_objective',
                                                'loc_difference_objective']
                objectives_dict = {obj.__name__: obj for obj in objectives_list}
                ordered_objectives = [objectives_dict[name] for name in desired_order_for_objectives if
                                      name in objectives_dict]
                ordered_newrow = tuple(round(pyo.value(obj(concrete))) for obj in ordered_objectives)

                new_sol_tuple = tuple(new_sol)

                dominated = False
                for sol in solutions_set:
                    if dominates(sol, new_sol_tuple):
                        dominated = True

                j = new_sol_tuple[-1]

                if dominated:
                    print(f"Dominated solution.")
                    continue

                if new_sol_tuple not in solutions_set:
                    print(f"New solution found: {tuple(ordered_newrow)}.")
                    solutions_set.add(new_sol_tuple)

                    complete_data_new_row = algorithms_utils.write_complete_info(concrete, result, data_dict)
                    complete_data.append(complete_data_new_row)

                    output_data.append(
                        '===============================================================================')
                    if result.solver.status == 'ok':
                        for k, obj in enumerate(objectives_list):
                            output_data.append(f'Objective {obj.__name__}: {complete_data_new_row[k]}')
                        output_data.append('Sequences selected:')
                        for s in concrete.S:
                            output_data.append(f"x[{s}] = {concrete.x[s].value}")

                    output_data.append(f"CPLEX time: {cplex_time}.")

                else:
                    print(f"Repeated solution: {tuple(ordered_newrow)}.")

                    output_data.append(
                        "======================================================================================")
                    output_data.append(f"Repeated solution, CPLEX TIME: {cplex_time}")
                    output_data.append(
                        "======================================================================================")

            else:
                print(f"Infeasible.")

                output_data.append(
                    "======================================================================================")
                output_data.append(f"SOLUTION NOT FOUND, CPLEX TIME: {cplex_time}")
                output_data.append(
                    "======================================================================================")

                j = 0

    for sol in solutions_set:
        results_csv.append(list(sol))

    print(f"Solutions: {results_csv}.")

    return results_csv, concrete, output_data, complete_data



def solve_e_constraint(objectives_list: list, model:pyo.AbstractModel, e, data):
    eps = 1 / (10 ** 4)

    def make_objective(obj):
        return lambda m: obj(m) - eps * sum(m.s[i] for i in range(1, len(objectives_list)))

    algorithms_utils.modify_component(model, 'obj',
                                      pyo.Objective(rule=make_objective(objectives_list[0])))

    for k, objective in enumerate(objectives_list[1:]):
        attr_name = f'f{k + 1}z_constraint_eps_problem'

        def make_rule(itr, obj, ep):
            return lambda m: obj(m) + m.s[itr + 1] == ep

        algorithms_utils.modify_component(model, attr_name, pyo.Constraint(
            rule=make_rule(k, objective, e[k] - 1)))

    concrete, result = algorithms_utils.concrete_and_solve_model(model, data)

    # concrete.pprint()

    solution_found = (result.solver.status == 'ok') and (result.solver.termination_condition == 'optimal')

    return concrete, result, solution_found


def dominates(a: tuple, b: tuple) -> bool:
    """
    Returns True if point a dominates b.
    """
    return all(a[i] <= b[i] for i in range(len(a))) and any(a[i] < b[i] for i in range(len(a)))


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
