import pyomo.environ as pyo # ayuda a definir y resolver problemas de optimización

import numpy as np
import pandas as pd
import os

import utils.algorithms_utils as algorithm_utils

from ILP_CC_reducer.models import ILPmodelRsain
from ILP_CC_reducer.models import MultiobjectiveILPmodel
from ILP_CC_reducer.algorithm.algorithm import Algorithm


class ObtainResultsAlgorithm(Algorithm):

    @staticmethod
    def get_name() -> str:
        return 'Solve Model with one objective'
    
    @staticmethod
    def get_description() -> str:
        return "It obtains the solution for single-objective ILP problem."

    @staticmethod
    def execute(data_dict: dict, tau:int, info_dict: dict) -> list[str]:

        folders_data = info_dict.get("folders_data")
        objective = info_dict.get("objective")
        just_model = info_dict.get("just_model")

        if objective.__name__ == 'extractions_objective':
            model = ILPmodelRsain()
        else:
            model = MultiobjectiveILPmodel()

        data_row = []

        algorithm_utils.modify_component(model,'tau',
                                       pyo.Param(within=pyo.NonNegativeReals, initialize=tau, mutable=False))

        data_row.append(folders_data["project"])
        data_row.append(folders_data["class"])
        data_row.append(folders_data["method"])
        
        missing_file = ""
        empty_file = ""
                
        if "missingFiles" in data_dict:
            missing_file = " and ".join(data_dict["missingFiles"]) if len(data_dict["missingFiles"]) > 1 else "".join(data_dict["missingFiles"])
        if "emptyFiles" in data_dict:
            empty_file = " and ".join(data_dict["emptyFiles"]) if len(data_dict["emptyFiles"]) > 1 else "".join(data_dict["emptyFiles"])
        
        data_row.append(missing_file)
        data_row.append(empty_file)

        if not objective:
            objective = model.extractions_objective

        algorithm_utils.modify_component(model, 'obj', pyo.Objective(rule=lambda m: objective(m)))
    
        # Verify if all files were found
        if len(data_dict.get("missingFiles")) == 0 and "sequences" not in data_dict["emptyFiles"]:
            
            concrete = model.create_instance(data_dict["data"]) # create a model instance and make it concrete
            
            solver = pyo.SolverFactory('cplex')
            solver.options["timelimit"] = 3600 # time limit for solver

            if not os.path.exists("models"):
                os.makedirs("models")
            # Save model in a .lp file before solving it
            concrete.write(f'models/{folders_data["class"]}-{folders_data["method"]}.lp',
                           io_options={'symbolic_solver_labels': True})
            print(f"Model correctly saved as {folders_data["class"]}-{folders_data["method"]}.lp.")

            num_extractions = len([s for s in concrete.S])
            print(f"There are {num_extractions} x[i] variables")
            data_row.append(num_extractions)

            num_constraints = sum(
                len(constraint) for constraint in concrete.component_objects(pyo.Constraint, active=True))
            print(f"There are {num_constraints} constraints")
            data_row.append(num_constraints)

            if not just_model:

                results = solver.solve(concrete)

                num_used_vars = results.Problem[0].number_of_variables
                data_row.append(num_used_vars)

                if (results.solver.status == 'ok') and (results.solver.termination_condition == 'optimal'):
                    print('Optimal solution found')
                    print('Objective value: ', pyo.value(concrete.obj))
                    print('Sequences selected:')
                    for s in concrete.S:
                        print(f"x[{s}] = {concrete.x[s].value}")



                    initial_complexity = concrete.nmcc[0]
                    data_row.append(initial_complexity)

                    # SOLUTION
                    solution = [concrete.x[s].index() for s in concrete.S if concrete.x[s].value == 1 and s != 0]
                    data_row.append(solution)
                    print(f"SOLUTION: {solution}")


                    # OFFSETS
                    df_csv = pd.read_csv(data_dict["offsets"], header=None, names=["index", "start", "end"])

                    # Filter by index in solution str list and obtain values
                    solution_str = [str(i) for i in solution]
                    offsets_list = df_csv[df_csv["index"].isin(solution_str)][["start", "end"]].values.tolist()

                    offsets_list = [[int(start), int(end)] for start, end in offsets_list]
                    data_row.append(offsets_list)


                    # EXTRACTIONS
                    extractions = len(solution)
                    data_row.append(extractions)

                    # NOT NESTED SOLUTION
                    not_nested_solution = [concrete.x[s].index() for s,k in concrete.N if k == 0 and concrete.z[s,k].value != 0]
                    data_row.append(not_nested_solution)

                    # NOT NESTED EXTRACTIONS
                    not_nested_extractions = len(not_nested_solution)
                    data_row.append(not_nested_extractions)

                    # NESTED SOLUTION
                    nested_solution = {}

                    for s, k in concrete.N:
                        if concrete.z[s, k].value != 0 and k in solution:
                            if k not in nested_solution:
                                nested_solution[k] = []  # Crear una nueva lista para cada k
                            nested_solution[k].append(concrete.x[s].index())


                    if len(nested_solution) != 0:
                        data_row.append(nested_solution)
                    else:
                        data_row.append("")



                    # NESTED EXTRACTIONS
                    nested_extractions = sum(len(v) for v in nested_solution.values())
                    data_row.append(nested_extractions)

                    # CC REDUCTION
                    cc_reduction = [(concrete.ccr[j, k] * concrete.z[j, k].value) for j,k in concrete.N if k == 0 and concrete.z[j,k].value != 0]

                    reduction_complexity = sum(cc_reduction)
                    data_row.append(reduction_complexity)

                    # FINAL COMPLEXITY
                    final_complexity = initial_complexity - reduction_complexity
                    data_row.append(final_complexity)

                    # LOC FOR EACH SEQUENCE: m.loc[i] * m.x[i] - sum(m.loc[j] * m.z[j, k] for j,k in m.N if k == i)
                    loc_for_each_sequence = [(concrete.loc[j] * concrete.z[j, k].value) for j,k in concrete.N if k == 0 and concrete.z[j,k].value != 0]
                    if len(loc_for_each_sequence) > 0:
                        min_extracted_loc = min(loc_for_each_sequence)
                        data_row.append(min_extracted_loc)
                        max_extracted_loc = max(loc_for_each_sequence)
                        data_row.append(max_extracted_loc)
                        mean_extracted_loc = round(float(np.mean(loc_for_each_sequence)))
                        data_row.append(mean_extracted_loc)
                        total_extracted_loc = sum(loc_for_each_sequence)
                        data_row.append(total_extracted_loc)
                        # NESTED LOC
                        nested_loc = {}
                        for v in nested_solution.values():
                            for n in v:
                                nested_loc[n] = concrete.loc[n]
                        if len(nested_loc) > 0:
                            data_row.append(nested_loc)
                        else:
                            data_row.append("")
                    else:
                        for _ in range(5):
                            data_row.append("")




                    # CC FOR EACH SEQUENCE
                    if len(cc_reduction) > 0:
                        min_extracted_cc = min(cc_reduction)
                        data_row.append(min_extracted_cc)
                        max_extracted_cc = max(cc_reduction)
                        data_row.append(max_extracted_cc)
                        mean_extracted_cc = round(float(np.mean(cc_reduction)))
                        data_row.append(mean_extracted_cc)
                        total_extracted_cc = reduction_complexity
                        data_row.append(total_extracted_cc)
                        # NESTED CC
                        nested_cc = {}
                        for v in nested_solution.values():
                            for n in v:
                                nested_cc[n] = concrete.nmcc[n]
                        if len(nested_cc) > 0:
                            data_row.append(nested_cc)
                        else:
                            data_row.append("")
                    else:
                        for _ in range(5):
                            data_row.append("")

                    params_for_each_sequence = [(concrete.params[j] * concrete.z[j, k].value) for j,k in concrete.N if concrete.z[j,k].value != 0]
                    if len(params_for_each_sequence) > 0:
                        min_extracted_params = min(params_for_each_sequence)
                        data_row.append(min_extracted_params)
                        max_extracted_params = max(params_for_each_sequence)
                        data_row.append(max_extracted_params)
                        mean_extracted_params = round(float(np.mean(params_for_each_sequence)))
                        data_row.append(mean_extracted_params)
                        total_extracted_params = sum(params_for_each_sequence)
                        data_row.append(total_extracted_params)
                    else:
                        for _ in range(4):
                            data_row.append("")
                else:
                    for _ in range(24):
                        data_row.append("")

                data_row.append(str(results.solver.status))
                data_row.append(str(results.solver.termination_condition))
                data_row.append(results.solver.time)

                print(data_row)
        
        return data_row
    




