import pyomo.environ as pyo # ayuda a definir y resolver problemas de optimización

import numpy as np
import pandas as pd
import os

from ILP_CC_reducer.models import ILPmodelRsain
from ILP_CC_reducer.models import MultiobjectiveILPmodel
from ILP_CC_reducer.algorithm.algorithm import Algorithm

import general_utils




class ObtainResultsAlgorithm(Algorithm):

    @staticmethod
    def get_name() -> str:
        return 'Solve Model with one objective'
    
    @staticmethod
    def get_description() -> str:
        return ("It obtains the solution for just sequences objective.")

    @staticmethod
    def execute(data: dict, tau:int, folders_data: dict, objective: str=None) -> list[str]:

        if objective.__name__ == 'sequences_objective':
            model = ILPmodelRsain()
        else:
            model = MultiobjectiveILPmodel()

        data_row = []

        general_utils.modify_component(model,'tau',
                                       pyo.Param(within=pyo.NonNegativeReals, initialize=tau, mutable=False))

        data_row.append(folders_data["project"])
        data_row.append(folders_data["class"])
        data_row.append(folders_data["method"])
        
        missing_file = ""
        empty_file = ""
                
        if "missingFiles" in data:
            missing_file = " and ".join(data["missingFiles"]) if len(data["missingFiles"]) > 1 else "".join(data["missingFiles"])
        if "emptyFiles" in data:    
            empty_file = " and ".join(data["emptyFiles"]) if len(data["emptyFiles"]) > 1 else "".join(data["emptyFiles"])
        
        data_row.append(missing_file)
        data_row.append(empty_file)

        if not objective:
            objective = model.sequences_objective

        general_utils.modify_component(model, 'obj', pyo.Objective(rule=lambda m: objective(m)))
    
        # Verificar si todos los archivos han sido encontrados
        if len(data.get("missingFiles")) == 0 and "sequences" not in data["emptyFiles"]:
            
            concrete = model.create_instance(data["data"]) # para crear una instancia de modelo y hacerlo concreto
            
            solver = pyo.SolverFactory('cplex')
            solver.options["timelimit"] = 3600 # time limit for solver
            results = solver.solve(concrete)

            if not os.path.exists("models"):
                os.makedirs("models")
            # Guardar el modelo en un archivo .lp antes de resolverlo
            concrete.write(f'models/{folders_data["class"]}-{folders_data["method"]}.lp', io_options={'symbolic_solver_labels': True})
        
        
            num_sequences = len([s for s in concrete.S])
            print(f"There are {num_sequences} x[i] variables")
            data_row.append(num_sequences)
            
            
            # num_variables = sum(1 for v in concrete.component_data_objects(pyo.Var, active=True) if v.active)
            # print(f"There are {num_variables} variables")
            # data_row.append(num_variables)
            
            
            num_vars_utilizadas = results.Problem[0].number_of_variables
            data_row.append(num_vars_utilizadas)
            
            num_constraints = sum(len(constraint) for constraint in concrete.component_objects(pyo.Constraint, active=True))
            print(f"There are {num_constraints} constraints")
            data_row.append(num_constraints)
            
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
                df_csv = pd.read_csv(data["offsets"], header=None, names=["index", "start", "end"])
            
                # Filter by intex in solution str list and obtain values
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
                CC_reduction = [(concrete.ccr[j, k] * concrete.z[j, k].value) for j,k in concrete.N if k == 0 and concrete.z[j,k].value != 0]
                
                reduction_complexity = sum(CC_reduction)
                data_row.append(reduction_complexity)
                
                # FINAL COMPLEXITY
                final_complexity = initial_complexity - reduction_complexity
                data_row.append(final_complexity)
                
                # LOC FOR EACH SEQUENCE: m.loc[i] * m.x[i] - sum(m.loc[j] * m.z[j, k] for j,k in m.N if k == i)
                LOC_for_each_sequence = [(concrete.loc[j] * concrete.z[j, k].value) for j,k in concrete.N if k == 0 and concrete.z[j,k].value != 0]
                if len(LOC_for_each_sequence) > 0:
                    minExtractedLOC = min(LOC_for_each_sequence)
                    data_row.append(minExtractedLOC)
                    maxExtractedLOC = max(LOC_for_each_sequence)
                    data_row.append(maxExtractedLOC)
                    meanExtractedLOC = round(float(np.mean(LOC_for_each_sequence)))
                    data_row.append(meanExtractedLOC)
                    totalExtractedLOC = sum(LOC_for_each_sequence)
                    data_row.append(totalExtractedLOC)
                    # NESTED LOC
                    nested_LOC = {}
                    for v in nested_solution.values():
                        for n in v:
                            nested_LOC[n] = concrete.loc[n]
                    if len(nested_LOC) > 0:
                        data_row.append(nested_LOC)
                    else:
                        data_row.append("")
                else:
                    for _ in range(5):
                        data_row.append("")
                        
                
                
                
                # CC FOR EACH SEQUENCE
                if len(CC_reduction) > 0:
                    minExtractedCC = min(CC_reduction)
                    data_row.append(minExtractedCC)
                    maxExtractedCC = max(CC_reduction)
                    data_row.append(maxExtractedCC)
                    meanExtractedCC = round(float(np.mean(CC_reduction)))
                    data_row.append(meanExtractedCC)
                    totalExtractedCC = reduction_complexity
                    data_row.append(totalExtractedCC)
                    # NESTED CC
                    nested_CC = {}
                    for v in nested_solution.values():
                        for n in v:
                            nested_CC[n] = concrete.nmcc[n]
                    if len(nested_CC) > 0:
                        data_row.append(nested_CC)
                    else:
                        data_row.append("")
                else:
                    for _ in range(5):
                        data_row.append("")
                        
                PARAMS_for_each_sequence = [(concrete.params[j] * concrete.z[j, k].value) for j,k in concrete.N if concrete.z[j,k].value != 0]
                if len(PARAMS_for_each_sequence) > 0:
                    minExtractedPARAMS = min(PARAMS_for_each_sequence)
                    data_row.append(minExtractedPARAMS)
                    maxExtractedPARAMS = max(PARAMS_for_each_sequence)
                    data_row.append(maxExtractedPARAMS)
                    meanExtractedPARAMS = round(float(np.mean(PARAMS_for_each_sequence)))
                    data_row.append(meanExtractedPARAMS)
                    totalExtractedPARAMS = sum(PARAMS_for_each_sequence)
                    data_row.append(totalExtractedPARAMS)
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
    




