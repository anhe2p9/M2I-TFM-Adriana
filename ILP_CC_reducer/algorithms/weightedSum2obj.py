import pyomo.environ as pyo # ayuda a definir y resolver problemas de optimización
import pyomo.dataportal as dp # permite cargar datos para usar en esos modelos de optimización

import sys
import algorithms_utils

# import csv
# import os

from ILP_CC_reducer.Algorithm.Algorithm import Algorithm
from ILP_CC_reducer.models import MultiobjectiveILPmodel
from openpyxl.styles.builtins import output


class WeightedSumAlgorithm2obj(Algorithm):

    @staticmethod
    def get_name() -> str:
        return 'Weigthed Sum Algorithm'
    
    @staticmethod
    def get_description() -> str:
        return ("It obtains soported ILP solutions based on the given weights.")

    @staticmethod
    def execute(data: dp.DataPortal, tau: int, subs, obj2) -> None:
        
        csv_data = [["Weight1","Weight2 (CCdiff)","Num.sequences","CCdif"]]
        output_data = []
        
        multiobj_model = MultiobjectiveILPmodel()
        multiobj_model.model.tau = pyo.Param(within=pyo.NonNegativeReals, initialize=tau, mutable=False) # Threshold
                
        
        output_data.append(f"ARGS WEIGHTED SUM 2 OBJS: {subs}, {obj2}")
        
        if isinstance(subs, int):
            output_data.append(f"Proccessing all ILP results with {subs} subdivisions")
        
            for i in range(subs+1):
                w1, w2 = algorithms_utils.generate_weights_2obj(subs, i)
               
                algorithms_utils.modify_component(multiobj_model, 'obj', pyo.Objective(rule=lambda m: multiobj_model.weightedSum2obj(m, w1, w2, obj2)))
                
                concrete, results = algorithms_utils.concrete_and_solve_model(multiobj_model, data)
                
                newrow = algorithms_utils.calculate_results(concrete, obj2)
                            
                # algorithms_utils.print_result_and_sequences(concrete, results.solver.status, newrow, obj2)
                
                output_data.append('===============================================================================')
                if (results.solver.status == 'ok'):
                    output_data.append(f'Objective SEQUENCES: {newrow[0]}')
                    output_data.append(f'Objective {obj2}: {newrow[1]}')
                    output_data.append('Sequences selected:')
                    for s in concrete.S:
                        output_data.append(f"x[{s}] = {concrete.x[s].value}")
                output_data.append('===============================================================================')
                
                
                
                csv_data.append(newrow)
                
        else:
            sys.exit(f'The Weighted Sum Algorithm parameters for two objectives must be a number of subdivisions s and the second objective.')
            
        return csv_data, concrete, output_data



    




