import pyomo.environ as pyo # ayuda a definir y resolver problemas de optimización
import pyomo.dataportal as dp # permite cargar datos para usar en esos modelos de optimización

import sys
import algorithms_utils

from typing import Any

# import csv
# import os

from ILP_CC_reducer.Algorithm.Algorithm import Algorithm
from ILP_CC_reducer.models import MultiobjectiveILPmodel


class WeightedSumAlgorithm2obj(Algorithm):

    @staticmethod
    def get_name() -> str:
        return 'Weigthed Sum Algorithm'
    
    @staticmethod
    def get_description() -> str:
        return ("It obtains soported ILP solutions based on the given weights.")

    @staticmethod
    def execute(data: dp.DataPortal, tau: int, weights_config: Any, objectives_list: list) -> None:
        
        multiobj_model = MultiobjectiveILPmodel()
        
        obj1 = objectives_list[0]
        obj2 = objectives_list[1]
        
        csv_data = [[f"Weight1_{obj1.__name__}",f"Weight2_{obj2.__name__}",obj1.__name__,obj2.__name__]]
        
        output_data = []
        
        multiobj_model.model.tau = pyo.Param(within=pyo.NonNegativeReals, initialize=tau, mutable=False) # Threshold
        
        if isinstance(weights_config, int):
            print(f"Proccessing all ILP results with {weights_config} subdivisions...")
        
            for i in range(weights_config+1):
                w1, w2 = algorithms_utils.generate_two_weights(weights_config, i)
               
                algorithms_utils.modify_component(multiobj_model, 'obj', pyo.Objective(rule=lambda m: multiobj_model.weightedSum2obj(m, w1, w2, obj1, obj2)))
                concrete, results = algorithms_utils.concrete_and_solve_model(multiobj_model, data)
                
                newrow = algorithms_utils.calculate_results(concrete, obj2)
                
                # algorithms_utils.print_result_and_sequences(concrete, results.solver.status, newrow, obj2)
                
                algorithms_utils.add_info_to_list(concrete, output_data, results.solver.status, obj1, obj2, newrow)
                
                newrow = [round(w1,3),round(w2,3)] + newrow
                csv_data.append(newrow)
                
        elif all(isinstance(w, float) for w in weights_config):
            print(f"Proccessing ILP results with weights: {weights_config}...")
            w1, w2 = weights_config
            
            algorithms_utils.modify_component(multiobj_model, 'obj', pyo.Objective(rule=lambda m: multiobj_model.weightedSum2obj(m, w1, w2, obj1, obj2)))
            concrete, results = algorithms_utils.concrete_and_solve_model(multiobj_model, data)
            
            newrow = algorithms_utils.calculate_results(concrete, obj2)
            csv_data.append(newrow)
            
            output_data.append('===============================================================================')
            if (results.solver.status == 'ok'):
                output_data.append(f'{obj1.__name__}: {newrow[0]}')
                output_data.append(f'{obj2.__name__}: {newrow[1]}')
                output_data.append('Sequences selected:')
                for s in concrete.S:
                    output_data.append(f"x[{s}] = {concrete.x[s].value}")
            output_data.append('===============================================================================')
            
                
            
        else:
            sys.exit(f'The Weighted Sum Algorithm parameters for two objectives must be a number of subdivisions s and the second objective.')
            
        return csv_data, concrete, output_data



    




